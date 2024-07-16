import pytest
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple
from mcnnm.main import objective_function, cross_validation, compute_treatment_effect, fit, MCNNMResults
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)

# Set a fixed seed for reproducibility
key = random.PRNGKey(2024)


@pytest.fixture
def sample_data():
    N, T, P, Q, J = 10, 5, 3, 2, 4
    Y = random.normal(key, (N, T))
    W = random.bernoulli(key, 0.2, (N, T))
    X = random.normal(key, (N, P))
    Z = random.normal(key, (T, Q))
    V = random.normal(key, (N, T, J))
    return Y, W, X, Z, V


def test_objective_function(sample_data):
    Y, W, X, Z, V = sample_data
    L = random.normal(key, Y.shape)
    H = random.normal(key, (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1]))
    gamma = random.normal(key, (Y.shape[0],))
    delta = random.normal(key, (Y.shape[1],))
    beta = random.normal(key, (V.shape[2],))
    obj_value = objective_function(Y, L, gamma=gamma, delta=delta, beta=beta, H=H, X=X, Z=Z, V=V)
    assert obj_value.shape == ()
    assert jnp.isfinite(obj_value)


@pytest.mark.timeout(30)
def test_cross_validation(sample_data):
    Y, W, X, Z, V = sample_data
    lambda_L, lambda_H = cross_validation(Y, W, X=X, Z=Z, V=V)
    assert jnp.isscalar(lambda_L)
    assert jnp.isscalar(lambda_H)
    assert jnp.isfinite(lambda_L)
    assert jnp.isfinite(lambda_H)


# def test_compute_treatment_effect(sample_data):
#     Y, W, X, Z, V = sample_data
#     L = random.normal(key, Y.shape)
#     gamma = random.normal(key, (Y.shape[0],))
#     delta = random.normal(key, (Y.shape[1],))
#     beta = random.normal(key, (V.shape[2],))
#     H = random.normal(key, (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1]))
#     tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
#     assert jnp.isfinite(tau)

def test_compute_treatment_effect(sample_data):
    Y, W, X, Z, V = sample_data
    N, T = Y.shape
    P = X.shape[1]
    Q = Z.shape[1]

    L = random.normal(key, Y.shape)
    gamma = random.normal(key, (N,))
    delta = random.normal(key, (T,))
    beta = random.normal(key, (V.shape[2],))

    # Correct H shape
    H = random.normal(key, (N + P, T + Q))

    print(f"Shape of X: {X.shape}")
    print(f"Shape of H: {H.shape}")
    print(f"Shape of Z: {Z.shape}")
    print(f"Shape of X_tilde: {jnp.hstack((X, jnp.eye(N))).shape}")
    print(f"Shape of Z_tilde: {jnp.hstack((Z, jnp.eye(T))).shape}")

    tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
    assert jnp.isfinite(tau)


@pytest.mark.timeout(30)
def test_fit(sample_data):
    Y, W, X, Z, V = sample_data
    results = fit(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True)

    assert len(results) == 9
    tau, lambda_L, lambda_H, L, Y_completed, gamma, delta, beta, H = results

    assert jnp.isfinite(tau)
    assert jnp.isfinite(lambda_L)
    assert L.shape == Y.shape
    assert Y_completed.shape == Y.shape
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (V.shape[2],)
    assert H.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1])

    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(Y_completed))
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(jnp.isfinite(beta))
    assert jnp.all(jnp.isfinite(H))
