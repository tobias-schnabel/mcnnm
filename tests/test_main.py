import pytest
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple
from mcnnm.main import p_o, p_perp_o, shrink_lambda, objective_function, compute_fixed_effects, compute_H, compute_L, cross_validation, compute_treatment_effect, fit

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

def test_p_o(sample_data):
    Y, W, _, _, _ = sample_data
    mask = (W == 0)
    assert jnp.allclose(p_o(Y, mask), Y * mask)

def test_p_perp_o(sample_data):
    Y, W, _, _, _ = sample_data
    mask = (W == 0)
    assert jnp.allclose(p_perp_o(Y, mask), Y * (1 - mask))

def test_shrink_lambda(sample_data):
    Y, _, _, _, _ = sample_data
    lambda_ = 0.1
    Y_shrunk = shrink_lambda(Y, lambda_)
    assert Y_shrunk.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y_shrunk))

def test_objective_function(sample_data):
    Y, W, X, Z, V = sample_data
    L = random.normal(key, Y.shape)
    H = random.normal(key, (Y.shape[0] + X.shape[1], Y.shape[1] + Z.shape[1]))
    beta = random.normal(key, (V.shape[2],))
    obj_value = objective_function(Y, L, beta=beta, H=H, X=X, Z=Z, V=V)
    assert obj_value.shape == ()
    assert jnp.isfinite(obj_value)

def test_compute_fixed_effects(sample_data):
    Y, _, X, Z, V = sample_data
    L = random.normal(key, Y.shape)
    H = random.normal(key, (Y.shape[0] + X.shape[1], Y.shape[1] + Z.shape[1]))
    beta = random.normal(key, (V.shape[2],))
    gamma, delta = compute_fixed_effects(Y, L, beta=beta, H=H, X=X, Z=Z, V=V)
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))

def test_compute_H(sample_data):
    Y, _, X, Z, V = sample_data
    L = random.normal(key, Y.shape)
    gamma = random.normal(key, (Y.shape[0],))
    delta = random.normal(key, (Y.shape[1],))
    beta = random.normal(key, (V.shape[2],))
    H = compute_H(Y, L, gamma, delta, beta=beta, X=X, Z=Z, V=V)
    assert H.shape == (Y.shape[0] + X.shape[1], Y.shape[1] + Z.shape[1])
    assert jnp.all(jnp.isfinite(H))

def test_compute_L(sample_data):
    Y, W, X, Z, V = sample_data
    lambda_L = 0.1
    gamma = random.normal(key, (Y.shape[0],))
    delta = random.normal(key, (Y.shape[1],))
    H = random.normal(key, (Y.shape[0] + X.shape[1], Y.shape[1] + Z.shape[1]))
    beta = random.normal(key, (V.shape[2],))
    L = compute_L(Y, (W == 0), lambda_L, gamma=gamma, delta=delta, beta=beta, H=H, X=X, Z=Z, V=V)
    assert L.shape == Y.shape
    assert jnp.all(jnp.isfinite(L))

# def test_cross_validation(sample_data):
#     Y, W, X, Z, V = sample_data
#     lambda_L, lambda_H = cross_validation(Y, W, X=X, Z=Z, V=V)
#     assert jnp.isscalar(lambda_L)
#     assert jnp.isscalar(lambda_H)
#     assert jnp.isfinite(lambda_L)
#     assert jnp.isfinite(lambda_H)
def test_cross_validation(sample_data):
    Y, W, X, Z, V = sample_data

    # Test with all inputs
    lambda_L, lambda_H = cross_validation(Y, W, X=X, Z=Z, V=V)
    assert jnp.isscalar(lambda_L) and jnp.isscalar(lambda_H)
    assert jnp.isfinite(lambda_L) and jnp.isfinite(lambda_H)
    assert lambda_L > 0 and lambda_H > 0

    # Test without optional inputs
    lambda_L, lambda_H = cross_validation(Y, W)
    assert jnp.isscalar(lambda_L) and jnp.isscalar(lambda_H)
    assert jnp.isfinite(lambda_L) and jnp.isfinite(lambda_H)

    # Test with proposed lambda values
    proposed_lambda_L, proposed_lambda_H = 0.1, 0.2
    lambda_L, lambda_H = cross_validation(Y, W, proposed_lambda_L=proposed_lambda_L,
                                          proposed_lambda_H=proposed_lambda_H)
    assert jnp.isscalar(lambda_L) and jnp.isscalar(lambda_H)
    assert jnp.isfinite(lambda_L) and jnp.isfinite(lambda_H)

    # Test with different number of folds
    lambda_L, lambda_H = cross_validation(Y, W, K=3)
    assert jnp.isscalar(lambda_L) and jnp.isscalar(lambda_H)
    assert jnp.isfinite(lambda_L) and jnp.isfinite(lambda_H)

def test_compute_treatment_effect(sample_data):
    Y, W, X, Z, V = sample_data
    L = random.normal(key, Y.shape)
    gamma = random.normal(key, (Y.shape[0],))
    delta = random.normal(key, (Y.shape[1],))
    beta = random.normal(key, (V.shape[2],))
    H = random.normal(key, (Y.shape[0] + X.shape[1], Y.shape[1] + Z.shape[1]))
    tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
    assert jnp.isfinite(tau)

def test_fit(sample_data):
    Y, W, X, Z, V = sample_data
    results = fit(Y, W, X=X, Z=Z, V=V)
    assert len(results) == 4
    tau, lambda_L, L, Y_completed = results
    assert jnp.isfinite(tau)
    assert jnp.isfinite(lambda_L)
    assert L.shape == Y.shape
    assert Y_completed.shape == Y.shape
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(Y_completed))