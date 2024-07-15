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
    results = fit(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True)

    assert len(results) == 8, f"Expected 8 return values from fit function, got {len(results)}"
    tau, lambda_L, L, Y_completed, gamma, delta, beta, H = results

    assert isinstance(tau, (float, jnp.ndarray)), f"tau should be a float or JAX array, got {type(tau)}"
    if isinstance(tau, jnp.ndarray):
        assert tau.shape == (), f"tau should be a scalar, got shape {tau.shape}"
    assert isinstance(lambda_L, (float, jnp.ndarray)), f"lambda_L should be a float or JAX array, got {type(lambda_L)}"
    assert isinstance(L, jnp.ndarray), f"L should be a JAX array, got {type(L)}"
    assert isinstance(Y_completed, jnp.ndarray), f"Y_completed should be a JAX array, got {type(Y_completed)}"
    assert isinstance(gamma, jnp.ndarray), f"gamma should be a JAX array, got {type(gamma)}"
    assert isinstance(delta, jnp.ndarray), f"delta should be a JAX array, got {type(delta)}"
    assert isinstance(beta, jnp.ndarray), f"beta should be a JAX array, got {type(beta)}"
    assert isinstance(H, jnp.ndarray), f"H should be a JAX array, got {type(H)}"

    assert L.shape == Y.shape, f"L shape {L.shape} should match Y shape {Y.shape}"
    assert Y_completed.shape == Y.shape, f"Y_completed shape {Y_completed.shape} should match Y shape {Y.shape}"
    assert gamma.shape == (Y.shape[0],), f"gamma shape {gamma.shape} should be ({Y.shape[0]},)"
    assert delta.shape == (Y.shape[1],), f"delta shape {delta.shape} should be ({Y.shape[1]},)"
    assert beta.shape == (V.shape[2],), f"beta shape {beta.shape} should be ({V.shape[2]},)"
    assert H.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[
        1]), f"H shape {H.shape} should be ({X.shape[1] + Y.shape[0]}, {Z.shape[1] + Y.shape[1]})"

    print("All assertions passed!")