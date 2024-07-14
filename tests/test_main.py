import pytest
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple
from src.mcnnm.main import p_o, p_perp_o, shrink_lambda, objective_function, compute_fixed_effects, compute_H, compute_L, cross_validation, compute_treatment_effect, fit
from src.mcnnm.util import timer, time_fit

# Set a fixed seed for reproducibility
key = random.PRNGKey(2024)

@pytest.fixture
def sample_data():
    N, T = 10, 5
    Y = random.normal(key, (N, T))
    W = random.bernoulli(key, 0.2, (N, T))
    X = random.normal(key, (N, T))
    return Y, W, X

def test_p_o(sample_data):
    Y, W, _ = sample_data
    mask = (W == 0)
    assert jnp.allclose(p_o(Y, mask), Y * mask)

def test_p_perp_o(sample_data):
    Y, W, _ = sample_data
    mask = (W == 0)
    assert jnp.allclose(p_perp_o(Y, mask), Y * (1 - mask))

def test_shrink_lambda(sample_data):
    Y, _, _ = sample_data
    lambda_ = 0.1
    Y_shrunk = shrink_lambda(Y, lambda_)
    assert Y_shrunk.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y_shrunk))

def test_objective_function(sample_data):
    Y, W, X = sample_data
    L = random.normal(key, Y.shape)
    obj_value = objective_function(Y, L, X=X)
    assert jnp.isscalar(obj_value)
    assert jnp.isfinite(obj_value)

def test_compute_fixed_effects(sample_data):
    Y, _, X = sample_data
    L = random.normal(key, Y.shape)
    gamma, delta = compute_fixed_effects(Y, L, X=X)
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))

def test_compute_H(sample_data):
    Y, _, X = sample_data
    L = random.normal(key, Y.shape)
    H = compute_H(Y, L, X=X)
    assert H.shape == (X.shape[0], X.shape[1])
    assert jnp.all(jnp.isfinite(H))

def test_compute_L(sample_data):
    Y, W, X = sample_data
    lambda_L = 0.1
    L = compute_L(Y, (W == 0), lambda_L, X=X)
    assert L.shape == Y.shape
    assert jnp.all(jnp.isfinite(L))

def test_cross_validation(sample_data):
    Y, W, X = sample_data
    lambda_L, lambda_H = cross_validation(Y, W, X)
    assert jnp.isscalar(lambda_L)
    assert jnp.isscalar(lambda_H)
    assert jnp.isfinite(lambda_L)
    assert jnp.isfinite(lambda_H)

def test_compute_treatment_effect(sample_data):
    Y, W, X = sample_data
    L = random.normal(key, Y.shape)
    gamma = random.normal(key, (Y.shape[0],))
    delta = random.normal(key, (Y.shape[1],))
    beta = random.normal(key, Y.shape)
    H = random.normal(key, (X.shape[0], X.shape[1]))
    tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W)
    assert jnp.isscalar(tau)
    assert jnp.isfinite(tau)

def test_fit(sample_data):
    Y, W, X = sample_data
    tau, lambda_L, L, Y_completed = fit(Y, W, X=X)
    assert jnp.isscalar(tau)
    assert jnp.isscalar(lambda_L)
    assert L.shape == Y.shape
    assert Y_completed.shape == Y.shape
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(Y_completed))

def test_timer():
    @timer
    def dummy_function():
        return 42

    assert dummy_function() == 42

def test_time_fit(sample_data):
    Y, W, X = sample_data
    result = time_fit(Y, W, X=X)
    assert isinstance(result, tuple)