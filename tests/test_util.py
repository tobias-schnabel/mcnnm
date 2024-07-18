import pytest
from typing import Optional, Tuple
import jax.numpy as jnp
from mcnnm.util import *
import jax

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)

key = jax.random.PRNGKey(2024)

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

def test_frobenius_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(frobenius_norm(A), jnp.sqrt(30))

def test_nuclear_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(nuclear_norm(A), jnp.sum(jnp.linalg.svd(A, compute_uv=False)))

def test_element_wise_l1_norm():
    A = jnp.array([[1, -2], [-3, 4]])
    assert jnp.allclose(element_wise_l1_norm(A), 10)

def test_propose_lambda():
    lambdas = propose_lambda(1.0, 5)
    assert len(lambdas) == 5
    assert jnp.allclose(lambdas[0], 0.01)
    assert jnp.allclose(lambdas[-1], 100.0)


def test_check_inputs():
    Y = jnp.array([[1, 2], [3, 4]])
    W = jnp.array([[0, 1], [0, 0]])
    X, Z, V, Omega = check_inputs(Y, W)
    assert X.shape == (2, 0)
    assert Z.shape == (2, 0)
    assert V.shape == (2, 2, 0)
    assert Omega.shape == (2, 2)
