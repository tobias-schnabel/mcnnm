import pytest
import jax.numpy as jnp
from mcnnm.core_utils import mask_observed, mask_unobserved, frobenius_norm, nuclear_norm
from mcnnm.core_utils import element_wise_l1_norm, shrink_lambda, initialize_coefficients
import jax
from jax import random

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


def test_mask_observed(sample_data):
    Y, W, _, _, _ = sample_data
    mask = jnp.where(W == 0, True, False)
    assert jnp.allclose(mask_observed(Y, mask), Y * mask)


def test_mask_observed_shape_mismatch():
    A = jnp.array([[1, 2], [3, 4]])
    mask = jnp.array([[True, False]])
    with pytest.raises(ValueError):
        mask_observed(A, mask)


def test_mask_unobserved(sample_data):
    Y, W, _, _, _ = sample_data
    mask = jnp.where(W == 0, True, False)
    assert jnp.allclose(mask_unobserved(Y, mask), Y * (1 - mask))


def test_mask_unobserved_shape_mismatch():
    A = jnp.array([[1, 2], [3, 4]])
    mask = jnp.array([[True, False]])
    with pytest.raises(ValueError):
        mask_unobserved(A, mask)


def test_shrink_lambda(sample_data):
    Y, _, _, _, _ = sample_data
    lambda_ = 0.1
    Y_shrunk = shrink_lambda(Y, lambda_)
    assert Y_shrunk.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y_shrunk))


def test_frobenius_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(frobenius_norm(A), jnp.sqrt(30))


def test_frobenius_norm_non_2d():
    A = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be a 2D array."):
        frobenius_norm(A)


def test_nuclear_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(nuclear_norm(A), jnp.sum(jnp.linalg.svd(A, compute_uv=False)))


def test_nuclear_norm_non_2d():
    A = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be a 2D array."):
        nuclear_norm(A)


def test_element_wise_l1_norm():
    A = jnp.array([[1, -2], [-3, 4]])
    assert jnp.allclose(element_wise_l1_norm(A), 10)


def test_element_wise_l1_norm_non_2d():
    A = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be a 2D array."):
        element_wise_l1_norm(A)


def test_initialize_coefficients_shape():
    Y = jnp.zeros((10, 5))
    X = jnp.zeros((10, 3))
    Z = jnp.zeros((5, 2))
    V = jnp.zeros((10, 5, 4))
    L, H, gamma, delta, beta = initialize_coefficients(Y, X, Z, V)
    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (max(V.shape[2], 1),)


def test_initialize_coefficients_empty():
    Y = jnp.zeros((0, 0))
    X = jnp.zeros((0, 0))
    Z = jnp.zeros((0, 0))
    V = jnp.zeros((0, 0, 0))
    L, H, gamma, delta, beta = initialize_coefficients(Y, X, Z, V)
    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (max(V.shape[2], 1),)


def test_initialize_coefficients_single_element():
    Y = jnp.zeros((1, 1))
    X = jnp.zeros((1, 1))
    Z = jnp.zeros((1, 1))
    V = jnp.zeros((1, 1, 1))
    L, H, gamma, delta, beta = initialize_coefficients(Y, X, Z, V)
    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (max(V.shape[2], 1),)
