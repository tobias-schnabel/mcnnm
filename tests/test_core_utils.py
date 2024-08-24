import pytest
import jax.numpy as jnp
from mcnnm.core_utils import is_positive_definite
from mcnnm.core_utils import mask_observed, mask_unobserved, frobenius_norm, nuclear_norm
from mcnnm.core_utils import element_wise_l1_norm, shrink_lambda, normalize, normalize_back
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


def test_is_positive_definite_positive_definite():
    @jax.jit
    def create_and_check_pd(key):
        A = random.normal(key, (3, 3))
        pd_matrix = A @ A.T + jnp.eye(3)  # Ensure it's symmetric and positive definite
        return is_positive_definite(pd_matrix)

    assert create_and_check_pd(key)


def test_is_positive_definite_negative_definite():
    @jax.jit
    def create_and_check_nd(key):
        A = random.normal(key, (3, 3))
        nd_matrix = -(A @ A.T + jnp.eye(3))
        return is_positive_definite(nd_matrix)

    assert not create_and_check_nd(key)


def test_is_positive_definite_positive_semidefinite():
    psd_matrix = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    assert not is_positive_definite(psd_matrix)


def test_is_positive_definite_symmetric_indefinite():
    indef_matrix = jnp.array([[1.0, 2.0], [2.0, -1.0]])
    assert not is_positive_definite(indef_matrix)


def test_is_positive_definite_identity():
    assert is_positive_definite(jnp.eye(3))


def test_is_positive_definite_zero():
    assert not is_positive_definite(jnp.zeros((3, 3)))


def test_is_positive_definite_single_element():
    assert is_positive_definite(jnp.array([[5.0]]))
    assert not is_positive_definite(jnp.array([[-5.0]]))


def test_is_positive_definite_almost_positive_definite():
    almost_pd = jnp.array([[1.0, 0.0], [0.0, 1e-10]])
    assert is_positive_definite(almost_pd)


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


def test_normalize_happy_path():
    mat = jnp.array([[1, 2], [3, 4]])
    mat_norm, col_norms = normalize(mat)
    assert mat_norm.shape == mat.shape
    assert col_norms.shape == (mat.shape[1],)
    assert jnp.allclose(jnp.linalg.norm(mat_norm, axis=0), 1)


def test_normalize_single_column():
    mat = jnp.array([[1], [3]])
    mat_norm, col_norms = normalize(mat)
    assert mat_norm.shape == mat.shape
    assert col_norms.shape == (mat.shape[1],)
    assert jnp.allclose(jnp.linalg.norm(mat_norm, axis=0), 1)


def test_normalize_empty_matrix():
    mat = jnp.zeros((0, 0))
    mat_norm, col_norms = normalize(mat)
    assert mat_norm.shape == mat.shape
    assert col_norms.shape == (0,)


def test_normalize_zero_column():
    mat = jnp.array([[0, 0], [0, 0]])
    mat_norm, col_norms = normalize(mat)
    assert mat_norm.shape == mat.shape
    assert jnp.allclose(col_norms, jnp.array([0, 0]))
    assert jnp.all(mat_norm == 0)


def test_normalize_non_finite_values():
    mat = jnp.array([[1, jnp.inf], [3, jnp.nan]])
    mat_norm, col_norms = normalize(mat)
    assert mat_norm.shape == mat.shape
    assert col_norms.shape == (mat.shape[1],)
    assert jnp.isnan(mat_norm).any() or jnp.isinf(mat_norm).any()


def test_normalize_back_happy_path():
    H = jnp.array([[1, 2], [3, 4]])
    row_scales = jnp.array([1, 2])
    col_scales = jnp.array([1, 2])
    H_rescaled = normalize_back(H, row_scales, col_scales)
    expected = jnp.array([[1, 1], [1.5, 1]])
    assert jnp.allclose(H_rescaled, expected)


def test_normalize_back_no_row_scales():
    H = jnp.array([[1, 2], [3, 4]])
    row_scales = jnp.array([])
    col_scales = jnp.array([1, 2])
    H_rescaled = normalize_back(H, row_scales, col_scales)
    expected = jnp.array([[1, 1], [3, 2]])
    assert jnp.allclose(H_rescaled, expected)


def test_normalize_back_no_col_scales():
    H = jnp.array([[1, 2], [3, 4]])
    row_scales = jnp.array([1, 2])
    col_scales = jnp.array([])
    H_rescaled = normalize_back(H, row_scales, col_scales)
    expected = jnp.array([[1, 2], [1.5, 2]])
    assert jnp.allclose(H_rescaled, expected)


def test_normalize_back_empty_matrix():
    H = jnp.zeros((0, 0))
    row_scales = jnp.array([])
    col_scales = jnp.array([])
    H_rescaled = normalize_back(H, row_scales, col_scales)
    expected = jnp.zeros((0, 0))
    assert jnp.allclose(H_rescaled, expected)


def test_normalize_back_non_finite_values():
    H = jnp.array([[1, jnp.inf], [3, jnp.nan]])
    row_scales = jnp.array([1, 2])
    col_scales = jnp.array([1, 2])
    H_rescaled = normalize_back(H, row_scales, col_scales)
    assert jnp.isnan(H_rescaled).any() or jnp.isinf(H_rescaled).any()
