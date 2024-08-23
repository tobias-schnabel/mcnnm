import jax.numpy as jnp
import numpy as np
from mcnnm.core import (
    initialize_coefficients,
    compute_svd,
    update_unit_fe,
    update_time_fe,
    initialize_matrices,
)
from mcnnm.core import compute_decomposition
import jax
from jax import random
from mcnnm.types import Array

key = jax.random.PRNGKey(2024)


def update_unit_fe_numpy(
    Y: Array, X: Array, Z: Array, H: Array, W: Array, L: Array, time_fe: Array
) -> np.ndarray:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X = np.array(X)
    Z = np.array(Z)
    H = np.array(H)
    W = np.array(W)
    L = np.array(L)
    time_fe = np.array(time_fe)

    N, T = Y.shape
    P, Q = H.shape

    res = np.zeros(N)
    for i in range(N):
        T_ = X[i].reshape(1, -1) @ H @ Z.T
        b_ = T_ + L[i].reshape(1, -1) + time_fe.reshape(1, -1) - Y[i].reshape(1, -1)
        b_mask_ = b_ * W[i].reshape(1, -1)
        l = np.sum(W[i])
        if l > 0:
            res[i] = -np.sum(b_mask_) / l
        else:
            res[i] = 0

    return res


def update_time_fe_numpy(
    Y: Array, X: Array, Z: Array, H: Array, W: Array, L: Array, unit_fe: Array
) -> np.ndarray:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X = np.array(X)
    Z = np.array(Z)
    H = np.array(H)
    W = np.array(W)
    L = np.array(L)
    unit_fe = np.array(unit_fe)
    N, T = Y.shape
    P, Q = H.shape

    res = np.zeros(T)
    for i in range(T):
        T_ = X @ H @ Z[i].reshape(-1, 1)
        b_ = T_.reshape(-1) + L[:, i] + unit_fe - Y[:, i]
        h_ = W[:, i]
        b_mask_ = b_ * h_
        l = np.sum(h_ > 0)
        if l > 0:
            res[i] = -np.sum(b_mask_) / l
        else:
            res[i] = 0

    return res


def compute_decomposition_numpy(
    L: Array,
    X: Array,
    Z: Array,
    V: Array,
    H: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> np.ndarray:
    # Convert all inputs to numpy arrays
    L = np.array(L)
    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)
    H = np.array(H)
    gamma = np.array(gamma)
    delta = np.array(delta)
    beta = np.array(beta)

    N, T = L.shape
    P = X.shape[1]
    Q = Z.shape[1]

    decomposition = L.copy()

    if use_unit_fe:
        decomposition += np.outer(gamma, np.ones(T))
    if use_time_fe:
        decomposition += np.outer(np.ones(N), delta)

    decomposition += (
        X @ H[:P, :Q] @ Z.T + X @ H[:P, Q:] + H[P:, :Q] @ Z.T + np.einsum("ntj,j->nt", V, beta)
    )

    return decomposition


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


def test_compute_svd_happy_path():
    M = jnp.array([[1, 2], [3, 4]])
    U, V, Sigma = compute_svd(M)
    assert U.shape == (2, 2)
    assert V.shape == (2, 2)
    assert Sigma.shape == (2,)


def test_compute_svd_single_element():
    M = jnp.array([[5]])
    U, V, Sigma = compute_svd(M)
    assert U.shape == (1, 1)
    assert V.shape == (1, 1)
    assert Sigma.shape == (1,)


def test_compute_svd_empty_matrix():
    M = jnp.zeros((0, 0))
    U, V, Sigma = compute_svd(M)
    assert U.shape == (0, 0)
    assert V.shape == (0, 0)
    assert Sigma.shape == (0,)


def test_compute_svd_non_square_matrix():
    M = jnp.array([[1, 2, 3], [4, 5, 6]])
    U, V, Sigma = compute_svd(M)
    assert U.shape == (2, 2)
    assert V.shape == (3, 2)
    assert Sigma.shape == (2,)


def test_update_unit_fe_happy_path():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    time_fe = jnp.array([1, 1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, True)
    assert jnp.allclose(output, expected_output)


def test_update_unit_fe_partial_mask():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 0], [0, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    time_fe = jnp.array([1, 1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, True)
    assert jnp.allclose(output, expected_output)


def test_update_unit_fe_single_element():
    Y = jnp.array([[1]])
    X = jnp.array([[1]])
    Z = jnp.array([[1]])
    H = jnp.array([[1]])
    W = jnp.array([[1]])
    L = jnp.array([[1]])
    time_fe = jnp.array([1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, True)
    assert jnp.allclose(output, expected_output)


def test_update_unit_fe_no_unit_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    time_fe = jnp.array([1, 1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, False)
    assert jnp.allclose(output, jnp.zeros_like(expected_output))


def test_initialize_matrices_happy_path():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    V = jnp.array([[[1], [1]], [[1], [1]]])
    L, X_out, Z_out, V_out, unit_fe, time_fe = initialize_matrices(Y, X, Z, V, True, True)
    assert L.shape == Y.shape
    assert X_out.shape == X.shape
    assert Z_out.shape == Z.shape
    assert V_out.shape == V.shape
    assert unit_fe.shape == (Y.shape[0],)
    assert time_fe.shape == (Y.shape[1],)


def test_initialize_matrices_no_covariates():
    Y = jnp.array([[1, 2], [3, 4]])
    L, X_out, Z_out, V_out, unit_fe, time_fe = initialize_matrices(Y, None, None, None, True, True)
    assert L.shape == Y.shape
    assert X_out.shape == (Y.shape[0], 1)
    assert Z_out.shape == (Y.shape[1], 1)
    assert V_out.shape == (Y.shape[0], Y.shape[1], 1)
    assert unit_fe.shape == (Y.shape[0],)
    assert time_fe.shape == (Y.shape[1],)


def test_initialize_matrices_no_unit_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    V = jnp.array([[[1], [1]], [[1], [1]]])
    L, X_out, Z_out, V_out, unit_fe, time_fe = initialize_matrices(Y, X, Z, V, False, True)
    assert L.shape == Y.shape
    assert X_out.shape == X.shape
    assert Z_out.shape == Z.shape
    assert V_out.shape == V.shape
    assert unit_fe.shape == (Y.shape[0],)
    assert time_fe.shape == (Y.shape[1],)


def test_initialize_matrices_no_time_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    V = jnp.array([[[1], [1]], [[1], [1]]])
    L, X_out, Z_out, V_out, unit_fe, time_fe = initialize_matrices(Y, X, Z, V, True, False)
    assert L.shape == Y.shape
    assert X_out.shape == X.shape
    assert Z_out.shape == Z.shape
    assert V_out.shape == V.shape
    assert unit_fe.shape == (Y.shape[0],)
    assert time_fe.shape == (Y.shape[1],)


def test_update_time_fe_happy_path():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, True)
    assert jnp.allclose(output, expected_output)


def test_update_time_fe_partial_mask():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 0], [0, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, True)
    assert jnp.allclose(output, expected_output)


def test_update_time_fe_single_element():
    Y = jnp.array([[1]])
    X = jnp.array([[1]])
    Z = jnp.array([[1]])
    H = jnp.array([[1]])
    W = jnp.array([[1]])
    L = jnp.array([[1]])
    unit_fe = jnp.array([1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, True)
    assert jnp.allclose(output, expected_output)


def test_update_time_fe_no_time_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, False)
    assert jnp.allclose(output, jnp.zeros_like(expected_output))


def test_compute_decomposition():
    # Set random seed for reproducibility
    key = random.PRNGKey(0)

    # Generate random input arrays
    N, T, P, Q, J = 5, 4, 3, 2, 2
    L = random.normal(key, (N, T))
    X = random.normal(key, (N, P))
    Z = random.normal(key, (T, Q))
    V = random.normal(key, (N, T, J))
    H = random.normal(key, (P + N, Q + T))
    gamma = random.normal(key, (N,))
    delta = random.normal(key, (T,))
    beta = random.normal(key, (J,))

    # Test with both unit and time fixed effects
    use_unit_fe = True
    use_time_fe = True
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)

    # Test with only unit fixed effects
    use_unit_fe = True
    use_time_fe = False
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)

    # Test with only time fixed effects
    use_unit_fe = False
    use_time_fe = True
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)

    # Test without fixed effects
    use_unit_fe = False
    use_time_fe = False
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)
