import jax.numpy as jnp
import numpy as np
from mcnnm.core import (
    initialize_coefficients,
    compute_svd,
    update_unit_fe,
    update_time_fe,
    initialize_matrices,
    compute_objective_value,
    compute_decomposition,
    initialize_fixed_effects_and_H,
)
import jax
from jax import random
from mcnnm.types import Array
from typing import Optional, Tuple

key = jax.random.PRNGKey(2024)


def initialize_matrices_numpy(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    V: Optional[np.ndarray],
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, T = Y.shape
    L = np.zeros_like(Y)
    # Initialize covariates as 0 if not used
    if X is None:
        X = np.zeros((N, 1))
    if Z is None:
        Z = np.zeros((T, 1))
    if V is None:
        V = np.zeros((N, T, 1))

    # Add identity matrices to X and Z
    X_add = np.eye(N)
    Z_add = np.eye(T)
    X_tilde = np.concatenate((X, X_add), axis=1)
    Z_tilde = np.concatenate((Z, Z_add), axis=1)

    # Initialize unit and time fixed effects
    unit_fe = np.ones(N) if use_unit_fe else np.zeros(N)
    time_fe = np.ones(T) if use_time_fe else np.zeros(T)

    return L, X_tilde, Z_tilde, V, unit_fe, time_fe


def initialize_coefficients_numpy(
    Y: np.ndarray, X: np.ndarray, Z: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, T = Y.shape
    L = np.zeros_like(Y)
    gamma = np.zeros(N)  # unit FE coefficients
    delta = np.zeros(T)  # time FE coefficients

    H = np.zeros((X.shape[1], Z.shape[1]))  # X and Z-covariate coefficients

    beta_shape = max(V.shape[2], 1)
    beta = np.zeros((beta_shape,))  # unit-time covariate coefficients

    return L, H, gamma, delta, beta


def mask_observed_numpy(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if A.shape != mask.shape:
        raise ValueError(f"The shapes of A ({A.shape}) and mask ({mask.shape}) do not match.")
    return A * mask


def mask_unobserved_numpy(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if A.shape != mask.shape:
        raise ValueError(f"The shapes of A ({A.shape}) and mask ({mask.shape}) do not match.")
    return np.where(mask, np.zeros_like(A), A)


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
    L: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    V: np.ndarray,
    H: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    beta: np.ndarray,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> np.ndarray:
    N, T = L.shape
    P = X.shape[1]
    Q = Z.shape[1]

    decomposition = L.copy()

    unit_fe_term = np.outer(gamma, np.ones(T))
    decomposition += np.where(use_unit_fe, unit_fe_term, np.zeros_like(unit_fe_term))

    time_fe_term = np.outer(np.ones(N), delta)
    decomposition += np.where(use_time_fe, time_fe_term, np.zeros_like(time_fe_term))

    if Q > 0:
        decomposition += X @ H[:P, :Q] @ Z.T
    if H.shape[1] > Q:
        decomposition += X @ H[:P, Q:]
    if P + N <= H.shape[0] and Q > 0:
        decomposition += H[P : P + N, :Q] @ Z.T
    decomposition += np.einsum("ntj,j->nt", V, beta)

    return decomposition


def compute_objective_value_numpy(
    Y: Array,
    X: Array,
    Z: Array,
    V: Array,
    H: Array,
    W: Array,
    L: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    sum_sing_vals: float,
    lambda_L: float,
    lambda_H: float,
    use_unit_fe: bool,
    use_time_fe: bool,
    inv_omega: Optional[Array] = None,
) -> float:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)
    H = np.array(H)
    W = np.array(W)
    L = np.array(L)
    gamma = np.array(gamma)
    delta = np.array(delta)
    beta = np.array(beta)

    N, T = Y.shape

    if inv_omega is None:
        inv_omega = np.eye(T)

    est_mat = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )

    err_mat = est_mat - Y
    weighted_err_mat = np.einsum(
        "ij,ntj,nsj->nts", inv_omega, err_mat[:, None, :], err_mat[:, :, None]
    )
    masked_weighted_err_mat = weighted_err_mat * W[:, None, :]
    obj_val = (
        np.sum(masked_weighted_err_mat) / np.sum(W)
        + lambda_L * sum_sing_vals
        + lambda_H * np.sum(np.abs(H))
    )

    return obj_val


def initialize_fixed_effects_and_H_numpy(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    use_unit_fe: bool,
    use_time_fe: bool,
    niter: int = 1000,
    rel_tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)
    W = np.array(W)

    N, T = Y.shape
    L, X_tilde, Z_tilde, V, unit_fe, time_fe = initialize_matrices_numpy(
        Y, X, Z, V, use_unit_fe, use_time_fe
    )
    _, H, _, _, beta = initialize_coefficients_numpy(Y, X_tilde, Z_tilde, V)

    H_rows, H_cols = X_tilde.shape[1], Z_tilde.shape[1]

    obj_val = np.inf
    new_obj_val = 0.0

    for iter_ in range(niter):
        if use_unit_fe:
            unit_fe = update_unit_fe_numpy(Y, X_tilde, Z_tilde, H, W, L, time_fe)
        else:
            unit_fe = np.zeros_like(unit_fe)

        if use_time_fe:
            time_fe = update_time_fe_numpy(Y, X_tilde, Z_tilde, H, W, L, unit_fe)
        else:
            time_fe = np.zeros_like(time_fe)

        new_obj_val = compute_objective_value_numpy(
            Y,
            X_tilde,
            Z_tilde,
            V,
            H,
            W,
            L,
            unit_fe,
            time_fe,
            beta,
            0.0,
            0.0,
            0.0,
            use_unit_fe,
            use_time_fe,
        )

        rel_error = np.abs(new_obj_val - obj_val) / obj_val
        if rel_error < rel_tol:
            break

        obj_val = new_obj_val

    E = compute_decomposition_numpy(
        L, X_tilde, Z_tilde, V, H, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )
    P_omega = mask_observed_numpy(Y - E, W)
    _, _, s = np.linalg.svd(P_omega, full_matrices=False)
    lambda_L_max = 2.0 * np.max(s) / np.sum(W)

    num_train = np.sum(W)
    T_mat = np.zeros((Y.size, H_rows * H_cols))
    in_prod_T = np.zeros(H_rows * H_cols)

    for j in range(H_cols):
        for i in range(H_rows):
            out_prod = mask_observed_numpy(np.outer(X_tilde[:, i], Z_tilde[:, j]), W)
            index = j * H_rows + i
            T_mat[:, index] = out_prod.ravel()
            in_prod_T[index] = np.sum(T_mat[:, index] ** 2)

    T_mat /= np.sqrt(num_train)
    in_prod_T /= num_train

    P_omega_resh = P_omega.ravel()
    all_Vs = np.dot(T_mat.T, P_omega_resh) / np.sqrt(num_train)
    lambda_H_max = 2 * np.max(np.abs(all_Vs))

    return unit_fe, time_fe, lambda_L_max, lambda_H_max, T_mat, in_prod_T


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


def test_compute_objective_value():
    # Set random seed for reproducibility
    key = random.PRNGKey(0)

    # Generate random input data
    N, T, P, Q, J = 10, 20, 5, 3, 4
    Y = random.normal(key, (N, T))
    X = random.normal(key, (N, P))
    Z = random.normal(key, (T, Q))
    V = random.normal(key, (N, T, J))
    H = random.normal(key, (P + N, Q + T))
    W = random.bernoulli(key, 0.8, (N, T))
    L = random.normal(key, (N, T))
    gamma = random.normal(key, (N,))
    delta = random.normal(key, (T,))
    beta = random.normal(key, (J,))
    sum_sing_vals = 10.0
    lambda_L = 0.1
    lambda_H = 0.05

    # Test case 1: With unit and time fixed effects, and inv_omega provided
    inv_omega = random.normal(key, (T, T))
    inv_omega = inv_omega @ inv_omega.T  # Make inv_omega symmetric positive definite
    expected_output = compute_objective_value_numpy(
        Y,
        X,
        Z,
        V,
        H,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sing_vals,
        lambda_L,
        lambda_H,
        True,
        True,
        inv_omega,
    )
    output = compute_objective_value(
        Y,
        X,
        Z,
        V,
        H,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sing_vals,
        lambda_L,
        lambda_H,
        True,
        True,
        inv_omega,
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)

    # Test case 2: With unit fixed effects only, and inv_omega not provided
    expected_output = compute_objective_value_numpy(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, True, False
    )
    output = compute_objective_value(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, True, False
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)

    # Test case 3: With time fixed effects only, and inv_omega provided
    inv_omega = random.normal(key, (T, T))
    inv_omega = inv_omega @ inv_omega.T  # Make inv_omega symmetric positive definite
    expected_output = compute_objective_value_numpy(
        Y,
        X,
        Z,
        V,
        H,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sing_vals,
        lambda_L,
        lambda_H,
        False,
        True,
        inv_omega,
    )
    output = compute_objective_value(
        Y,
        X,
        Z,
        V,
        H,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sing_vals,
        lambda_L,
        lambda_H,
        False,
        True,
        inv_omega,
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)

    # Test case 4: Without unit and time fixed effects, and inv_omega not provided
    expected_output = compute_objective_value_numpy(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, False, False
    )
    output = compute_objective_value(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, False, False
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)

    # Test case 5: Same as test case one, but with print statements
    inv_omega = random.normal(key, (T, T))
    inv_omega = inv_omega @ inv_omega.T  # Make inv_omega symmetric positive definite
    expected_output = compute_objective_value_numpy(
        Y,
        X,
        Z,
        V,
        H,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sing_vals,
        lambda_L,
        lambda_H,
        True,
        True,
        inv_omega,
    )
    output = compute_objective_value(
        Y,
        X,
        Z,
        V,
        H,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sing_vals,
        lambda_L,
        lambda_H,
        True,
        True,
        inv_omega,
        True,
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)


def test_initialize_fixed_effects_and_H():
    # Set random seed for reproducibility
    key = random.PRNGKey(0)

    # Generate random input data
    N, T, P, Q, J = 10, 20, 5, 3, 4
    Y = random.normal(key, (N, T))
    X = random.normal(key, (N, P))
    Z = random.normal(key, (T, Q))
    V = random.normal(key, (N, T, J))
    W = random.bernoulli(key, 0.8, (N, T))

    # Test case 1: With unit and time fixed effects
    expected_output = initialize_fixed_effects_and_H_numpy(Y, X, Z, V, W, True, True)
    output = initialize_fixed_effects_and_H(Y, X, Z, V, W, True, True)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # Test case 2: With unit fixed effects only
    expected_output = initialize_fixed_effects_and_H_numpy(Y, X, Z, V, W, True, False)
    output = initialize_fixed_effects_and_H(Y, X, Z, V, W, True, False)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # Test case 3: With time fixed effects only
    expected_output = initialize_fixed_effects_and_H_numpy(Y, X, Z, V, W, False, True)
    output = initialize_fixed_effects_and_H(Y, X, Z, V, W, False, True)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # Test case 4: Without unit and time fixed effects
    expected_output = initialize_fixed_effects_and_H_numpy(Y, X, Z, V, W, False, False)
    output = initialize_fixed_effects_and_H(Y, X, Z, V, W, False, False)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
