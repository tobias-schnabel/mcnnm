# mypy: ignore-errors
# type: ignore
import jax.numpy as jnp
import numpy as np
import jax
from jax import random
from mcnnm.types import Array
from typing import Optional, Tuple
from mcnnm.core_utils import normalize, normalize_back
from mcnnm.utils import generate_data
from mcnnm.core import (
    initialize_coefficients,
    compute_svd,
    update_unit_fe,
    update_time_fe,
    initialize_matrices,
    compute_objective_value,
    compute_decomposition,
    initialize_fixed_effects_and_H,
    svt,
    update_H,
    update_L,
    update_beta,
    fit,
)

key = jax.random.PRNGKey(2024)


def initialize_matrices_numpy(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    V: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return L, X_tilde, Z_tilde, V


def initialize_coefficients_numpy(
    Y: np.ndarray, X_tilde: np.ndarray, Z_tilde: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, T = Y.shape
    gamma = np.zeros(N)  # unit FE coefficients
    delta = np.zeros(T)  # time FE coefficients

    H_tilde = np.zeros(
        (X_tilde.shape[1], Z_tilde.shape[1])
    )  # X_tilde and Z_tilde-covariate coefficients

    beta_shape = max(V.shape[2], 1)
    beta = np.zeros((beta_shape,))  # unit-time covariate coefficients

    return H_tilde, gamma, delta, beta


def mask_observed_numpy(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if A.shape != mask.shape:
        raise ValueError(f"The shapes of A ({A.shape}) and mask ({mask.shape}) do not match.")
    return A * mask


def mask_unobserved_numpy(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if A.shape != mask.shape:
        raise ValueError(f"The shapes of A ({A.shape}) and mask ({mask.shape}) do not match.")
    return np.where(mask, np.zeros_like(A), A)


def update_unit_fe_numpy(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    time_fe: Array,
    use_unit_fe: bool,
) -> np.ndarray:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X_tilde = np.array(X_tilde)
    Z_tilde = np.array(Z_tilde)
    H_tilde = np.array(H_tilde)
    W = np.array(W)
    L = np.array(L)
    time_fe = np.array(time_fe)

    T_ = np.einsum("np,pq,tq->nt", X_tilde, H_tilde, Z_tilde)
    b_ = T_ + L + time_fe - Y
    b_mask_ = b_ * W
    l = np.sum(W, axis=1)
    res = np.where(l > 0, -np.sum(b_mask_, axis=1) / l, 0.0)
    return np.where(use_unit_fe, res, np.zeros_like(res))


def update_time_fe_numpy(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    use_time_fe: bool,
) -> np.ndarray:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X_tilde = np.array(X_tilde)
    Z_tilde = np.array(Z_tilde)
    H_tilde = np.array(H_tilde)
    W = np.array(W)
    L = np.array(L)
    unit_fe = np.array(unit_fe)

    T_ = np.einsum("np,pq,tq->nt", X_tilde, H_tilde, Z_tilde)
    b_ = T_ + L + np.expand_dims(unit_fe, axis=1) - Y
    b_mask_ = b_ * W
    l = np.sum(W, axis=0)
    res = np.where(l > 0, -np.sum(b_mask_, axis=0) / l, 0.0)
    return np.where(use_time_fe, res, np.zeros_like(res))


def compute_beta_numpy(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    time_fe: Array,
) -> np.ndarray:
    # Convert all inputs to numpy arrays
    Y = np.array(Y)
    X_tilde = np.array(X_tilde)
    Z_tilde = np.array(Z_tilde)
    V = np.array(V)
    H_tilde = np.array(H_tilde)
    W = np.array(W)
    L = np.array(L)
    unit_fe = np.array(unit_fe)
    time_fe = np.array(time_fe)

    T_ = np.einsum("np,pq,tq->nt", X_tilde, H_tilde, Z_tilde)
    b_ = T_ + L + np.expand_dims(unit_fe, axis=1) + time_fe - Y
    b_mask_ = b_ * W

    V_mask_ = V * np.expand_dims(W, axis=-1)
    V_sum_ = np.sum(V_mask_, axis=(0, 1))

    V_b_prod_ = np.einsum("ntj,nt->j", V_mask_, b_mask_)

    return np.where(V_sum_ > 0, -V_b_prod_ / V_sum_, 0.0)


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
    err_mask = err_mat * W
    weighted_error_term = (1 / np.sum(W)) * np.trace(err_mask @ inv_omega @ err_mask.T)

    norm_H = lambda_H * np.sum(np.abs(H))
    obj_val = weighted_error_term + lambda_L * sum_sing_vals + norm_H

    return obj_val


def initialize_fixed_effects_and_H_numpy(
    Y: np.ndarray,
    L: np.ndarray,
    X_tilde: np.ndarray,
    Z_tilde: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    use_unit_fe: bool,
    use_time_fe: bool,
    niter: int = 1000,
    rel_tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
]:
    num_train = np.sum(W)
    in_prod = np.zeros_like(W)

    H_tilde, gamma, delta, beta = initialize_coefficients_numpy(Y, X_tilde, Z_tilde, V)

    H_rows, H_cols = X_tilde.shape[1], Z_tilde.shape[1]

    obj_val = 1e10
    prev_obj_val = 1e10

    for i in range(niter):
        gamma = update_unit_fe_numpy(Y, X_tilde, Z_tilde, H_tilde, W, L, delta, use_unit_fe)
        delta = update_time_fe_numpy(Y, X_tilde, Z_tilde, H_tilde, W, L, gamma, use_time_fe)

        new_obj_val = compute_objective_value_numpy(
            Y,
            X_tilde,
            Z_tilde,
            V,
            H_tilde,
            W,
            L,
            gamma,
            delta,
            beta,
            0.0,
            0.0,
            0.0,
            use_unit_fe,
            use_time_fe,
        )

        rel_error = (new_obj_val - prev_obj_val) / (np.abs(prev_obj_val) + 1e-8)
        if (rel_error < rel_tol) and (rel_error > 0):
            break

        prev_obj_val = obj_val
        obj_val = new_obj_val

    Y_hat = compute_decomposition_numpy(
        L, X_tilde, Z_tilde, V, H_tilde, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    masked_error_matrix = mask_observed_numpy(Y - Y_hat, W)
    s = np.linalg.svd(masked_error_matrix, compute_uv=False)
    lambda_L_max = 2.0 * np.max(s) / num_train

    T_mat = np.zeros((Y.size, H_rows * H_cols))

    for j in range(H_rows * H_cols):
        out_prod = mask_observed_numpy(np.outer(X_tilde[:, j // H_rows], Z_tilde[:, j % H_cols]), W)
        T_mat[:, j] = out_prod.ravel()

    T_mat /= np.sqrt(num_train)

    in_prod_T = np.sum(T_mat**2, axis=0)

    P_omega_resh = masked_error_matrix.ravel()
    all_Vs = np.dot(T_mat.T, P_omega_resh) / np.sqrt(num_train)
    lambda_H_max = 2 * np.max(np.abs(all_Vs))

    if verbose:
        truncated_ov = np.round(obj_val, decimals=5)
        print(f"Initialization complete, objective value: {truncated_ov}")

    return gamma, delta, beta, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max


def test_initialize_coefficients_shape():
    Y = jnp.zeros((10, 5))
    X = jnp.zeros((10, 3))
    Z = jnp.zeros((5, 2))
    V = jnp.zeros((10, 5, 4))

    # Create X_tilde and Z_tilde
    X_add = jnp.eye(Y.shape[0])
    Z_add = jnp.eye(Y.shape[1])
    X_tilde = jnp.concatenate((X, X_add), axis=1)
    Z_tilde = jnp.concatenate((Z, Z_add), axis=1)

    H_tilde, gamma, delta, beta = initialize_coefficients(Y, X_tilde, Z_tilde, V)
    assert H_tilde.shape == (X_tilde.shape[1], Z_tilde.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (max(V.shape[2], 1),)
    assert jnp.allclose(H_tilde, jnp.zeros_like(H_tilde))
    assert jnp.allclose(gamma, jnp.zeros_like(gamma))
    assert jnp.allclose(delta, jnp.zeros_like(delta))
    assert jnp.allclose(beta, jnp.zeros_like(beta))


def test_initialize_coefficients_empty():
    Y = jnp.zeros((0, 0))
    X = jnp.zeros((0, 0))
    Z = jnp.zeros((0, 0))
    V = jnp.zeros((0, 0, 0))
    H_tilde, gamma, delta, beta = initialize_coefficients(Y, X, Z, V)
    assert H_tilde.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (max(V.shape[2], 0),)
    assert jnp.allclose(H_tilde, jnp.zeros_like(H_tilde))
    assert jnp.allclose(gamma, jnp.zeros_like(gamma))
    assert jnp.allclose(delta, jnp.zeros_like(delta))
    assert jnp.allclose(beta, jnp.zeros_like(beta))


def test_initialize_coefficients_single_element():
    Y = jnp.zeros((1, 1))
    X = jnp.zeros((1, 1))
    Z = jnp.zeros((1, 1))
    V = jnp.zeros((1, 1, 1))

    # Create X_tilde and Z_tilde
    X_add = jnp.eye(Y.shape[0])
    Z_add = jnp.eye(Y.shape[1])
    X_tilde = jnp.concatenate((X, X_add), axis=1)
    Z_tilde = jnp.concatenate((Z, Z_add), axis=1)

    H_tilde, gamma, delta, beta = initialize_coefficients(Y, X_tilde, Z_tilde, V)
    assert H_tilde.shape == (X_tilde.shape[1], Z_tilde.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (max(V.shape[2], 1),)
    assert jnp.allclose(H_tilde, jnp.zeros_like(H_tilde))
    assert jnp.allclose(gamma, jnp.zeros_like(gamma))
    assert jnp.allclose(delta, jnp.zeros_like(delta))
    assert jnp.allclose(beta, jnp.zeros_like(beta))


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
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe, True)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, True)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_unit_fe_partial_mask():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 0], [0, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    time_fe = jnp.array([1, 1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe, True)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, True)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_unit_fe_single_element():
    Y = jnp.array([[1]])
    X = jnp.array([[1]])
    Z = jnp.array([[1]])
    H = jnp.array([[1]])
    W = jnp.array([[1]])
    L = jnp.array([[1]])
    time_fe = jnp.array([1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe, True)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, True)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_unit_fe_no_unit_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    time_fe = jnp.array([1, 1])
    expected_output = update_unit_fe_numpy(Y, X, Z, H, W, L, time_fe, False)
    output = update_unit_fe(Y, X, Z, H, W, L, time_fe, False)
    assert jnp.allclose(output, jnp.zeros_like(expected_output))


def test_update_time_fe_happy_path():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe, True)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, True)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_time_fe_partial_mask():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 0], [0, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe, True)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, True)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_time_fe_single_element():
    Y = jnp.array([[1]])
    X = jnp.array([[1]])
    Z = jnp.array([[1]])
    H = jnp.array([[1]])
    W = jnp.array([[1]])
    L = jnp.array([[1]])
    unit_fe = jnp.array([1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe, True)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, True)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_time_fe_no_time_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    H = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    expected_output = update_time_fe_numpy(Y, X, Z, H, W, L, unit_fe, True)
    output = update_time_fe(Y, X, Z, H, W, L, unit_fe, False)
    assert jnp.allclose(output, jnp.zeros_like(expected_output))


def test_update_beta_happy_path():
    Y = jnp.array([[1, 2], [3, 4]])
    X_tilde = jnp.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    Z_tilde = jnp.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    V = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    H_tilde = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    time_fe = jnp.array([1, 1])
    expected_output = compute_beta_numpy(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    output = update_beta(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_beta_partial_mask():
    Y = jnp.array([[1, 2], [3, 4]])
    X_tilde = jnp.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    Z_tilde = jnp.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    V = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    H_tilde = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    W = jnp.array([[1, 0], [0, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    time_fe = jnp.array([1, 1])
    expected_output = compute_beta_numpy(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    output = update_beta(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_update_beta_single_element():
    Y = jnp.array([[1]])
    X_tilde = jnp.array([[1, 1]])
    Z_tilde = jnp.array([[1, 1]])
    V = jnp.array([[[1, 2]]])
    H_tilde = jnp.array([[1, 1], [1, 1]])
    W = jnp.array([[1]])
    L = jnp.array([[1]])
    unit_fe = jnp.array([1])
    time_fe = jnp.array([1])
    expected_output = compute_beta_numpy(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    output = update_beta(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    assert jnp.allclose(output, expected_output)


def test_update_beta_zero_V():
    Y = jnp.array([[1, 2], [3, 4]])
    X_tilde = jnp.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    Z_tilde = jnp.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    V = jnp.zeros((2, 2, 2))
    H_tilde = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    W = jnp.array([[1, 1], [1, 1]])
    L = jnp.array([[1, 1], [1, 1]])
    unit_fe = jnp.array([1, 1])
    time_fe = jnp.array([1, 1])
    expected_output = compute_beta_numpy(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    output = update_beta(Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe)
    assert jnp.allclose(output, expected_output)
    assert jnp.allclose(output, jnp.zeros_like(output))


def test_initialize_matrices_happy_path():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    V = jnp.array([[[1], [1]], [[1], [1]]])
    L, X_out, Z_out, V_out = initialize_matrices(Y, X, Z, V)
    assert L.shape == Y.shape
    assert X_out.shape == (Y.shape[0], X.shape[1] + Y.shape[0])
    assert Z_out.shape == (Y.shape[1], Z.shape[1] + Y.shape[1])
    assert V_out.shape == V.shape
    assert jnp.allclose(L, jnp.zeros_like(L))


def test_initialize_matrices_no_covariates():
    Y = jnp.array([[1, 2], [3, 4]])
    L, X_out, Z_out, V_out = initialize_matrices(Y, None, None, None)
    assert L.shape == Y.shape
    assert X_out.shape == (Y.shape[0], Y.shape[0])
    assert Z_out.shape == (Y.shape[1], Y.shape[1])
    assert V_out.shape == (Y.shape[0], Y.shape[1], 0)
    assert jnp.allclose(L, jnp.zeros_like(L))


def test_initialize_matrices_no_unit_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    V = jnp.array([[[1], [1]], [[1], [1]]])
    L, X_out, Z_out, V_out = initialize_matrices(Y, X, Z, V)
    assert L.shape == Y.shape
    assert X_out.shape == (Y.shape[0], X.shape[1] + Y.shape[0])
    assert Z_out.shape == (Y.shape[1], Z.shape[1] + Y.shape[1])
    assert V_out.shape == V.shape
    assert jnp.allclose(L, jnp.zeros_like(L))


def test_initialize_matrices_no_time_fe():
    Y = jnp.array([[1, 2], [3, 4]])
    X = jnp.array([[1, 1], [1, 1]])
    Z = jnp.array([[1, 1], [1, 1]])
    V = jnp.array([[[1], [1]], [[1], [1]]])
    L, X_out, Z_out, V_out = initialize_matrices(Y, X, Z, V)
    assert L.shape == Y.shape
    assert X_out.shape == (Y.shape[0], X.shape[1] + Y.shape[0])
    assert Z_out.shape == (Y.shape[1], Z.shape[1] + Y.shape[1])
    assert V_out.shape == V.shape
    assert jnp.allclose(L, jnp.zeros_like(L))


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
    assert not jnp.allclose(output, jnp.zeros_like(output))

    # Test with only unit fixed effects
    use_unit_fe = True
    use_time_fe = False
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))

    # Test with only time fixed effects
    use_unit_fe = False
    use_time_fe = True
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))

    # Test without fixed effects
    use_unit_fe = False
    use_time_fe = False
    expected_output = compute_decomposition_numpy(
        L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    output = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))


def test_compute_objective_value():
    # Set random seed for reproducibility
    key = random.PRNGKey(0)

    # Generate random input data
    N, T, P, Q, J = 4, 4, 5, 3, 4
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
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))

    # Test case 2: With unit fixed effects only, and inv_omega not provided
    expected_output = compute_objective_value_numpy(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, True, False
    )
    output = compute_objective_value(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, True, False
    )
    assert jnp.allclose(output, expected_output)
    assert not jnp.allclose(output, jnp.zeros_like(output))

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
    assert not jnp.allclose(output, jnp.zeros_like(output))

    # Test case 4: Without unit and time fixed effects, and inv_omega not provided
    expected_output = compute_objective_value_numpy(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, False, False
    )
    output = compute_objective_value(
        Y, X, Z, V, H, W, L, gamma, delta, beta, sum_sing_vals, lambda_L, lambda_H, False, False
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    assert not jnp.allclose(output, jnp.zeros_like(output))

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
    )
    assert jnp.allclose(output, expected_output, rtol=1e-5, atol=1e-5)


def test_initialize_fixed_effects_and_H():
    key = random.PRNGKey(2024)
    N, T = 5, 4
    Y, W, X, Z, V, true_params = generate_data(N, T, seed=key)

    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)
    # Test case 1: With unit and time fixed effects
    expected_output = initialize_fixed_effects_and_H_numpy(Y, L, X_tilde, Z_tilde, V, W, True, True)
    output = initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, True, True)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected)

    # Test case 2: With unit fixed effects only
    expected_output = initialize_fixed_effects_and_H_numpy(
        Y, L, X_tilde, Z_tilde, V, W, True, False
    )
    output = initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, True, False)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected)

    # Test case 3: With time fixed effects only
    expected_output = initialize_fixed_effects_and_H_numpy(
        Y, L, X_tilde, Z_tilde, V, W, False, True
    )
    output = initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, False, True)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected)

    # Test case 4: Without unit and time fixed effects
    expected_output = initialize_fixed_effects_and_H_numpy(
        Y, L, X_tilde, Z_tilde, V, W, False, False
    )
    output = initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, False, False)
    for expected, actual in zip(expected_output, output):
        assert jnp.allclose(actual, expected)


def test_svt_no_thresholding():
    # Test case where the threshold is zero (no thresholding)
    key = random.PRNGKey(0)
    U = random.normal(key, (5, 5))
    V = random.normal(key, (5, 5))
    sing_vals = jnp.array([3.0, 2.0, 1.0, 0.5, 0.1])
    threshold = 0.0

    expected_output = U @ jnp.diag(sing_vals) @ V.T
    output = svt(U, V, sing_vals, threshold)

    assert jnp.allclose(output, expected_output)


def test_svt_partial_thresholding():
    # Test case where some singular values are thresholded
    key = random.PRNGKey(0)
    U = random.normal(key, (5, 5))
    V = random.normal(key, (5, 5))
    sing_vals = jnp.array([3.0, 2.0, 1.0, 0.5, 0.1])
    threshold = 0.8

    expected_sing_vals = jnp.array([2.2, 1.2, 0.2, 0.0, 0.0])
    expected_output = U @ jnp.diag(expected_sing_vals) @ V.T
    output = svt(U, V, sing_vals, threshold)

    assert jnp.allclose(output, expected_output)


def test_svt_complete_thresholding():
    # Test case where all singular values are thresholded
    key = random.PRNGKey(0)
    U = random.normal(key, (5, 5))
    V = random.normal(key, (5, 5))
    sing_vals = jnp.array([3.0, 2.0, 1.0, 0.5, 0.1])
    threshold = 5.0

    expected_output = jnp.zeros((5, 5))
    output = svt(U, V, sing_vals, threshold)

    assert jnp.allclose(output, expected_output)


def test_svt_zero_matrix():
    U = jnp.zeros((5, 5))
    V = jnp.zeros((5, 5))
    sing_vals = jnp.zeros(5)
    threshold = 1.0

    expected_output = jnp.zeros((5, 5))
    output = svt(U, V, sing_vals, threshold)

    assert jnp.allclose(output, expected_output)


def test_update_H_with_regularization():
    # Test case with regularization (lambda_H > 0)
    key = random.PRNGKey(0)
    N, T, P, Q, J = 5, 4, 3, 2, 2
    Y = random.normal(key, (N, T))
    X_tilde = random.normal(key, (N, P + N))
    Z_tilde = random.normal(key, (T, Q + T))
    V = random.normal(key, (N, T, J))
    H_tilde = random.normal(key, (P + N, Q + T))
    T_mat = random.normal(key, (N * T, (P + N) * (Q + T)))
    in_prod = random.normal(key, (N * T,))
    in_prod_T = random.normal(key, ((P + N) * (Q + T),))
    W = random.bernoulli(key, 0.8, (N, T))
    L = random.normal(key, (N, T))
    unit_fe = random.normal(key, (N,))
    time_fe = random.normal(key, (T,))
    beta = random.normal(key, (J,))
    lambda_H = 0.1

    H_tilde_updated, updated_in_prod = update_H(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L,
        unit_fe,
        time_fe,
        beta,
        lambda_H,
        True,
        True,
    )

    assert H_tilde_updated.shape == H_tilde.shape
    assert updated_in_prod.shape == in_prod.shape
    assert not jnp.allclose(H_tilde_updated, H_tilde)
    assert not jnp.any(jnp.isnan(H_tilde_updated))
    assert not jnp.any(jnp.isnan(updated_in_prod))
    assert not jnp.allclose(H_tilde_updated, jnp.zeros_like(H_tilde_updated))


def test_update_H_without_fixed_effects():
    # Test case without unit and time fixed effects
    key = random.PRNGKey(0)
    N, T, P, Q, J = 5, 4, 3, 2, 2
    Y = random.normal(key, (N, T))
    X_tilde = random.normal(key, (N, P + N))
    Z_tilde = random.normal(key, (T, Q + T))
    V = random.normal(key, (N, T, J))
    H_tilde = random.normal(key, (P + N, Q + T))
    T_mat = random.normal(key, (N * T, (P + N) * (Q + T)))
    in_prod = random.normal(key, (N * T,))
    in_prod_T = random.normal(key, ((P + N) * (Q + T),))
    W = random.bernoulli(key, 0.8, (N, T))
    L = random.normal(key, (N, T))
    unit_fe = jnp.zeros((N,))
    time_fe = jnp.zeros((T,))
    beta = random.normal(key, (J,))
    lambda_H = 0.1

    H_tilde_updated, updated_in_prod = update_H(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L,
        unit_fe,
        time_fe,
        beta,
        lambda_H,
        False,
        False,
    )

    assert H_tilde_updated.shape == H_tilde.shape
    assert updated_in_prod.shape == in_prod.shape
    assert not jnp.allclose(H_tilde_updated, H_tilde)
    assert not jnp.any(jnp.isnan(H_tilde_updated))
    assert not jnp.any(jnp.isnan(updated_in_prod))
    assert not jnp.allclose(H_tilde_updated, jnp.zeros_like(H_tilde_updated))


def test_update_L_no_regularization():
    # Test case with no regularization (lambda_L = 0)
    key = random.PRNGKey(0)
    N, T, P, Q, J = 5, 4, 3, 2, 2
    Y = random.normal(key, (N, T))
    X_tilde = random.normal(key, (N, P + N))
    Z_tilde = random.normal(key, (T, Q + T))
    V = random.normal(key, (N, T, J))
    H_tilde = random.normal(key, (P + N, Q + T))
    W = random.bernoulli(key, 0.8, (N, T))
    L = random.normal(key, (N, T))
    unit_fe = random.normal(key, (N,))
    time_fe = random.normal(key, (T,))
    beta = random.normal(key, (J,))
    lambda_L = 0.0

    L_upd, S = update_L(
        Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe, beta, lambda_L, True, True
    )

    assert L_upd.shape == L.shape
    assert S.shape == (min(N, T),)
    assert not jnp.allclose(L_upd, L)
    assert not jnp.any(jnp.isnan(L_upd))
    assert not jnp.any(jnp.isnan(S))
    assert not jnp.allclose(L_upd, jnp.zeros_like(L))


def test_update_L_with_regularization():
    # Test case with regularization (lambda_L > 0)
    key = random.PRNGKey(0)
    N, T, P, Q, J = 5, 4, 3, 2, 2
    Y = random.normal(key, (N, T))
    X_tilde = random.normal(key, (N, P + N))
    Z_tilde = random.normal(key, (T, Q + T))
    V = random.normal(key, (N, T, J))
    H_tilde = random.normal(key, (P + N, Q + T))
    W = random.bernoulli(key, 0.8, (N, T))
    L = random.normal(key, (N, T))
    unit_fe = random.normal(key, (N,))
    time_fe = random.normal(key, (T,))
    beta = random.normal(key, (J,))
    lambda_L = 0.1

    L_upd, S = update_L(
        Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe, beta, lambda_L, True, True
    )

    assert L_upd.shape == L.shape
    assert S.shape == (min(N, T),)
    assert not jnp.allclose(L_upd, L)
    assert not jnp.any(jnp.isnan(L_upd))
    assert not jnp.any(jnp.isnan(S))
    assert not jnp.allclose(L_upd, jnp.zeros_like(L))


def test_update_L_without_fixed_effects():
    # Test case without unit and time fixed effects
    key = random.PRNGKey(0)
    N, T, P, Q, J = 5, 4, 3, 2, 2
    Y = random.normal(key, (N, T))
    X_tilde = random.normal(key, (N, P + N))
    Z_tilde = random.normal(key, (T, Q + T))
    V = random.normal(key, (N, T, J))
    H_tilde = random.normal(key, (P + N, Q + T))
    W = random.bernoulli(key, 0.8, (N, T))
    L = random.normal(key, (N, T))
    unit_fe = jnp.zeros((N,))
    time_fe = jnp.zeros((T,))
    beta = random.normal(key, (J,))
    lambda_L = 0.1

    L_upd, S = update_L(
        Y, X_tilde, Z_tilde, V, H_tilde, W, L, unit_fe, time_fe, beta, lambda_L, False, False
    )

    assert L_upd.shape == L.shape
    assert S.shape == (min(N, T),)
    assert not jnp.allclose(L_upd, L)
    assert not jnp.any(jnp.isnan(L_upd))
    assert not jnp.any(jnp.isnan(S))
    assert not jnp.allclose(L_upd, jnp.zeros_like(L))


def test_fit_happy_path():
    N, T = 24, 48
    Y, W, X, Z, V, true_params = generate_data(N, T, seed=2024)

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, True, True, verbose=True)
    )

    lambda_L = lambda_L_max
    lambda_H = lambda_H_max

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L,
        lambda_H,
        True,
        True,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.allclose(H_out, jnp.zeros_like(H_out))
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert not jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert not jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.allclose(beta_out, jnp.zeros_like(beta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.allclose(in_prod, jnp.zeros_like(in_prod))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_X_covariates():
    N, T = 24, 48
    Y, W, X, Z, V, true_params = generate_data(N, T, X_cov=False, Z_cov=True, V_cov=True, seed=2024)

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, False, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        True,
        True,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert not jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert not jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_Z_covariates():
    N, T = 24, 48
    Y, W, X, Z, V, true_params = generate_data(N, T, X_cov=True, Z_cov=False, V_cov=True, seed=2024)

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, False, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        True,
        True,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert not jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert not jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_V_covariates():
    N, T = 24, 48
    Y, W, X, Z, V, true_params = generate_data(N, T, X_cov=True, Z_cov=True, V_cov=False, seed=2024)

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, False, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        True,
        True,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert not jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert not jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_covariates():
    N, T = 5, 5
    Y, W, X, Z, V, true_params = generate_data(
        N, T, X_cov=False, Z_cov=False, V_cov=False, seed=2024
    )

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, False, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        True,
        True,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert not jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert not jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_unit_fixed_effects():
    N, T = 5, 5
    Y, W, X, Z, V, true_params = generate_data(N, T, unit_fe=False, time_fe=True, seed=2024)

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, True, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        False,
        True,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.allclose(H_out, jnp.zeros_like(H_out))
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert not jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.allclose(beta_out, jnp.zeros_like(beta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.allclose(in_prod, jnp.zeros_like(in_prod))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_time_fixed_effects():
    N, T = 5, 5
    Y, W, X, Z, V, true_params = generate_data(N, T, unit_fe=True, time_fe=False, seed=2024)

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, True, False, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        True,
        False,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.allclose(H_out, jnp.zeros_like(H_out))
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.allclose(beta_out, jnp.zeros_like(beta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    assert not jnp.allclose(in_prod, jnp.zeros_like(in_prod))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_fixed_effects():
    N, T = 5, 5
    Y, W, X, Z, V, true_params = generate_data(
        N, T, unit_fe=False, time_fe=False, seed=2024, noise_scale=0.1
    )

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    X_tilde, X_tilde_col_norms = normalize(X_tilde)
    Z_tilde, Z_tilde_col_norms = normalize(Z_tilde)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, False, verbose=True)
    )

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L_max,
        lambda_H_max,
        False,
        False,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    # renormalize H_tilde
    H_tilde = normalize_back(H_tilde, X_tilde_col_norms, Z_tilde_col_norms)

    assert H_out.shape == H_tilde.shape
    assert not jnp.allclose(H_out, jnp.zeros_like(H_out))
    # assert not jnp.any(jnp.isnan(H_out)) TODO
    assert L_out.shape == (N, T)
    # assert not jnp.any(jnp.isnan(L_out)) TODO
    assert gamma_out.shape == (N,)
    assert jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert not jnp.allclose(beta_out, jnp.zeros_like(beta_out))
    # assert not jnp.any(jnp.isnan(beta_out))  TODO
    assert not jnp.allclose(in_prod, jnp.zeros_like(in_prod))
    # assert not jnp.any(jnp.isnan(in_prod)) TODO

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, True, True
    )
    assert Y_hat.shape == Y.shape
    # assert not jnp.any(jnp.isnan(Y_hat)) TODO
    assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))


def test_fit_no_fixed_effects_no_covariates():
    N, T = 5, 5
    Y, W, X, Z, V, true_params = generate_data(
        N,
        T,
        X_cov=False,
        Z_cov=False,
        V_cov=False,
        unit_fe=False,
        time_fe=False,
        seed=2024,
        noise_scale=0.5,
    )

    L_out, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    # X_tilde, X_tilde_col_norms = normalize(X_tilde)
    # Z_tilde, Z_tilde_col_norms = normalize(Z_tilde)

    gamma, delta, beta_out, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L_out, X_tilde, Z_tilde, V, W, False, False, verbose=True)
    )

    lambda_L = lambda_L_max - 1e-8
    lambda_H = lambda_H_max - 1e-8

    H_out, L_out, gamma_out, delta_out, beta_out, in_prod, obj_val_final = fit(
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        T_mat,
        in_prod,
        in_prod_T,
        W,
        L_out,
        gamma,
        delta,
        beta_out,
        lambda_L,
        lambda_H,
        False,
        False,
        niter=1000,
        verbose=True,
        print_iters=False,
    )

    assert H_out.shape == H_tilde.shape
    assert not jnp.any(jnp.isnan(H_out))
    assert L_out.shape == (N, T)
    assert not jnp.any(jnp.isnan(L_out))
    assert gamma_out.shape == (N,)
    assert jnp.allclose(gamma_out, jnp.zeros_like(gamma_out))
    assert not jnp.any(jnp.isnan(gamma_out))
    assert delta_out.shape == (T,)
    assert jnp.allclose(delta_out, jnp.zeros_like(delta_out))
    assert not jnp.any(jnp.isnan(delta_out))
    assert jnp.allclose(beta_out, jnp.zeros_like(beta_out))
    assert not jnp.any(jnp.isnan(beta_out))
    # assert jnp.allclose(in_prod, jnp.zeros_like(in_prod))
    assert not jnp.any(jnp.isnan(in_prod))

    # reconstruct Y
    Y_hat = compute_decomposition(
        L_out, X_tilde, Z_tilde, V, H_out, gamma_out, delta_out, beta_out, False, False
    )
    assert Y_hat.shape == Y.shape
    assert not jnp.any(jnp.isnan(Y_hat))
    # assert not jnp.allclose(Y_hat, jnp.zeros_like(Y_hat))
