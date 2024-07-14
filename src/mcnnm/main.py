import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from src.mcnnm import Array
from util import *


def p_o(A: Array, mask: Array) -> Array:
    """
    Projects the matrix A onto the observed entries specified by the binary mask O.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        The projected matrix.

    Raises:
        ValueError: If the shapes of A and O do not match.
    """
    if A.shape != mask.shape:
        raise ValueError("Shapes of A and mask must match.")
    return jnp.where(mask, A, jnp.zeros_like(A))


def p_perp_o(A: Array, O: Array) -> Array:
    """
    Projects the matrix A onto the unobserved entries specified by the binary mask.

    Args:
        A: The input matrix.
        O: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.
    """
    if A.shape != O.shape:
        raise ValueError("Shapes of A and mask must match.")
    return jnp.where(O, jnp.zeros_like(A), A)


@jax.jit
def shrink_lambda(A: Array, lambda_: float) -> Array:
    """
    Applies the soft-thresholding operator to the singular values of a matrix A.

    Args:
        A: The input matrix.
        lambda_: The shrinkage parameter.

    Returns:
        The matrix with soft-thresholded singular values.
    """
    u, s, v_transpose = jnp.linalg.svd(A, full_matrices=False)
    s_shrunk = jnp.maximum(s - lambda_, 0)
    return jnp.dot(u * s_shrunk, v_transpose)


def objective_function(
        Y: Array, L: Array, Omega_inv: Optional[Array] = None, gamma: Optional[Array] = None,
        delta: Optional[Array] = None, beta: Optional[Array] = None, H: Optional[Array] = None,
        X: Optional[Array] = None
) -> float:
    """
    Computes the objective function value for the MC-NNM estimator (Equation 18).

    Args:
        Y: The observed outcome matrix.
        L: The low-rank matrix.
        Omega_inv: The autocorrelation matrix. If None, no autocorrelation is assumed.
        gamma: The unit fixed effects vector. If None, unit fixed effects are not included.
        delta: The time fixed effects vector. If None, time fixed effects are not included.
        beta: The coefficient vector for the covariates. If None, unit-time specific covariates are not included.
        H: The coefficient matrix for the  covariates. If None, unit and time specific covariates are not included.
        X: The matrix of unit and time specific covariates. If None, unit and time specific covariates are not included.

    Returns:
        The objective function value.
    """
    N, T = Y.shape
    if gamma is None:
        gamma = jnp.zeros(N)
    if delta is None:
        delta = jnp.zeros(T)
    if beta is None:
        beta = jnp.zeros((N, T))
    if H is None or X is None:
        H = jnp.zeros((N + Y.shape[1], T + Y.shape[0]))
        X = jnp.zeros((N, T))
    if Omega_inv is None:
        Omega_inv = jnp.eye(T)


    residual = Y - L - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N), delta) - beta - jnp.dot(X, H)
    return jnp.sum(jnp.dot(residual, Omega_inv) * residual) / (N * T)


def compute_fixed_effects(Y: Array, L: Array, beta: Optional[Array] = None, H: Optional[Array] = None,
                          X: Optional[Array] = None) -> tuple:
    """
    Computes the unit and time fixed effects (gamma and delta) for the MC-NNM estimator (Section 8.1).

    Args:
        Y: The observed outcome matrix.
        L: The low-rank matrix.
        beta: The coefficient vector for the unit-time specific covariates. If None, covariates are not included.
        H: The coefficient matrix for the unit and time specific covariates. If None, covariates are not included.
        X: The matrix of unit and time specific covariates. If None, covariates are not included.

    Returns:
        A tuple containing the unit fixed effects vector (gamma) and the time fixed effects vector (delta).
    """
    N, T = Y.shape
    if beta is None:
        beta = jnp.zeros((N, T))
    if H is None or X is None:
        H = jnp.zeros((N + Y.shape[1], T + Y.shape[0]))
        X = jnp.zeros((N, T))

    Y_adjusted = Y - L - beta - jnp.dot(X, H)
    gamma = jnp.mean(Y_adjusted, axis=1)
    delta = jnp.mean(Y_adjusted - jnp.outer(gamma, jnp.ones(T)), axis=0)

    return gamma, delta


def compute_H(Y: Array, L: Array, gamma: Optional[Array] = None, delta: Optional[Array] = None, beta: Optional[Array] = None, X: Optional[Array] = None) -> Array:
    """
    Computes the coefficient matrix H for the unit and time specific covariates (Section 8.1).

    Args:
        Y: The observed outcome matrix.
        L: The low-rank matrix.
        gamma: The unit fixed effects vector. If None, unit fixed effects are not included.
        delta: The time fixed effects vector. If None, time fixed effects are not included.
        beta: The coefficient vector for the unit-time specific covariates. If None, unit-time specific covariates are not included.
        X: The matrix of unit and time specific covariates. If None, unit and time specific covariates are not included.

    Returns:
        The coefficient matrix H.
    """
    N, T = Y.shape
    if gamma is None:
        gamma = jnp.zeros(N)
    if delta is None:
        delta = jnp.zeros(T)
    if beta is None:
        beta = jnp.zeros((N, T))
    if X is None:
        X = jnp.zeros((N, T))

    Y_adjusted = Y - L - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N), delta) - beta
    H = jnp.linalg.lstsq(X, Y_adjusted)[0]

    return H


def compute_L(
    Y: Array, O: Array, lambda_L: float, gamma: Optional[Array] = None, delta: Optional[Array] = None,
    beta: Optional[Array] = None, H: Optional[Array] = None, X: Optional[Array] = None,
    Omega_inv: Optional[Array] = None, max_iter: int = 1000, tol: float = 1e-4
) -> Array:
    """
    Computes the low-rank matrix L using the iterative algorithm (Equation 10).

    Args:
        ...
        Omega_inv: The inverse of the autocorrelation matrix. If None, no autocorrelation is assumed.
        ...

    Returns:
        The low-rank matrix L.
    """
    N, T = Y.shape
    if gamma is None:
        gamma = jnp.zeros(N)
    if delta is None:
        delta = jnp.zeros(T)
    if beta is None:
        beta = jnp.zeros((N, T))
    if H is None or X is None:
        H = jnp.zeros((N + Y.shape[1], T + Y.shape[0]))
        X = jnp.zeros((N, T))
    if Omega_inv is None:
        Omega_inv = jnp.eye(T)

    L = jnp.zeros_like(Y)
    for _ in range(max_iter):
        Y_adj = Y - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N), delta) - beta - jnp.dot(X, H)
        L_new = shrink_lambda(p_o(jnp.dot(Y_adj, Omega_inv), O) + p_perp_o(L, O), lambda_L / jnp.sqrt(jnp.sum(O)))
        if jnp.linalg.norm(L_new - L, ord='fro') < tol:
            break
        L = L_new

    return L


def cross_validation(
    Y: Array, W: Array, X: Optional[Array] = None, proposed_lambda_L: Optional[float] = None,
    proposed_lambda_H: Optional[float] = None, n_lambdas: int = 10, Omega: Optional[Array] = None, K: int = 5
) -> tuple:
    """
    Performs K-fold cross-validation to select the best regularization parameters lambda_L and lambda_H (Section 4.3).

    Args:
        Y: The observed outcome matrix.
        W: The binary treatment matrix.
        X: The matrix of unit and time specific covariates. If None, unit and time specific covariates are not included.
        proposed_lambda_L: The proposed lambda_L value. If None, the default sequence is used.
        proposed_lambda_H: The proposed lambda_H value. If None, the default sequence is used.
        n_lambdas: The number of lambda values to generate.
        Omega: The autocorrelation matrix. If None, no autocorrelation is assumed.
        K: The number of folds for cross-validation.

    Returns:
        A tuple containing the best values of lambda_L and lambda_H.
    """
    N, T = Y.shape
    if X is None:
        X = jnp.zeros((N, T))
    if Omega is None:
        Omega = jnp.eye(T)

    lambda_L_seq = propose_lambda(proposed_lambda_L, n_lambdas)
    lambda_H_seq = propose_lambda(proposed_lambda_H, n_lambdas)

    best_lambda_L = None
    best_lambda_H = None
    best_loss = jnp.inf

    for lambda_L in lambda_L_seq:
        for lambda_H in lambda_H_seq:
            loss = 0.0
            for k in range(K):
                mask = jnp.arange(N) % K == k
                Y_train, Y_test = Y[~mask], Y[mask]
                W_train, W_test = W[~mask], W[mask]
                X_train, X_test = X[~mask], X[mask]

                O_train = (W_train == 0)
                L_train = compute_L(Y_train, O_train, lambda_L, X=X_train, Omega_inv=jnp.linalg.inv(Omega))
                gamma_train, delta_train = compute_fixed_effects(Y_train, L_train, X=X_train)
                H_train = compute_H(Y_train, L_train, gamma_train, delta_train, X=X_train)
                beta_train = jnp.zeros_like(Y_train)

                O_test = (W_test == 0)
                Y_pred = L_train[mask] + jnp.outer(gamma_train[mask], jnp.ones(T)) + jnp.outer(jnp.ones(jnp.sum(mask)), delta_train) + beta_train[mask] + jnp.dot(X_test, H_train)
                loss += jnp.sum((Y_test - Y_pred)**2 * O_test) / jnp.sum(O_test)

            loss /= K
            if loss < best_loss:
                best_lambda_L = lambda_L
                best_lambda_H = lambda_H
                best_loss = loss

    return best_lambda_L, best_lambda_H


def compute_treatment_effect(Y: Array, L: Array, gamma: Array, delta: Array, beta: Array, H: Array, X: Array, W: Array) -> float:
    """
    Computes the average treatment effect (tau) for the treated units.

    Args:
        Y: The observed outcome matrix.
        L: The completed low-rank matrix.
        gamma: The unit fixed effects vector.
        delta: The time fixed effects vector.
        beta: The coefficient vector for the covariates.
        H: The coefficient matrix for the covariates.
        X: The matrix of unit and time specific covariates.
        W: The binary treatment matrix.

    Returns:
        The average treatment effect (tau) for the treated units.
    """
    N, T = Y.shape
    Y_completed = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta) + beta + jnp.dot(X, H)
    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    return tau

def fit(
    Y: Array, W: Array, X: Optional[Array] = None, Omega: Optional[Array] = None,
    lambda_L: Optional[float] = None, lambda_H: Optional[float] = None,
    return_tau: bool = True, return_lambda: bool = True,
    return_completed_L: bool = True, return_completed_Y: bool = True,
    max_iter: int = 1000, tol: float = 1e-4
) -> Tuple:
    """
    Estimates the MC-NNM model and returns the selected outputs.

    Args:
        Y: The observed outcome matrix.
        W: The binary treatment matrix.
        X: The matrix of unit and time specific covariates. If None, covariates are not included.
        Omega: The autocorrelation matrix. If None, no autocorrelation is assumed.
        lambda_L: The regularization parameter for the nuclear norm of L. If None, it is selected via cross-validation.
        lambda_H: The regularization parameter for the element-wise L1 norm of H. If None, it is selected via cross-validation.
        return_tau: Whether to return the average treatment effect (tau) for the treated units.
        return_lambda: Whether to return the optimal regularization parameter lambda_L.
        return_completed_L: Whether to return the completed low-rank matrix L.
        return_completed_Y: Whether to return the completed outcome matrix Y.
        max_iter: The maximum number of iterations for the algorithm.
        tol: The tolerance for the convergence of the algorithm.

    Returns:
        A tuple containing the selected outputs (tau, lambda_L, completed L, completed Y).
    """
    N, T = Y.shape
    if X is None:
        X = jnp.zeros((N, T))
    if Omega is None:
        Omega = jnp.eye(T)

    if lambda_L is None or lambda_H is None:
        lambda_L, lambda_H = cross_validation(Y, W, X, Omega=Omega)

    O = (W == 0)
    L = compute_L(Y, O, lambda_L, X=X, Omega_inv=jnp.linalg.inv(Omega), max_iter=max_iter, tol=tol)
    gamma, delta = compute_fixed_effects(Y, L, X=X)
    H = compute_H(Y, L, gamma, delta, X=X)
    beta = jnp.zeros_like(Y)

    results = []
    if return_tau:
        tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W)
        results.append(tau)
    if return_lambda:
        results.append(lambda_L)
    if return_completed_L:
        results.append(L)
    if return_completed_Y:
        Y_completed = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta) + beta + jnp.dot(X, H)
        results.append(Y_completed)

    return tuple(results)
