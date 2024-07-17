import jax
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple, NamedTuple
from . import Array
from mcnnm.util import *
# from .util import propose_lambda


def objective_function(
        Y: Array, L: Array, Omega_inv: Optional[Array] = None, gamma: Optional[Array] = None,
        delta: Optional[Array] = None, beta: Optional[Array] = None, H: Optional[Array] = None,
        X: Optional[Array] = None, Z: Optional[Array] = None, V: Optional[Array] = None
) -> float:
    """
    Computes the objective function value for the MC-NNM estimator (Equation 18).

    Args:
        Y: The observed outcome matrix of shape (N, T).
        L: The low-rank matrix of shape (N, T).
        Omega_inv: The inverse of the autocorrelation matrix of shape (T, T). If None, no autocorrelation is assumed.
        gamma: The unit fixed effects vector of shape (N,). If None, unit fixed effects are not included.
        delta: The time fixed effects vector of shape (T,). If None, time fixed effects are not included.
        beta: The coefficient vector for the unit-time specific covariates of shape (J,). If None, unit-time specific covariates are not included.
        H: The coefficient matrix for the covariates of shape (N+P, T+Q). If None, unit and time specific covariates are not included.
        X: The unit-specific covariates matrix of shape (N, P). If None, unit-specific covariates are not included.
        Z: The time-specific covariates matrix of shape (T, Q). If None, time-specific covariates are not included.
        V: The unit-time specific covariates tensor of shape (N, T, J). If None, unit-time specific covariates are not included.

    Returns:
        The objective function value.
    """
    N, T = Y.shape
    if gamma is None:
        gamma = jnp.zeros(N)
    if delta is None:
        delta = jnp.zeros(T)
    if beta is None:
        beta = jnp.zeros(V.shape[2]) if V is not None else jnp.zeros(0)
    if H is None or X is None or Z is None:
        H = jnp.zeros((N, T))
        X_tilde = jnp.zeros((N, 0))
        Z_tilde = jnp.zeros((T, 0))
    else:
        X_tilde = jnp.hstack((X, jnp.eye(N)))
        Z_tilde = jnp.hstack((Z, jnp.eye(T)))
    if V is None:
        V = jnp.zeros((N, T, 0))
    if Omega_inv is None:
        Omega_inv = jnp.eye(T)

    residual = (Y - L - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N), delta)
                - jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T)) - jnp.sum(V * beta, axis=2))
    return jnp.sum(jnp.dot(residual, Omega_inv) * residual) / (N * T)


def cross_validation(
        Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None, V: Optional[Array] = None,
        proposed_lambda_L: Optional[float] = None, proposed_lambda_H: Optional[float] = None,
        n_lambdas: int = 6, Omega: Optional[Array] = None, K: int = 3
) -> Tuple[float, float]:
    """
    Performs K-fold cross-validation to select optimal regularization parameters lambda_L and lambda_H.

    This function splits the data into K folds, trains the model on K-1 folds, and evaluates
    on the remaining fold for different combinations of lambda_L and lambda_H. It returns
    the combination that minimizes the mean squared error on the validation sets.

    Args:
        Y: The observed outcome matrix of shape (N, T).
        W: The binary treatment matrix of shape (N, T).
        X: The unit-specific covariates matrix of shape (N, P). If None, not included.
        Z: The time-specific covariates matrix of shape (T, Q). If None, not included.
        V: The unit-time specific covariates tensor of shape (N, T, J). If None, not included.
        proposed_lambda_L: The proposed value for lambda_L. If None, a default sequence is used.
        proposed_lambda_H: The proposed value for lambda_H. If None, a default sequence is used.
        n_lambdas: The number of lambda values to test in the grid search.
        Omega: The autocorrelation matrix of shape (T, T). If None, no autocorrelation is assumed.
        K: The number of folds for cross-validation.

    Returns:
        A tuple containing the optimal values for lambda_L and lambda_H.
    """
    N, T = Y.shape
    if X is None:
        X = jnp.zeros((N, 0))
    if Z is None:
        Z = jnp.zeros((T, 0))
    if V is None:
        V = jnp.zeros((N, T, 0))
    if Omega is None:
        Omega = jnp.eye(T)

    lambda_L_seq = propose_lambda(proposed_lambda_L, n_lambdas)
    lambda_H_seq = propose_lambda(proposed_lambda_H, n_lambdas)

    best_lambda_L = None
    best_lambda_H = None
    best_loss = jnp.inf

    key = random.PRNGKey(0)

    for lambda_L in lambda_L_seq:
        for lambda_H in lambda_H_seq:
            loss = 0.0
            for k in range(K):
                key, subkey = random.split(key)
                mask = random.bernoulli(subkey, 0.8, (N,))
                train_idx = jnp.where(mask)[0]
                test_idx = jnp.where(~mask)[0]

                Y_train, Y_test = Y[train_idx], Y[test_idx]
                W_train, W_test = W[train_idx], W[test_idx]
                X_train, X_test = X[train_idx], X[test_idx]
                V_train, V_test = V[train_idx], V[test_idx]

                results = fit(Y_train, W_train, X=X_train, Z=Z, V=V_train, Omega=Omega,
                              lambda_L=lambda_L, lambda_H=lambda_H, max_iter=50, tol=1e-4,
                              return_tau=False, return_lambda=False, return_completed_L=False,
                              return_completed_Y=True, return_fixed_effects=True,
                              return_covariate_coefficients=True)

                Y_pred = (results.Y_completed[test_idx] +
                          jnp.outer(results.gamma[test_idx], jnp.ones(T)) +
                          jnp.outer(jnp.ones(test_idx.shape[0]), results.delta))

                if V_test.shape[2] > 0:
                    Y_pred += jnp.sum(V_test * results.beta, axis=2)

                O_test = (W_test == 0)
                loss += jnp.sum((Y_test - Y_pred) ** 2 * O_test) / jnp.sum(O_test)

            loss /= K
            if loss < best_loss:
                best_lambda_L = lambda_L.item()
                best_lambda_H = lambda_H.item()
                best_loss = loss.item()

    return best_lambda_L, best_lambda_H


def compute_treatment_effect(Y: Array, L: Array, gamma: Array, delta: Array, beta: Array, H: Array,
                             X: Array, W: Array, Z: Optional[Array] = None, V: Optional[Array] = None) -> float:
    """
    Computes the average treatment effect based on the estimated parameters and observed data.

    This function calculates the difference between the observed outcomes and the predicted
    outcomes for the treated units, and then averages this difference to estimate the
    treatment effect.

    Args:
        Y: The observed outcome matrix of shape (N, T).
        L: The estimated low-rank matrix of shape (N, T).
        gamma: The estimated unit fixed effects vector of shape (N,).
        delta: The estimated time fixed effects vector of shape (T,).
        beta: The estimated coefficient vector for unit-time specific covariates of shape (J,).
        H: The estimated coefficient matrix for covariates of shape (N+P, T+Q).
        X: The unit-specific covariates matrix of shape (N, P).
        W: The binary treatment matrix of shape (N, T).
        Z: The time-specific covariates matrix of shape (T, Q). If None, not included.
        V: The unit-time specific covariates tensor of shape (N, T, J). If None, not included.

    Returns:
        The estimated average treatment effect as a float.
    """
    N, T = Y.shape
    P = X.shape[1]
    Q = Z.shape[1]

    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))
    Y_completed = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)

    if X is not None and Z is not None and X.shape[1] > 0 and Z.shape[1] > 0:
        Y_completed += jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))

    if V is not None:
        Y_completed += jnp.sum(V * beta[None, None, :], axis=2)

    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    return tau


class MCNNMResults(NamedTuple):
    tau: Optional[float] = None
    lambda_L: Optional[float] = None
    lambda_H: Optional[float] = None
    L: Optional[Array] = None
    Y_completed: Optional[Array] = None
    gamma: Optional[Array] = None
    delta: Optional[Array] = None
    beta: Optional[Array] = None
    H: Optional[Array] = None



# @jax.jit
def fit(
        Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None, V: Optional[Array] = None,
        Omega: Optional[Array] = None, lambda_L: Optional[float] = None, lambda_H: Optional[float] = None,
        return_tau: bool = True, return_lambda: bool = True,
        return_completed_L: bool = True, return_completed_Y: bool = True,
        return_fixed_effects: bool = False, return_covariate_coefficients: bool = False,
        max_iter: int = 1000, tol: float = 1e-4, verbose: bool = False
) -> MCNNMResults:
    N, T = Y.shape
    if X is None:
        X = jnp.zeros((N, 0))
    if Z is None:
        Z = jnp.zeros((T, 0))
    if V is None:
        V = jnp.zeros((N, T, 0))
    if Omega is None:
        Omega = jnp.eye(T)

    if lambda_L is None or lambda_H is None:
        if verbose:
            print_with_timestamp(f"Cross-validating lambda_L, lambda_H")
        lambda_L, lambda_H = cross_validation(Y, W, X=X, Z=Z, V=V, Omega=Omega)
        if verbose:
            print_with_timestamp(f"Selected lambda_L: {lambda_L:.4f}, lambda_H: {lambda_H:.4f}")

    O = (W == 0)
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))

    # Initialize parameters
    L = jnp.zeros_like(Y)
    H = jnp.zeros((X_tilde.shape[1], Z_tilde.shape[1]))
    gamma = jnp.zeros(N)
    delta = jnp.zeros(T)
    beta = jnp.zeros(V.shape[2]) if V is not None and V.shape[2] > 0 else jnp.zeros(0)

    for iteration in range(max_iter):
        # Update L using shrinkage operator
        Y_adj = Y - jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T)) - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N),
                                                                                                        delta)
        if V is not None and V.shape[2] > 0:
            Y_adj -= jnp.sum(V * beta, axis=2)
        L_new = shrink_lambda(p_o(jnp.dot(Y_adj, Omega), O) + p_perp_o(L, O), lambda_L * jnp.sum(O) / 2)

        # Check convergence
        if jnp.linalg.norm(L_new - L, ord='fro') < tol:
            if verbose:
                print_with_timestamp(f"Converged after {iteration + 1} iterations")
            break

        L = L_new

        # Update H using coordinate descent
        Y_adj = Y - L - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N), delta)
        if V is not None and V.shape[2] > 0:
            Y_adj -= jnp.sum(V * beta, axis=2)
        H = shrink_lambda(jnp.linalg.lstsq(X_tilde, jnp.dot(Y_adj, Z_tilde))[0], lambda_H)

        # Update gamma, delta, and beta using coordinate descent
        Y_adj = Y - L - jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))
        if V is not None and V.shape[2] > 0:
            Y_adj -= jnp.sum(V * beta, axis=2)
        gamma = jnp.mean(Y_adj, axis=1)
        delta = jnp.mean(Y_adj - jnp.outer(gamma, jnp.ones(T)), axis=0)
        if V is not None and V.shape[2] > 0:
            beta = jnp.linalg.lstsq(V.reshape(N * T, -1), Y_adj.reshape(N * T))[0]

        # Calculate and print objective function value if verbose
        if verbose:
            obj_value = objective_function(Y, L, Omega_inv=jnp.linalg.inv(Omega), gamma=gamma, delta=delta,
                                           beta=beta, H=H, X=X, Z=Z, V=V)
            print_with_timestamp(f"Iteration {iteration + 1}, Objective value: {obj_value:.6f}")

    results = {}
    if return_tau:
        tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
        results['tau'] = tau
        if verbose:
            print(f"Estimated treatment effect (tau): {tau:.4f}")
    if return_lambda:
        results['lambda_L'] = lambda_L
        results['lambda_H'] = lambda_H
    if return_completed_L:
        results['L'] = L
    if return_completed_Y:
        Y_completed = L + jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T)) + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(
            jnp.ones(N), delta)
        if V is not None and V.shape[2] > 0:
            Y_completed += jnp.sum(V * beta, axis=2)
        results['Y_completed'] = Y_completed
    if return_fixed_effects:
        results['gamma'] = gamma
        results['delta'] = delta
    if return_covariate_coefficients:
        results['beta'] = beta
        results['H'] = H

    return MCNNMResults(**results)


def check_inputs(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None, V: Optional[Array] = None,
                 Omega: Optional[Array] = None) -> None:
    """
    Checks the validity of the input arrays and raises appropriate errors if the inputs are invalid.

    Args:
        Y: The observed outcome matrix of shape (N, T).
        W: The binary treatment matrix of shape (N, T).
        X: The unit-specific covariates matrix of shape (N, P). If None, unit-specific covariates are not included.
        Z: The time-specific covariates matrix of shape (T, Q). If None, time-specific covariates are not included.
        V: The unit-time specific covariates tensor of shape (N, T, J). If None, unit-time specific covariates are not included.
        Omega: The autocorrelation matrix of shape (T, T). If None, no autocorrelation is assumed.

    Raises:
        ValueError: If the shapes of the input arrays are invalid or inconsistent.
    """
    N, T = Y.shape
    if W.shape != (N, T):
        raise ValueError("The shape of W must match the shape of Y.")
    if X is not None and X.shape[0] != N:
        raise ValueError("The number of rows in X must match the number of rows in Y.")
    if Z is not None and Z.shape[0] != T:
        raise ValueError("The number of rows in Z must match the number of columns in Y.")
    if V is not None and V.shape[:2] != (N, T):
        raise ValueError("The first two dimensions of V must match the shape of Y.")
    if Omega is not None and Omega.shape != (T, T):
        raise ValueError("The shape of Omega must be (T, T).")


def estimate(
    Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None, V: Optional[Array] = None,
    Omega: Optional[Array] = None, lambda_L: Optional[float] = None, lambda_H: Optional[float] = None,
    return_tau: bool = True, return_lambda: bool = True,
    return_completed_L: bool = True, return_completed_Y: bool = True,
    return_fixed_effects: bool = False, return_covariate_coefficients: bool = False,
    max_iter: int = 1000, tol: float = 1e-4, verbose: bool = False
) -> MCNNMResults:
    """
    Estimates the MC-NNM model and returns the selected outputs.

    Args:
        Y: The observed outcome matrix.
        W: The binary treatment matrix.
        X: The matrix of unit and time specific covariates. If None, covariates are not included.
        Z: The time-specific covariates matrix. If None, time-specific covariates are not included.
        V: The unit-time specific covariates tensor. If None, unit-time specific covariates are not included.
        Omega: The autocorrelation matrix. If None, no autocorrelation is assumed.
        lambda_L: The regularization parameter for the nuclear norm of L. If None, it is selected via cross-validation.
        lambda_H: The regularization parameter for the element-wise L1 norm of H. If None, it is selected via cross-validation.
        return_tau: Whether to return the average treatment effect (tau) for the treated units.
        return_lambda: Whether to return the optimal regularization parameters lambda_L and lambda_H.
        return_completed_L: Whether to return the completed low-rank matrix L.
        return_completed_Y: Whether to return the completed outcome matrix Y.
        return_fixed_effects: Whether to return the estimated fixed effects (gamma and delta).
        return_covariate_coefficients: Whether to return the estimated covariate coefficients (beta and H).
        max_iter: The maximum number of iterations for the algorithm.
        tol: The tolerance for the convergence of the algorithm.

    Returns:
        A named tuple (MCNNMResults) containing the selected outputs.

    Raises:
        ValueError: If the shapes of the input arrays are invalid or inconsistent.
    """
    check_inputs(Y, W, X, Z, V, Omega)
    if verbose:
        print_with_timestamp("Estimating MC-NNM...")

    return fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, return_tau, return_lambda,
               return_completed_L, return_completed_Y, return_fixed_effects, return_covariate_coefficients,
               max_iter, tol, verbose)


@jax.jit
def complete_matrix(
    Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None, V: Optional[Array] = None,
    Omega: Optional[Array] = None, lambda_L: Optional[float] = None, lambda_H: Optional[float] = None,
    max_iter: int = 1000, tol: float = 1e-4
) -> Array:
    """
    Completes the missing values in the outcome matrix Y using the MC-NNM model.

    Args:
        Y: The observed outcome matrix.
        W: The binary treatment matrix.
        X: The matrix of unit and time specific covariates. If None, covariates are not included.
        Z: The time-specific covariates matrix. If None, time-specific covariates are not included.
        V: The unit-time specific covariates tensor. If None, unit-time specific covariates are not included.
        Omega: The autocorrelation matrix. If None, no autocorrelation is assumed.
        lambda_L: The regularization parameter for the nuclear norm of L. If None, it is selected via cross-validation.
        lambda_H: The regularization parameter for the element-wise L1 norm of H. If None, it is selected via cross-validation.
        max_iter: The maximum number of iterations for the algorithm.
        tol: The tolerance for the convergence of the algorithm.

    Returns:
        The completed outcome matrix Y.

    Raises:
        ValueError: If the shapes of the input arrays are invalid or inconsistent.
    """
    check_inputs(Y, W, X, Z, V, Omega)
    results = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H,
                  return_tau=False, return_lambda=False,
                  return_completed_L=False, return_completed_Y=True,
                  max_iter=max_iter, tol=tol)
    return results.Y_completed
