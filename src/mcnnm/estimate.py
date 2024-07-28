import jax
import jax.numpy as jnp
from typing import Optional, Tuple, NamedTuple
from .types import Array
from mcnnm.util import *
from jax import lax


def update_L(Y_adj: Array, L: Array, Omega: Array, O: Array, lambda_L: float) -> Array:
    """
    Update the low-rank matrix L in the MC-NNM algorithm.

    Args:
        Y_adj (Array): The adjusted outcome matrix.
        L (Array): The current estimate of the low-rank matrix.
        Omega (Array): The autocorrelation matrix.
        O (Array): The binary mask for observed entries.
        lambda_L (float): The regularization parameter for L.

    Returns:
        Array: The updated low-rank matrix L.
    """
    Y_adj_Omega = jnp.dot(Y_adj, Omega)
    L_new = jnp.where(O, Y_adj_Omega, L)
    return shrink_lambda(L_new, lambda_L * jnp.sum(O) / 2)


def update_H(X_tilde: Array, Y_adj: Array, Z_tilde: Array, lambda_H: float) -> Array:
    """
    Update the covariate coefficient matrix H in the MC-NNM algorithm.

    Args:
        X_tilde (Array): The augmented unit-specific covariates matrix.
        Y_adj (Array): The adjusted outcome matrix.
        Z_tilde (Array): The augmented time-specific covariates matrix.
        lambda_H (float): The regularization parameter for H.

    Returns:
        Array: The updated covariate coefficient matrix H.
    """
    H_unreg = jnp.linalg.lstsq(X_tilde, jnp.dot(Y_adj, Z_tilde))[0]
    return shrink_lambda(H_unreg, lambda_H)



def update_gamma_delta_beta(Y_adj: Array, V: Array) -> Tuple[Array, Array, Array]:
    """
    Update the fixed effects (gamma, delta) and unit-time specific covariate coefficients (beta).

    Args:
        Y_adj (Array): The adjusted outcome matrix.
        V (Array): The unit-time specific covariates tensor.

    Returns:
        Tuple[Array, Array, Array]: Updated gamma, delta, and beta arrays.
    """
    N, T = Y_adj.shape
    gamma = jnp.mean(Y_adj, axis=1)
    delta = jnp.mean(Y_adj - gamma[:, jnp.newaxis], axis=0)

    if V.size > 0:
        V_flat = V.reshape(-1, V.shape[-1])
        Y_adj_flat = Y_adj.reshape(-1)
        beta = jnp.linalg.lstsq(V_flat, Y_adj_flat)[0]
    else:
        beta = jnp.array([])

    return gamma, delta, beta


def fit_step(Y: Array, W: Array, X_tilde: Array, Z_tilde: Array, V: Array, Omega: Array,
             lambda_L: float, lambda_H: float, L: Array, H: Array, gamma: Array, delta: Array, beta: Array) -> Tuple:
    """
    Perform one step of the MC-NNM fitting algorithm.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X_tilde (Array): The augmented unit-specific covariates matrix.
        Z_tilde (Array): The augmented time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_L (float): The regularization parameter for L.
        lambda_H (float): The regularization parameter for H.
        L (Array): The current estimate of the low-rank matrix.
        H (Array): The current estimate of the covariate coefficient matrix.
        gamma (Array): The current estimate of unit fixed effects.
        delta (Array): The current estimate of time fixed effects.
        beta (Array): The current estimate of unit-time specific covariate coefficients.

    Returns:
        Tuple: Updated estimates of L, H, gamma, delta, and beta.
    """
    O = (W == 0)
    Y_adj_base = Y - gamma[:, jnp.newaxis] - delta[jnp.newaxis, :]

    Y_adj = Y_adj_base - jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))
    Y_adj = Y_adj - jnp.sum(V * beta, axis=-1) if V.size > 0 else Y_adj
    L_new = update_L(Y_adj, L, Omega, O, lambda_L)

    Y_adj = Y_adj_base - L_new
    Y_adj = Y_adj - jnp.sum(V * beta, axis=-1) if V.size > 0 else Y_adj
    H_new = update_H(X_tilde, Y_adj, Z_tilde, lambda_H)

    Y_adj = Y_adj_base - L_new - jnp.dot(X_tilde, jnp.dot(H_new, Z_tilde.T))
    Y_adj = Y_adj - jnp.sum(V * beta, axis=-1) if V.size > 0 else Y_adj
    gamma_new, delta_new, beta_new = update_gamma_delta_beta(Y_adj, V)

    return L_new, H_new, gamma_new, delta_new, beta_new

def fit(Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array,
        lambda_L: float, lambda_H: float, initial_params: Tuple,
        max_iter: int, tol: float) -> Tuple:
    """
    Fit the MC-NNM model using the given parameters and data.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Array): The unit-specific covariates matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_L (float): The regularization parameter for L.
        lambda_H (float): The regularization parameter for H.
        initial_params (Tuple): Initial parameter estimates.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        Tuple: Final estimates of L, H, gamma, delta, and beta.
    """
    # Unpack initial parameters
    L, H, gamma, delta, beta = initial_params

    # Compute dimensions and augmented covariate matrices
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))

    # Ensure beta has the correct shape
    beta = jnp.zeros((V.shape[-1],)) if V.size > 0 else jnp.zeros((0,))

    # Define the condition function for the while loop
    def cond_fn(state):
        i, L, _, _, _, _, prev_L = state
        return (i < max_iter) & (jnp.linalg.norm(L - prev_L, ord='fro') >= tol)

    # Define the body function for the while loop
    def body_fn(state):
        i, L, H, gamma, delta, beta, prev_L = state
        L_new, H_new, gamma_new, delta_new, beta_new = fit_step(Y, W, X_tilde, Z_tilde, V, Omega, lambda_L, lambda_H, L,
                                                                H, gamma, delta, beta)
        return i + 1, L_new, H_new, gamma_new, delta_new, beta_new, L

    # Set the initial state of the while loop
    initial_state = (0, L, H, gamma, delta, beta, jnp.zeros_like(L))

    # Run the while loop until convergence or max iterations
    _, L, H, gamma, delta, beta, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return L, H, gamma, delta, beta



def compute_cv_loss(Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array,
                    lambda_L: float, lambda_H: float, max_iter: int, tol: float) -> float:
    """
    Compute the cross-validation loss for given regularization parameters.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Array): The unit-specific covariates matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_L (float): The regularization parameter for L.
        lambda_H (float): The regularization parameter for H.
        max_iter (int): Maximum number of iterations for fitting.
        tol (float): Convergence tolerance for fitting.

    Returns:
        float: The computed cross-validation loss.
    """
    N = Y.shape[0]
    loss = 0.0

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, 0.8, (N,))
    train_idx = jnp.where(mask)[0]
    test_idx = jnp.where(~mask)[0]

    Y_train, Y_test = Y[train_idx], Y[test_idx]
    W_train, W_test = W[train_idx], W[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    V_train, V_test = V[train_idx], V[test_idx]

    initial_params = initialize_params(Y_train, W_train, X_train, Z, V_train)

    L, H, gamma, delta, beta = fit(Y_train, W_train, X_train, Z, V_train, Omega,
                                   lambda_L, lambda_H, initial_params, max_iter, tol)

    Y_pred = (L[test_idx] + jnp.outer(gamma[test_idx], jnp.ones(Z.shape[0])) +
              jnp.outer(jnp.ones(test_idx.shape[0]), delta))

    if V_test.shape[2] > 0:
        Y_pred += jnp.sum(V_test * beta, axis=2)

    O_test = (W_test == 0)
    loss += jnp.sum((Y_test - Y_pred) ** 2 * O_test) / jnp.sum(O_test)
    return loss

def cross_validate(Y: Array, W: Array, X: Array, Z: Array, V: Array,
                   Omega: Array, lambda_grid: Array, max_iter: int, tol: float, K: int = 5) -> Tuple[float, float]:
    """
    Perform K-fold cross-validation to select optimal regularization parameters.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Array): The unit-specific covariates matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int): Maximum number of iterations for fitting.
        tol (float): Convergence tolerance for fitting.
        K (int): Number of folds for cross-validation. Default is 5.

    Returns:
        Tuple[float, float]: The optimal lambda_L and lambda_H values.
    """
    best_lambda_L = None
    best_lambda_H = None
    best_loss = jnp.inf

    for lambda_L, lambda_H in lambda_grid:
        loss = 0.0
        valid_folds = 0
        key = jax.random.PRNGKey(0)

        for k in range(K):
            mask = jax.random.bernoulli(key, 0.8, (Y.shape[0],))
            train_idx = jnp.where(mask)[0]
            test_idx = jnp.where(~mask)[0]

            if jnp.sum(W[test_idx] == 1) == 0:
                # print(f"Warning: No treated units in test set for fold {k + 1}, skipping fold")
                continue

            fold_loss = compute_cv_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H, max_iter, tol)
            # print(f"Fold loss for lambda_L={lambda_L}, lambda_H={lambda_H}: {fold_loss}")
            if jnp.isfinite(fold_loss):
                loss += fold_loss
                valid_folds += 1
            else:
                # print(f"Non-finite loss for lambda_L={lambda_L}, lambda_H={lambda_H}")
                pass

        if valid_folds > 0:
            loss /= valid_folds
            # print(f"Average loss for lambda_L={lambda_L}, lambda_H={lambda_H}: {loss}")
            if loss < best_loss:
                best_lambda_L = lambda_L
                best_lambda_H = lambda_H
                best_loss = loss
                # print(f"New best loss: {best_loss} for lambda_L={best_lambda_L}, lambda_H={best_lambda_H}")
        else:
            # print(f"No valid folds for lambda_L={lambda_L}, lambda_H={lambda_H}")
            pass

    if best_loss == jnp.inf:
        print("Warning: No valid loss found in cross_validate")
        return lambda_grid[0][0], lambda_grid[0][1]

    return best_lambda_L, best_lambda_H


def compute_time_based_loss(Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array,
                            lambda_L: float, lambda_H: float, max_iter: int, tol: float,
                            train_idx: Array, test_idx: Array) -> float:
    """
    Compute the time-based holdout loss for given regularization parameters.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Array): The unit-specific covariates matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_L (float): The regularization parameter for L.
        lambda_H (float): The regularization parameter for H.
        max_iter (int): Maximum number of iterations for fitting.
        tol (float): Convergence tolerance for fitting.
        train_idx (Array): Indices of training data.
        test_idx (Array): Indices of test data.

    Returns:
        float: The computed time-based holdout loss.
    """
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    W_train, W_test = W[train_idx], W[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    V_train, V_test = V[train_idx], V[test_idx]

    # print(f"Shapes: Y_train {Y_train.shape}, Y_test {Y_test.shape}, W_test {W_test.shape}")

    initial_params = initialize_params(Y_train, W_train, X_train, Z, V_train)
    L, H, gamma, delta, beta = fit(Y_train, W_train, X_train, Z, V_train, Omega,
                                   lambda_L, lambda_H, initial_params, max_iter, tol)

    # print(f"Fit results: L shape {L.shape}, gamma shape {gamma.shape}, delta shape {delta.shape}")

    Y_pred = (L[test_idx] + jnp.outer(gamma[test_idx], jnp.ones(Z.shape[0])) +
              jnp.outer(jnp.ones(test_idx.shape[0]), delta))

    if V_test.shape[2] > 0:
        Y_pred += jnp.sum(V_test * beta, axis=2)

    # print(f"Y_pred shape {Y_pred.shape}, Y_test shape {Y_test.shape}")

    O_test = (W_test == 0)
    # print(f"O_test shape {O_test.shape}, sum {jnp.sum(O_test)}")

    def true_fun(_):
        return jnp.inf

    def false_fun(_):
        return jnp.sum((Y_test - Y_pred) ** 2 * O_test) / jnp.sum(O_test)

    loss = lax.cond(
        jnp.sum(O_test) == 0,
        true_fun,
        false_fun,
        operand=None
    )

    # print(f"Computed loss: {loss}")
    return loss


def time_based_validate(Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array,
                        lambda_grid: Array, max_iter: int, tol: float,
                        window_size: Optional[int] = None, expanding_window: bool = False,
                        max_window_size: Optional[int] = None, n_folds: int = 5) -> Tuple[float, float]:
    """
    Perform time-based validation to select optimal regularization parameters.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Array): The unit-specific covariates matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int): Maximum number of iterations for fitting.
        tol (float): Convergence tolerance for fitting.
        window_size (Optional[int]): Size of the rolling window. Default is None.
        expanding_window (bool): Whether to use an expanding window. Default is False.
        max_window_size (Optional[int]): Maximum size of the expanding window. Default is None.
        n_folds (int): Number of folds for time-based validation. Default is 5.

    Returns:
        Tuple[float, float]: The optimal lambda_L and lambda_H values.
    """
    N, T = Y.shape

    if window_size is None:
        window_size = (T * 4) // 5

    if expanding_window and max_window_size is None:
        max_window_size = window_size

    best_lambda_L = None
    best_lambda_H = None
    best_loss = jnp.inf

    for lambda_L, lambda_H in lambda_grid:
        loss = 0.0
        valid_folds = 0
        t = window_size

        while t < T:
            if expanding_window:
                train_idx = jnp.arange(max(0, t - max_window_size), t)
            else:
                train_idx = jnp.arange(max(0, t - window_size), t)

            test_idx = jnp.arange(t, min(t + (T - window_size) // n_folds, T))
            fold_loss = 0.0

            for _ in range(n_folds):
                if test_idx[-1] == T:
                    break

                if jnp.sum(W[test_idx] == 0) == 0:
                    # print(f"Warning: No untreated units in test set for fold {_ + 1}, skipping fold")
                    continue

                fold_loss = compute_time_based_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H,
                                                    max_iter, tol, train_idx, test_idx)
                # print(f"Fold loss for lambda_L={lambda_L}, lambda_H={lambda_H}: {fold_loss}")
                if jnp.isfinite(fold_loss):
                    loss += fold_loss
                    valid_folds += 1
                else:
                    # print(f"Non-finite loss for lambda_L={lambda_L}, lambda_H={lambda_H}")
                    pass
                test_idx = jnp.arange(test_idx[-1], min(test_idx[-1] + (T - window_size) // n_folds, T))

            loss += fold_loss / n_folds
            t += (T - window_size) // n_folds

        loss /= (T - window_size) // ((T - window_size) // n_folds)

        if valid_folds > 0:
            loss /= valid_folds
            # print(f"Average loss for lambda_L={lambda_L}, lambda_H={lambda_H}: {loss}")
            if loss < best_loss:
                best_lambda_L = lambda_L
                best_lambda_H = lambda_H
                best_loss = loss
                # print(f"New best loss: {best_loss} for lambda_L={best_lambda_L}, lambda_H={best_lambda_H}")
        else:
            # print(f"No valid folds for lambda_L={lambda_L}, lambda_H={lambda_H}")
            pass
    if best_loss == jnp.inf:
        print("Warning: No valid loss found in time_based_validate")
        return lambda_grid[0][0], lambda_grid[0][1]

    return best_lambda_L, best_lambda_H


def compute_treatment_effect(Y: Array, L: Array, gamma: Array, delta: Array, beta: Array, H: Array,
                             X: Array, W: Array, Z: Array, V: Array) -> float:
    """
    Compute the average treatment effect.

    Args:
        Y (Array): The observed outcome matrix.
        L (Array): The estimated low-rank matrix.
        gamma (Array): The estimated unit fixed effects.
        delta (Array): The estimated time fixed effects.
        beta (Array): The estimated unit-time specific covariate coefficients.
        H (Array): The estimated covariate coefficient matrix.
        X (Array): The unit-specific covariates matrix.
        W (Array): The binary treatment matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.

    Returns:
        float: The estimated average treatment effect.
    """
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))
    Y_completed = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)

    if X.shape[1] > 0 and Z.shape[1] > 0:
        Y_completed += jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))

    if V.shape[2] > 0:
        Y_completed += jnp.sum(V * beta[None, None, :], axis=2)

    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    return tau

class MCNNMResults(NamedTuple):
    """
    A named tuple containing the results of the MC-NNM estimation.

    Attributes:
        tau (Optional[float]): The estimated average treatment effect.
        lambda_L (Optional[float]): The selected regularization parameter for L.
        lambda_H (Optional[float]): The selected regularization parameter for H.
        L (Optional[Array]): The estimated low-rank matrix.
        Y_completed (Optional[Array]): The completed outcome matrix.
        gamma (Optional[Array]): The estimated unit fixed effects.
        delta (Optional[Array]): The estimated time fixed effects.
        beta (Optional[Array]): The estimated unit-time specific covariate coefficients.
        H (Optional[Array]): The estimated covariate coefficient matrix.
    """
    tau: Optional[float] = None
    lambda_L: Optional[float] = None
    lambda_H: Optional[float] = None
    L: Optional[Array] = None
    Y_completed: Optional[Array] = None
    gamma: Optional[Array] = None
    delta: Optional[Array] = None
    beta: Optional[Array] = None
    H: Optional[Array] = None


def estimate(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None,
             V: Optional[Array] = None, Omega: Optional[Array] = None, lambda_L: Optional[float] = None,
             lambda_H: Optional[float] = None, n_lambda_L: int = 10, n_lambda_H: int = 10,
             return_tau: bool = True, return_lambda: bool = True,
             return_completed_L: bool = True, return_completed_Y: bool = True, return_fixed_effects: bool = False,
             return_covariate_coefficients: bool = False, max_iter: int = 1000, tol: float = 1e-4,
             verbose: bool = False, validation_method: str = 'cv', K: int = 5, window_size: Optional[int] = None,
             expanding_window: bool = False, max_window_size: Optional[int] = None) -> MCNNMResults:
    """
    Estimate the MC-NNM model and return results.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Optional[Array]): The unit-specific covariates matrix. Default is None.
        Z (Optional[Array]): The time-specific covariates matrix. Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
        Omega (Optional[Array]): The autocorrelation matrix. Default is None.
        lambda_L (Optional[float]): The regularization parameter for L. If None, it will be selected via validation.
        lambda_H (Optional[float]): The regularization parameter for H. If None, it will be selected via validation.
        n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
        n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
        return_tau (bool): Whether to return the estimated treatment effect. Default is True.
        return_lambda (bool): Whether to return the selected lambda values. Default is True.
        return_completed_L (bool): Whether to return the completed low-rank matrix. Default is True.
        return_completed_Y (bool): Whether to return the completed outcome matrix. Default is True.
        return_fixed_effects (bool): (For debugging) Whether to return the estimated fixed effects. Default is False.
        return_covariate_coefficients (bool): (For debugging) Whether to return the estimated covariate coefficients.
        Default is False.
        max_iter (int): Maximum number of iterations for fitting. Default is 1000.
        tol (float): Convergence tolerance for fitting. Default is 1e-4.
        verbose (bool): Whether to print progress messages. Default is False.
        validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
        K (int): Number of folds for cross-validation. Default is 5.
        window_size (Optional[int]): Size of the rolling window for time-based validation. Default is None.
        expanding_window (bool): Whether to use an expanding window for time-based validation. Default is False.
        max_window_size (Optional[int]): Maximum size of the expanding window for time-based validation. Default is None.

    Returns:
        MCNNMResults: A named tuple containing the requested results.
    """
    X, Z, V, Omega = check_inputs(Y, W, X, Z, V, Omega)
    N, T = Y.shape

    if lambda_L is None or lambda_H is None:
        if validation_method == 'cv':
            if verbose:
                print_with_timestamp("Cross-validating lambda_L, lambda_H")
            lambda_grid = jnp.array(jnp.meshgrid(propose_lambda(None, n_lambda_L), propose_lambda(None, n_lambda_L))).T.reshape(-1, 2)
            lambda_L, lambda_H = cross_validate(Y, W, X, Z, V, Omega, lambda_grid, K=K, max_iter = max_iter // 10, tol=tol * 10)
        elif validation_method == 'holdout':
            if T < 5:
                raise ValueError("The matrix does not have enough columns for time-based validation. "
                                 "Please increase the number of time periods or use cross-validation")
            if verbose:
                print_with_timestamp("Selecting lambda_L, lambda_H using time-based holdout validation")
            lambda_grid = jnp.array(
                jnp.meshgrid(propose_lambda(None, n_lambda_L), propose_lambda(None, n_lambda_H))).T.reshape(-1, 2)
            lambda_L, lambda_H = time_based_validate(Y, W, X, Z, V, Omega, lambda_grid, max_iter // 10, tol * 10,
                                                     window_size, expanding_window, max_window_size)
        else:
            raise ValueError("Invalid validation_method. Choose 'cv' or 'time'.")

        if verbose:
            print_with_timestamp(f"Selected lambda_L: {lambda_L:.4f}, lambda_H: {lambda_H:.4f}")

    initial_params = initialize_params(Y, W, X, Z, V)
    L, H, gamma, delta, beta = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, initial_params, max_iter, tol)

    results = {}
    if return_tau:
        tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
        results['tau'] = tau
    if return_lambda:
        results['lambda_L'] = lambda_L
        results['lambda_H'] = lambda_H
    if return_completed_L:
        results['L'] = L
    if return_completed_Y:
        Y_completed = L + jnp.dot(jnp.hstack((X, jnp.eye(N))), jnp.dot(H, jnp.hstack((Z, jnp.eye(T))).T))
        Y_completed += jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)
        if V.shape[2] > 0:
            Y_completed += jnp.sum(V * beta, axis=2)
        results['Y_completed'] = Y_completed
    if return_fixed_effects:
        results['gamma'] = gamma
        results['delta'] = delta
    if return_covariate_coefficients:
        results['beta'] = beta
        results['H'] = H

    return MCNNMResults(**results)


def complete_matrix(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None,
                    V: Optional[Array] = None, Omega: Optional[Array] = None, lambda_L: Optional[float] = None,
                    lambda_H: Optional[float] = None, n_lambda_L: int = 10, n_lambda_H: int = 10,
                    max_iter: int = 1000, tol: float = 1e-4, verbose: bool = False,
                    validation_method: str = 'cv', K: int = 5 , window_size: Optional[int] = None,
                    expanding_window: bool = False, max_window_size: Optional[int] = None) -> Tuple[Array, float, float]:
    """
    Complete the matrix Y using the MC-NNM model and return the optimal regularization parameters.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Optional[Array]): The unit-specific covariates matrix. Default is None.
        Z (Optional[Array]): The time-specific covariates matrix. Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
        Omega (Optional[Array]): The autocorrelation matrix. Default is None.
        lambda_L (Optional[float]): The regularization parameter for L. If None, it will be selected via validation.
        lambda_H (Optional[float]): The regularization parameter for H. If None, it will be selected via validation.
        n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
        n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
        max_iter (int): Maximum number of iterations for fitting. Default is 1000.
        tol (float): Convergence tolerance for fitting. Default is 1e-4.
        verbose (bool): Whether to print progress messages. Default is False.
        validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
        K (int): Number of folds for cross-validation. Default is 5.
        window_size (Optional[int]): Size of the rolling window for time-based validation. Default is None.
        expanding_window (bool): Whether to use an expanding window for time-based validation. Default is False.
        max_window_size (Optional[int]): Maximum size of the expanding window for time-based validation. Default is None.

    Returns:
        Tuple[Array, float, float]: A tuple containing:
            - The completed outcome matrix
            - The optimal lambda_L value
            - The optimal lambda_H value
    """
    results = estimate(Y, W, X, Z, V, Omega, lambda_L, lambda_H, n_lambda_L, n_lambda_H,
                       return_tau=False, return_lambda=True, return_completed_L=False, return_completed_Y=True,
                       return_fixed_effects=False, return_covariate_coefficients=False,
                       max_iter=max_iter, tol=tol, verbose=verbose, validation_method=validation_method, K=K,
                       window_size=window_size, expanding_window=expanding_window, max_window_size=max_window_size)
    return results.Y_completed, results.lambda_L, results.lambda_H