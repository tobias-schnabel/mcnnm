import jax
import jax.numpy as jnp
from typing import Optional, Tuple, NamedTuple, cast
from .types import Array, Scalar
from mcnnm.util import (
    shrink_lambda,
    initialize_params,
    propose_lambda,
    check_inputs,
    generate_time_based_validate_defaults,
)
from jax import lax
from functools import partial


def update_L(Y_adj: Array, L: Array, Omega: Array, O: Array, lambda_L: Scalar) -> Array:
    """
    Update the low-rank matrix L in the MC-NNM algorithm.

    Args:
        Y_adj (Array): The adjusted outcome matrix.
        L (Array): The current estimate of the low-rank matrix.
        Omega (Array): The autocorrelation matrix.
        O (Array): The binary mask for observed entries.
        lambda_L (Scalar): The regularization parameter for L.

    Returns:
        Array: The updated low-rank matrix L.
    """
    Y_adj_Omega = jnp.einsum("ij,jk->ik", Y_adj, Omega[: Y_adj.shape[1], : Y_adj.shape[1]])
    L_new = jnp.where(O, Y_adj_Omega, L)
    lambda_val = lambda_L * jnp.sum(O) / 2
    return shrink_lambda(L_new, lambda_val)


def update_H(X_tilde: Array, Y_adj: Array, Z_tilde: Array, lambda_H: Scalar) -> Array:
    """
    Update the covariate coefficient matrix H in the MC-NNM algorithm.

    Args:
        X_tilde (Array): The augmented unit-specific covariates matrix.
        Y_adj (Array): The adjusted outcome matrix.
        Z_tilde (Array): The augmented time-specific covariates matrix.
        lambda_H (Scalar): The regularization parameter for H.

    Returns:
        Array: The updated covariate coefficient matrix H.
    """
    H_unreg = jnp.linalg.lstsq(X_tilde, jnp.dot(Y_adj, Z_tilde))[0]
    return shrink_lambda(H_unreg, lambda_H)


def update_gamma_delta_beta(
    Y_adj: jnp.ndarray, V: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update the fixed effects (gamma, delta) and unit-time specific covariate coefficients (beta).

    Args:
        Y_adj (jnp.ndarray): The adjusted outcome matrix.
        V (jnp.ndarray): The unit-time specific covariates tensor.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Updated gamma, delta, and beta arrays.
    """
    N, T = Y_adj.shape
    gamma = jnp.mean(Y_adj, axis=1)
    delta = jnp.mean(Y_adj - gamma[:, jnp.newaxis], axis=0)

    # We assume V is non-empty and compatible due to initialize_params
    beta = compute_beta(V, Y_adj)

    return gamma, delta, beta


def compute_beta(V: Array, Y_adj: Array) -> Array:
    """
    Compute beta coefficients for unit-time specific covariates.

    Args:
        V (Array): The unit-time specific covariates tensor.
        Y_adj (Array): The adjusted outcome matrix.

    Returns:
        Array: The computed beta coefficients.
    """

    def solve_lstsq(_):
        V_flat = V.reshape(-1, V.shape[-1])
        Y_adj_flat = Y_adj.reshape(-1)
        return jnp.linalg.lstsq(V_flat, Y_adj_flat, rcond=None)[0]

    def return_empty(_):
        # Return an array of the same shape as solve_lstsq would return but filled with zeros
        return jnp.zeros(V.shape[-1])  # pragma: no cover

    beta = lax.cond(V.size > 0, solve_lstsq, return_empty, operand=None)
    return beta


def fit_step(
    Y: Array,
    W: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    Omega: Array,
    lambda_L: Scalar,
    lambda_H: Scalar,
    L: Array,
    H: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Perform one step of the MC-NNM fitting algorithm.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X_tilde (Array): The augmented unit-specific covariates matrix.
        Z_tilde (Array): The augmented time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        Omega (Array): The autocorrelation matrix.
        lambda_L (Scalar): The regularization parameter for L.
        lambda_H (Scalar): The regularization parameter for H.
        L (Array): The current estimate of the low-rank matrix.
        H (Array): The current estimate of the covariate coefficient matrix.
        gamma (Array): The current estimate of unit fixed effects.
        delta (Array): The current estimate of time fixed effects.
        beta (Array): The current estimate of unit-time specific covariate coefficients.

    Returns:
        Tuple[Array, Array, Array, Array, Array]: Updated estimates of L, H, gamma, delta, and beta.
    """
    O = jnp.array(W == 0, dtype=jnp.int32)
    Y_adj_base = Y - gamma[:, jnp.newaxis] - delta[jnp.newaxis, :]

    def adjust_Y_adj(Y_adj_base, X_tilde, H, Z_tilde, V, beta):
        Y_adj = Y_adj_base - jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))
        Y_adj = jax.lax.cond(
            V.size > 0, lambda x: x - jnp.sum(V * beta, axis=-1), lambda x: x, Y_adj
        )
        return Y_adj

    Y_adj = adjust_Y_adj(Y_adj_base, X_tilde, H, Z_tilde, V, beta)
    L_new = update_L(Y_adj, L, Omega, O, lambda_L)

    Y_adj = adjust_Y_adj(Y_adj_base, X_tilde, H, Z_tilde, V, beta)
    H_new = update_H(X_tilde, Y_adj, Z_tilde, lambda_H)

    Y_adj = Y_adj_base - L_new - jnp.dot(X_tilde, jnp.dot(H_new, Z_tilde.T))
    Y_adj = jax.lax.cond(V.size > 0, lambda x: x - jnp.sum(V * beta, axis=-1), lambda x: x, Y_adj)
    gamma_new, delta_new, beta_new = update_gamma_delta_beta(Y_adj, V)

    return L_new, H_new, gamma_new, delta_new, beta_new


def fit(
    Y: Array,
    W: Array,
    X: Array,
    Z: Array,
    V: Array,
    Omega: Array,
    lambda_L: Scalar,
    lambda_H: Scalar,
    initial_params: Tuple[Array, Array, Array, Array, Array],
    max_iter: int,
    tol: Scalar,
) -> Tuple[Array, Array, Array, Array, Array]:
    # Unpack initial parameters
    L, H, gamma, delta, beta = initial_params

    # Compute dimensions and augmented covariate matrices
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))

    # Ensure beta has the correct shape using JAX conditional
    beta = jax.lax.cond(
        V.size > 0,
        lambda _: beta,
        lambda _: jnp.zeros_like(beta),  # This ensures the same shape as the true branch
        operand=None,
    )

    # Define the condition function for the while loop
    def cond_fn(state):
        i, L, _, _, _, _, prev_L = state
        return (i < max_iter) & (jnp.linalg.norm(L - prev_L, ord="fro") >= tol)

    # Define the body function for the while loop
    def body_fn(state):
        i, L, H, gamma, delta, beta, prev_L = state
        L_new, H_new, gamma_new, delta_new, beta_new = fit_step(
            Y, W, X_tilde, Z_tilde, V, Omega, lambda_L, lambda_H, L, H, gamma, delta, beta
        )
        return i + 1, L_new, H_new, gamma_new, delta_new, beta_new, L

    # Set the initial state of the while loop
    initial_state = (
        jnp.array(0),
        L.astype(jnp.float32),
        H.astype(jnp.float32),
        gamma.astype(jnp.float32),
        delta.astype(jnp.float32),
        beta.astype(jnp.float32),
        jnp.zeros_like(L, dtype=jnp.float32),
    )

    # Run the while loop until convergence or max iterations
    _, L, H, gamma, delta, beta, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return L, H, gamma, delta, beta


def compute_cv_loss(
    Y: Array,
    W: Array,
    X: Array,
    Z: Array,
    V: Array,
    Omega: Array,
    lambda_L: float,
    lambda_H: float,
    max_iter: int,
    tol: float,
) -> Scalar:
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
    loss = jnp.array(0.0)

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, 0.8, (N,))

    train_idx = jnp.nonzero(mask, size=N)[0]
    test_idx = jnp.nonzero(~mask, size=N)[0]

    Y_train, Y_test = Y[train_idx], Y[test_idx]
    W_train, W_test = W[train_idx], W[test_idx]
    X_train = X[train_idx]
    V_train, V_test = V[train_idx], V[test_idx]

    initial_params = initialize_params(Y_train, X_train, Z, V_train)

    L, H, gamma, delta, beta = fit(
        Y_train,
        W_train,
        X_train,
        Z,
        V_train,
        Omega,
        lambda_L,
        lambda_H,
        initial_params,
        max_iter,
        tol,
    )

    Y_pred = (
        L[test_idx]
        + jnp.outer(gamma[test_idx], jnp.ones(Z.shape[0]))
        + jnp.outer(jnp.ones(test_idx.shape[0]), delta)
    )

    Y_pred = jax.lax.cond(
        V_test.shape[2] > 0,
        lambda Y_pred: Y_pred + jnp.sum(V_test * beta, axis=2),
        lambda Y_pred: Y_pred,
        Y_pred,
    )

    O_test = W_test == 0
    loss += jnp.sum((Y_test - Y_pred) ** 2 * O_test) / jnp.sum(O_test)
    return loss


def cross_validate(
    Y: Array,
    W: Array,
    X: Array,
    Z: Array,
    V: Array,
    Omega: Array,
    lambda_grid: Array,
    max_iter: int,
    tol: float,
    K: int = 5,
) -> Tuple[Scalar, Scalar]:
    """
    Perform K-fold cross-validation to select optimal regularization parameters for the MC-NNM model.

    This function splits the data into K folds along the unit dimension, trains the model on K-1 folds,
    and evaluates it on the remaining fold. This process is repeated for all folds and all lambda pairs
    in the lambda_grid. The lambda pair that yields the lowest average loss across all folds is selected.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).
        Omega (Array): The autocorrelation matrix of shape (T, T).
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int): Maximum number of iterations for fitting the model in each fold.
        tol (float): Convergence tolerance for fitting the model in each fold.
        K (int, optional): Number of folds for cross-validation. Default is 5.

    Returns:
        Tuple[Scalar, Scalar]: A tuple containing the optimal lambda_L and lambda_H values.

    Note:
        This function uses JAX's `vmap` for efficient computation across different lambda pairs.
        If all losses are infinite (e.g., due to numerical instability), it returns the middle
        lambda pair from the lambda_grid as a fallback.
    """
    N = Y.shape[0]
    fold_size = N // K

    def loss_fn(lambda_L_H):
        lambda_L, lambda_H = lambda_L_H

        def fold_loss(k):
            mask = (jnp.arange(N) >= k * fold_size) & (jnp.arange(N) < (k + 1) * fold_size)

            Y_train = jnp.where(mask[:, None], jnp.zeros_like(Y), Y)
            Y_test = jnp.where(mask[:, None], Y, jnp.zeros_like(Y))
            W_train = jnp.where(mask[:, None], jnp.zeros_like(W), W)
            W_test = jnp.where(mask[:, None], W, jnp.zeros_like(W))
            X_train = jnp.where(mask[:, None], jnp.zeros_like(X), X)
            V_train = jnp.where(mask[:, None, None], jnp.zeros_like(V), V)
            V_test = jnp.where(mask[:, None, None], V, jnp.zeros_like(V))

            initial_params = initialize_params(Y_train, X_train, Z, V_train)

            L, H, gamma, delta, beta = fit(
                Y_train,
                W_train,
                X_train,
                Z,
                V_train,
                Omega,
                lambda_L,
                lambda_H,
                initial_params,
                max_iter,
                tol,
            )

            Y_pred = L + jnp.outer(gamma, jnp.ones(Z.shape[0])) + jnp.outer(jnp.ones(N), delta)

            Y_pred = jax.lax.cond(
                V_test.shape[2] > 0,
                lambda Y_pred: Y_pred + jnp.sum(V_test * beta, axis=2),
                lambda Y_pred: Y_pred,
                Y_pred,
            )

            O_test = W_test == 0
            loss = jnp.sum((Y_test - Y_pred) ** 2 * O_test) / (jnp.sum(O_test) + 1e-10)
            return loss

        # losses = jax.lax.map(fold_loss, jnp.arange(K))
        def fold_loss_wrapper(i, acc):
            return acc + fold_loss(i)

        losses = jax.lax.fori_loop(0, K, fold_loss_wrapper, 0.0)

        return jnp.mean(losses)

    losses = jax.vmap(loss_fn)(lambda_grid)

    def select_best_lambda(_):
        best_idx = jnp.argmin(losses)
        return lambda_grid[best_idx]

    def use_default_lambda(_):  # pragma: no cover
        mid_idx = len(lambda_grid) // 2
        return lambda_grid[mid_idx]

    best_lambda_L_H = jax.lax.cond(
        jnp.any(jnp.isfinite(losses)), select_best_lambda, use_default_lambda, operand=None
    )

    return best_lambda_L_H[0], best_lambda_L_H[1]


@partial(jax.jit, static_argnums=(13, 14))
def time_based_validate(
    Y: Array,
    W: Array,
    X: Array,
    Z: Array,
    V: Array,
    Omega: Array,
    lambda_grid: Array,
    max_iter: int,
    tol: Scalar,
    initial_window: int,
    step_size: int,
    horizon: int,
    K: int,
    T: int,
    max_window_size: Optional[int] = None,
) -> Tuple[Scalar, Scalar]:
    """
    Perform time-based validation to select optimal regularization parameters.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).
        Omega (Array): The autocorrelation matrix of shape (T, T).
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int): Maximum number of iterations for fitting.
        tol (Scalar): Convergence tolerance for fitting.
        initial_window (int): Number of initial time periods to use for first training set.
        step_size (int): Number of time periods to move forward for each split.
        horizon (int): Number of future time periods to predict (forecast horizon).
        K (int): Number of folds to use in the time-based validation.
        max_window_size (Optional[int]): Maximum size of the window to consider. If None, use all data.

    Returns:
        Tuple[Scalar, Scalar]: The optimal lambda_L and lambda_H values.
    """
    N, foo = Y.shape

    if max_window_size is not None:
        T = min(T, max_window_size)
        Y = Y[:, -T:]
        W = W[:, -T:]
        Z = Z[-T:]
        V = V[:, -T:]
        Omega = Omega[-T:, -T:]

    def compute_fold_loss(
        train_end: int, test_end: int, lambda_L: Scalar, lambda_H: Scalar
    ) -> Scalar:
        train_mask = jnp.arange(T) < train_end
        test_mask = (jnp.arange(T) >= train_end) & (jnp.arange(T) < test_end)

        Y_train = jnp.where(train_mask[None, :], Y, 0.0)
        Y_test = jnp.where(test_mask[None, :], Y, 0.0)
        W_train = jnp.where(train_mask[None, :], W, 0)
        W_test = jnp.where(test_mask[None, :], W, 0)
        Z_train = jnp.where(train_mask[:, None], Z, 0.0)
        V_train = jnp.where(train_mask[None, :, None], V, 0.0)
        V_test = jnp.where(test_mask[None, :, None], V, 0.0)

        initial_params = initialize_params(Y_train, X, Z_train, V_train)
        L, H, gamma, delta, beta = fit(
            Y_train,
            W_train,
            X,
            Z_train,
            V_train,
            Omega,
            lambda_L,
            lambda_H,
            initial_params,
            max_iter,
            tol,
        )

        Y_pred = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)
        Y_pred = jax.lax.cond(
            V_test.shape[2] > 0,
            lambda _: Y_pred + jnp.sum(V_test * beta, axis=2),
            lambda _: Y_pred,
            operand=None,
        )

        O_test = W_test == 0
        loss = jnp.sum((Y_test - Y_pred) ** 2 * O_test) / (jnp.sum(O_test) + 1e-10)
        return loss

    def compute_lambda_loss(lambda_pair: Array) -> Scalar:
        lambda_L, lambda_H = lambda_pair

        def body_fun(i, acc):
            total_loss, count = acc
            train_end = initial_window + i * step_size
            test_end = jnp.int32(jnp.minimum(train_end + horizon, T))
            fold_loss = jax.lax.cond(
                test_end <= train_end,
                lambda _: jnp.inf,
                lambda _: compute_fold_loss(train_end, test_end, lambda_L, lambda_H),
                operand=None,
            )
            new_total_loss = total_loss + fold_loss
            new_count = count + 1
            return new_total_loss, new_count

        initial_acc = (0.0, 0)
        total_loss, count = jax.lax.fori_loop(0, K, body_fun, initial_acc)

        return jax.lax.cond(
            count > 0, lambda _: total_loss / count, lambda _: jnp.inf, operand=None
        )

    losses = jax.vmap(compute_lambda_loss)(lambda_grid)

    def select_best_lambda(_):
        best_idx = jnp.argmin(losses)
        return (lambda_grid[best_idx, 0], lambda_grid[best_idx, 1])

    def use_default_lambda(_):
        mid_idx = len(lambda_grid) // 2
        return (lambda_grid[mid_idx, 0], lambda_grid[mid_idx, 1])

    return jax.lax.cond(
        jnp.any(jnp.isfinite(losses)), select_best_lambda, use_default_lambda, operand=None
    )


def compute_treatment_effect(
    Y: Array,
    L: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    H: Array,
    X: Array,
    W: Array,
    Z: Array,
    V: Array,
) -> Scalar:
    """
    Compute the average treatment effect using the MC-NNM model estimates.

    This function calculates the difference between the observed outcomes and the
    completed (counterfactual) outcomes for treated units, then averages this
    difference to estimate the average treatment effect.

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
        Scalar: The estimated average treatment effect.
    """
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))
    Y_completed = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)

    Y_completed = jax.lax.cond(
        (X.shape[1] > 0) & (Z.shape[1] > 0),
        lambda _: Y_completed + jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T)),
        lambda _: Y_completed,
        operand=None,
    )

    Y_completed = jax.lax.cond(
        V.shape[2] > 0,
        lambda _: Y_completed + jnp.sum(V * beta[None, None, :], axis=2),
        lambda _: Y_completed,
        operand=None,
    )

    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    return tau


class MCNNMResults(NamedTuple):
    """
    A named tuple containing the results of the MC-NNM (Matrix Completion with Nuclear Norm Minimization) estimation.

    This class encapsulates all the key outputs from the MC-NNM model, including
    the estimated treatment effect, selected regularization parameters, and
    various estimated matrices and vectors.

    Attributes:
        tau (Optional[Scalar]): The estimated average treatment effect.
        lambda_L (Optional[Scalar]): The selected regularization parameter for the low-rank matrix L.
        lambda_H (Optional[Scalar]): The selected regularization parameter for the covariate coefficient matrix H.
        L (Optional[Array]): The estimated low-rank matrix.
        Y_completed (Optional[Array]): The completed outcome matrix (including counterfactuals).
        gamma (Optional[Array]): The estimated unit fixed effects.
        delta (Optional[Array]): The estimated time fixed effects.
        beta (Optional[Array]): The estimated unit-time specific covariate coefficients.
        H (Optional[Array]): The estimated covariate coefficient matrix.

    All attributes are optional and initialized to None by default.
    """

    tau: Optional[Scalar] = None
    lambda_L: Optional[Scalar] = None
    lambda_H: Optional[Scalar] = None
    L: Optional[Array] = None
    Y_completed: Optional[Array] = None
    gamma: Optional[Array] = None
    delta: Optional[Array] = None
    beta: Optional[Array] = None
    H: Optional[Array] = None


def estimate(
    Y: Array,
    W: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
    lambda_L: Optional[Scalar] = None,
    lambda_H: Optional[Scalar] = None,
    n_lambda_L: int = 10,
    n_lambda_H: int = 10,
    return_tau: bool = True,
    return_lambda: bool = True,
    return_completed_L: bool = True,
    return_completed_Y: bool = True,
    return_fixed_effects: bool = False,
    return_covariate_coefficients: bool = False,
    max_iter: int = 1000,
    tol: Scalar = 1e-4,
    validation_method: str = "cv",
    K: int = 5,
    initial_window: Optional[int] = None,
    step_size: Optional[int] = None,
    horizon: Optional[int] = None,
    max_window_size: Optional[int] = None,
) -> MCNNMResults:
    """
    Estimate the parameters of the MC-NNM (Matrix Completion with Nuclear Norm Minimization) model.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Optional[Array]): The unit-specific covariates matrix. Default is None.
        Z (Optional[Array]): The time-specific covariates matrix. Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
        Omega (Optional[Array]): The autocorrelation matrix. Default is None.
        lambda_L (Optional[Scalar]): The regularization parameter for L. If None, it will be selected via validation.
        lambda_H (Optional[Scalar]): The regularization parameter for H. If None, it will be selected via validation.
        n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
        n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
        return_tau (bool): Whether to return the estimated average treatment effect. Default is True.
        return_lambda (bool): Whether to return the selected regularization parameters. Default is True.
        return_completed_L (bool): Whether to return the estimated low-rank matrix L. Default is True.
        return_completed_Y (bool): Whether to return the completed outcome matrix. Default is True.
        return_fixed_effects (bool): Whether to return the estimated unit and time fixed effects. Default is False.
        return_covariate_coefficients (bool): Whether to return the estimated covariate coefficients. Default is False.
        max_iter (int): Maximum number of iterations for fitting. Default is 1000.
        tol (Scalar): Convergence tolerance for fitting. Default is 1e-4.
        validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
        K (int): Number of folds for cross-validation or time-based validation. Default is 5.
        initial_window (Optional[int]): Number of initial time periods to use for first training set in holdout
        validation. Only used when validation_method='holdout'. If None, defaults to 80% of total time periods.
        step_size (Optional[int]): Number of time periods to move forward for each split in holdout validation.
                                   Only used when validation_method='holdout'.
                                   If None, defaults to (T - initial_window) // K.
        horizon (Optional[int]): Number of future time periods to predict (forecast horizon) in holdout validation.
                                 Only used when validation_method='holdout'. If None, defaults to step_size.
        max_window_size (Optional[int]): Maximum size of the window to consider in holdout validation.
                                         Only used when validation_method='holdout'. If None, use all data.

    Returns:
        MCNNMResults: A named tuple containing the results of the MC-NNM estimation.
    """

    X, Z, V, Omega = check_inputs(Y, W, X, Z, V, Omega)
    X, Z, V, Omega = cast(Array, X), cast(Array, Z), cast(Array, V), cast(Array, Omega)
    N, T = Y.shape

    def select_lambda(_):
        lambda_grid = jnp.array(
            jnp.meshgrid(propose_lambda(None, n_lambda_L), propose_lambda(None, n_lambda_H))
        ).T.reshape(-1, 2)

        if validation_method == "cv":
            return cross_validate(
                Y, W, X, Z, V, Omega, lambda_grid, K=K, max_iter=max_iter // 10, tol=tol * 10
            )
        elif validation_method == "holdout":
            if T < 5:
                raise ValueError(
                    "The matrix does not have enough columns for time-based validation. "
                    "Please increase the number of time periods or use cross-validation"
                )

            defaults = generate_time_based_validate_defaults(Y, n_lambda_L, n_lambda_H)

            return time_based_validate(
                Y,
                W,
                X,
                Z,
                V,
                Omega,
                lambda_grid=defaults["lambda_grid"],
                max_iter=max_iter // 10,
                tol=tol * 10,
                initial_window=initial_window or defaults["initial_window"],
                step_size=step_size or defaults["step_size"],
                horizon=horizon or defaults["horizon"],
                K=K or defaults["K"],
                max_window_size=max_window_size,
                T=T,  # Pass T as a static argument
            )
        else:
            raise ValueError("Invalid validation_method. Choose 'cv' or 'holdout'.")

    def use_provided_lambda(lambda_L, lambda_H):
        def to_jax_array(x):
            if x is None:  # pragma: no cover
                return jnp.array(float("nan"))
            return jnp.array(x, dtype=float)

        return (to_jax_array(lambda_L), to_jax_array(lambda_H))

    lambda_L, lambda_H = jax.lax.cond(
        jnp.logical_or(lambda_L is None, lambda_H is None),
        lambda _: select_lambda(None),
        lambda _: use_provided_lambda(lambda_L, lambda_H),
        operand=None,
    )

    initial_params = initialize_params(Y, X, Z, V)
    L, H, gamma, delta, beta = fit(
        Y, W, X, Z, V, Omega, lambda_L, lambda_H, initial_params, max_iter, tol  # type: ignore
    )

    results = {}

    def compute_Y_completed():
        X_tilde = jnp.hstack((X, jnp.eye(N)))
        Z_tilde = jnp.hstack((Z, jnp.eye(T)))
        Y_completed = L + jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))
        Y_completed += jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)
        Y_completed = jax.lax.cond(
            V.shape[2] > 0, lambda y: y + jnp.sum(V * beta, axis=2), lambda y: y, Y_completed
        )
        return Y_completed

    def compute_tau():
        tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
        return jnp.array(tau, dtype=jnp.float32)

    results["tau"] = jax.lax.cond(
        return_tau, compute_tau, lambda: jnp.array(0.0, dtype=jnp.float32)
    )
    results["lambda_L"] = jax.lax.cond(
        return_lambda,
        lambda: jnp.array(lambda_L, dtype=jnp.float32),
        lambda: jnp.array(float("nan"), dtype=jnp.float32),
    )
    results["lambda_H"] = jax.lax.cond(
        return_lambda,
        lambda: jnp.array(lambda_H, dtype=jnp.float32),
        lambda: jnp.array(float("nan"), dtype=jnp.float32),
    )
    results["L"] = jax.lax.cond(return_completed_L, lambda: L, lambda: jnp.zeros_like(L))
    results["Y_completed"] = jax.lax.cond(
        return_completed_Y, compute_Y_completed, lambda: jnp.zeros_like(Y)
    )
    results["gamma"] = jax.lax.cond(
        return_fixed_effects, lambda: gamma, lambda: jnp.zeros_like(gamma)
    )
    results["delta"] = jax.lax.cond(
        return_fixed_effects, lambda: delta, lambda: jnp.zeros_like(delta)
    )
    results["beta"] = jax.lax.cond(
        return_covariate_coefficients, lambda: beta, lambda: jnp.zeros_like(beta)
    )
    results["H"] = jax.lax.cond(return_covariate_coefficients, lambda: H, lambda: jnp.zeros_like(H))

    return MCNNMResults(**results)


def complete_matrix(
    Y: Array,
    W: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
    lambda_L: Optional[Scalar] = None,
    lambda_H: Optional[Scalar] = None,
    n_lambda_L: int = 10,
    n_lambda_H: int = 10,
    max_iter: int = 1000,
    tol: Scalar = 1e-4,
    validation_method: str = "cv",
    K: int = 5,
    initial_window: Optional[int] = None,
    step_size: Optional[int] = None,
    horizon: Optional[int] = None,
    max_window_size: Optional[int] = None,
) -> MCNNMResults:
    """
    Complete the matrix Y using the MC-NNM model and return the optimal regularization parameters.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Optional[Array]): The unit-specific covariates matrix. Default is None.
        Z (Optional[Array]): The time-specific covariates matrix. Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
        Omega (Optional[Array]): The autocorrelation matrix. Default is None.
        lambda_L (Optional[Scalar]): The regularization parameter for L. If None, it will be selected via validation.
        lambda_H (Optional[Scalar]): The regularization parameter for H. If None, it will be selected via validation.
        n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
        n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
        max_iter (int): Maximum number of iterations for fitting. Default is 1000.
        tol (Scalar): Convergence tolerance for fitting. Default is 1e-4.
        validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
        K (int): Number of folds for cross-validation or time-based validation. Default is 5.
        initial_window (Optional[int]): Number of initial time periods to use for first train set in holdout validation.
                                        Only used when validation_method='holdout'. If None, defaults to 80% of T.
        step_size (Optional[int]): Number of time periods to move forward for each split in holdout validation.
                                   Only used when validation_method='holdout'.
                                   If None, defaults to (T - initial_window) // K.
        horizon (Optional[int]): Number of future time periods to predict (forecast horizon) in holdout validation.
                                 Only used when validation_method='holdout'. If None, defaults to step_size.
        max_window_size (Optional[int]): Maximum size of the window to consider in holdout validation.
                                         Only used when validation_method='holdout'. If None, use all data.

    Returns:
        Tuple[Array, Scalar, Scalar]: A tuple containing:
            - The completed outcome matrix
            - The optimal lambda_L value
            - The optimal lambda_H value
    """
    results = estimate(
        Y,
        W,
        X,
        Z,
        V,
        Omega,
        lambda_L,
        lambda_H,
        n_lambda_L,
        n_lambda_H,
        return_tau=False,
        return_lambda=True,
        return_completed_L=False,
        return_completed_Y=True,
        return_fixed_effects=False,
        return_covariate_coefficients=False,
        max_iter=max_iter,
        tol=tol,
        validation_method=validation_method,
        K=K,
        initial_window=initial_window,
        step_size=step_size,
        horizon=horizon,
        max_window_size=max_window_size,
    )

    return MCNNMResults(
        Y_completed=results.Y_completed, lambda_L=results.lambda_L, lambda_H=results.lambda_H
    )
