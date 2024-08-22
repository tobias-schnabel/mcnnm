from .types import Array, Scalar
from typing import Tuple, Optional
import jax.numpy as jnp
import jax
from core import initialize_coefficients, fit
from functools import partial


def generate_time_based_validate_defaults(Y: Array, n_lambda_L: int = 10, n_lambda_H: int = 10):
    N, T = Y.shape
    T = int(T)

    initial_window = int(0.8 * T)
    K = 5
    step_size = max(1, (T - initial_window) // K)
    horizon = step_size

    lambda_grid = jnp.array(
        jnp.meshgrid(jnp.logspace(-3, 0, n_lambda_L), jnp.logspace(-3, 0, n_lambda_H))
    ).T.reshape(-1, 2)

    max_iter = 1000
    tol = 1e-4

    return {
        "initial_window": initial_window,
        "step_size": step_size,
        "horizon": horizon,
        "K": K,
        "lambda_grid": lambda_grid,
        "max_iter": max_iter,
        "tol": tol,
    }


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
    use_unit_fe: bool,
    use_time_fe: bool,
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
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.

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

    initial_params = initialize_coefficients(Y_train, X_train, Z, V_train, use_unit_fe, use_time_fe)

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
        use_unit_fe,
        use_time_fe,
    )

    Y_pred = L[test_idx]
    if use_unit_fe:
        Y_pred += jnp.outer(gamma[test_idx], jnp.ones(Z.shape[0]))
    if use_time_fe:
        Y_pred += jnp.outer(jnp.ones(test_idx.shape[0]), delta)

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
    use_unit_fe: bool,
    use_time_fe: bool,
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
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int): Maximum number of iterations for fitting the model in each fold.
        tol (float): Convergence tolerance for fitting the model in each fold.
        K (int, optional): Number of folds for cross-validation. Default is 5.

    Returns:
        Tuple[Scalar, Scalar]: A tuple containing the optimal lambda_L and lambda_H values.
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

            initial_params = initialize_coefficients(
                Y_train, X_train, Z, V_train, use_unit_fe, use_time_fe
            )

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
                use_unit_fe,
                use_time_fe,
            )

            Y_pred = L
            if use_unit_fe:
                Y_pred += jnp.outer(gamma, jnp.ones(Z.shape[0]))
            if use_time_fe:
                Y_pred += jnp.outer(jnp.ones(N), delta)

            Y_pred = jax.lax.cond(
                V_test.shape[2] > 0,
                lambda Y_pred: Y_pred + jnp.sum(V_test * beta, axis=2),
                lambda Y_pred: Y_pred,
                Y_pred,
            )

            O_test = W_test == 0
            loss = jnp.sum((Y_test - Y_pred) ** 2 * O_test) / (jnp.sum(O_test) + 1e-10)
            return loss

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


@partial(jax.jit, static_argnums=(6, 7, 13, 14, 15, 16))
def time_based_validate(
    Y: Array,
    W: Array,
    X: Array,
    Z: Array,
    V: Array,
    Omega: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
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
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int): Maximum number of iterations for fitting.
        tol (Scalar): Convergence tolerance for fitting.
        initial_window (int): Number of initial time periods to use for first training set.
        step_size (int): Number of time periods to move forward for each split.
        horizon (int): Number of future time periods to predict (forecast horizon).
        K (int): Number of folds to use in the time-based validation.
        T (int): Total number of time periods.
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

        initial_params = initialize_coefficients(
            Y_train, X, Z_train, V_train, use_unit_fe, use_time_fe
        )
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
            use_unit_fe,
            use_time_fe,
        )

        Y_pred = L
        if use_unit_fe:
            Y_pred += jnp.outer(gamma, jnp.ones(T))
        if use_time_fe:
            Y_pred += jnp.outer(jnp.ones(N), delta)

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
