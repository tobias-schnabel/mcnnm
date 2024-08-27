from typing import Tuple, Optional

import jax
import jax.numpy as jnp

from .core import fit, initialize_matrices, initialize_fixed_effects_and_H
from .utils import generate_lambda_grid
from .types import Array


def cross_validate(
    Y: Array,
    X: Array,
    Z: Array,
    V: Array,
    W: Array,
    Omega_inv: Optional[Array],
    use_unit_fe: bool,
    use_time_fe: bool,
    num_lam: int,
    max_iter: Optional[int] = 1000,
    tol: Optional[float] = 1e-5,
    cv_ratio: Optional[float] = 0.8,
    K: Optional[int] = 5,
) -> Tuple[Array, Array, Array, Array]:
    """
    Perform K-fold cross-validation to select the best regularization parameters for the model.

    This function splits the data into K folds, trains the model on K-1 folds, and evaluates it
    on the remaining fold. The process is repeated for each fold and for different combinations
    of regularization parameters (lambda_L and lambda_H) specified in the lambda grid. The best
    lambda values are selected based on the minimum average root mean squared error (RMSE) across
    all folds.

    Args:
        Y (Array): The target variable matrix of shape (N, T).
        X (Array): The feature matrix for unit-specific covariates of shape (N, P).
        Z (Array): The feature matrix for time-specific covariates of shape (T, Q).
        V (Array): The feature matrix for unit-time covariates of shape (N, T, R).
        W (Array): The binary matrix indicating the presence of observations of shape (N, T).
        Omega_inv (Array, optional): The inverse of the covariance matrix of shape (T, T). Default is None.
        use_unit_fe (bool): Whether to include unit fixed effects in the model.
        use_time_fe (bool): Whether to include time fixed effects in the model.
        num_lam (int): The number of lambda values to include in the lambda grid.
        max_iter (int, optional): The maximum number of iterations for model fitting. Default is 1000.
        tol (float, optional): The tolerance for convergence in model fitting. Default is 1e-5.
        cv_ratio (float, optional): The ratio of data to use for training in each fold. Default is 0.8.
        K (int, optional): The number of folds for cross-validation. Default is 5.

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing the best lambda_L and lambda_H values along with the
        maximum lambda_L and lambda_H values.

    Raises:
        ValueError: If the input arrays have inconsistent shapes.

    Note:
        - The function assumes that the input arrays have consistent shapes and are of type `jax.numpy.ndarray`.
        - The function uses `jax.vmap` and `jax.lax.scan` for parallelization and efficient computation.
        - The function initializes the model parameters using the `initialize_matrices` and
        `initialize_fixed_effects_and_H` functions.
        - The function generates a lambda grid using the `generate_lambda_grid` and `extract_shortest_path` functions.
        - The function computes the RMSE for each fold and lambda combination using the `fit` function.
        - The function selects the best lambda values based on the minimum average RMSE across all folds.
    """
    N, T = Y.shape

    def create_folds(key):
        def create_fold_mask(key):
            return jax.random.bernoulli(key, cv_ratio, shape=(N, T))

        keys = jax.random.split(key, K)
        fold_masks = jax.vmap(create_fold_mask)(keys)
        return fold_masks * W

    fold_masks = create_folds(jax.random.PRNGKey(2024))
    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    def initialize_fold(fold_mask):
        Y_train = Y * fold_mask
        W_train = W * fold_mask

        (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
        ) = initialize_fixed_effects_and_H(
            Y_train, L, X_tilde, Z_tilde, V, W_train, use_unit_fe, use_time_fe, verbose=False
        )
        return (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
            fold_mask,
        )

    fold_configs = jax.vmap(initialize_fold)(fold_masks)

    max_lambda_L = jnp.max(fold_configs[7])
    max_lambda_H = jnp.max(fold_configs[8])

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, num_lam)

    def fold_loss(
        gamma_init,
        delta_init,
        beta_init,
        H_tilde_init,
        T_mat_init,
        in_prod_T_init,
        in_prod_init,
        lambda_L_max,
        lambda_H_max,
        fold_mask,
    ):
        Y_train = Y * fold_mask
        W_train = W * fold_mask

        def compute_loss(L, lambda_L_H):
            lambda_L, lambda_H = lambda_L_H
            _, L_new, _, _, _, _, loss = fit(
                Y=Y_train,
                X_tilde=X_tilde,
                Z_tilde=Z_tilde,
                V=V,
                H_tilde=H_tilde_init,
                T_mat=T_mat_init,
                in_prod=in_prod_init,
                in_prod_T=in_prod_T_init,
                W=W_train,
                L=L,
                gamma=gamma_init,
                delta=delta_init,
                beta=beta_init,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                Omega_inv=Omega_inv,
                niter=max_iter,
                rel_tol=tol,
            )

            def compute_rmse(loss):
                return jnp.sqrt(loss)

            def return_inf(loss):
                return jnp.inf

            fold_rmse = jax.lax.cond(loss >= 0, compute_rmse, return_inf, loss)

            return L_new, fold_rmse

        _, fold_rmses = jax.lax.scan(compute_loss, L, lambda_grid)
        return fold_rmses

    fold_rmses = jax.vmap(fold_loss, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(*fold_configs)
    mean_rmses = jnp.mean(fold_rmses, axis=0)
    min_lambda_pair_index = jnp.argmin(mean_rmses)

    best_lambda_L, best_lambda_H = lambda_grid[min_lambda_pair_index]

    return best_lambda_L, best_lambda_H, max_lambda_L, max_lambda_H


def holdout_validate(
    Y: Array,
    X: Array,
    Z: Array,
    V: Array,
    W: Array,
    Omega_inv: Optional[Array],
    use_unit_fe: bool,
    use_time_fe: bool,
    num_lam: int,
    initial_window: int,
    step_size: int,
    horizon: int,
    K: int,
    max_window_size: Optional[int] = None,
    max_iter: Optional[int] = 1000,
    tol: Optional[float] = 1e-5,
) -> Tuple[Array, Array, Array, Array]:
    """
    Perform holdout validation to select the optimal regularization parameters for the MC-NNM model.

    This function splits the data into K holdout folds along the time dimension, initializes the model
    configurations for each fold based on the observed data within the specified time window, and computes
    the holdout loss and RMSE for each fold and lambda pair. The lambda pair that yields the lowest average
    RMSE across all folds is selected as the optimal regularization parameters.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).
        W (Array): The binary matrix indicating observed (0) and missing (1) entries in Y, shape (N, T).
        Omega_inv (Array, optional): The autocorrelation matrix of shape (T, T). Defaults to None.
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.
        num_lam (int): The number of lambda values to generate in the grid.
        initial_window (int): The size of the initial time window for holdout validation. It determines the
            number of time steps used to initialize the model configurations for each holdout fold.
        step_size (int): The step size for moving the time window in each holdout fold. It determines the
            number of time steps to shift the window for each subsequent fold.
        horizon (int): The size of the holdout horizon (number of time steps to predict). It determines the
            number of time steps used for evaluating the model's performance in each holdout fold.
        K (int): The number of holdout folds.
        max_window_size (int, optional): The maximum size of the time window. If specified, it limits the
            size of the time window used for initializing the model configurations in each holdout fold.
            The window size will not exceed `max_window_size` even if `initial_window` + `horizon` is larger.
            Defaults to None, meaning no limit on the window size.
        max_iter (int, optional): Maximum number of iterations for fitting the model. Defaults to 1000.
        tol (float, optional): Convergence tolerance for fitting the model. Defaults to 1e-5.

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing the optimal lambda_L and lambda_H values along with
        the maximum lambda_L and lambda_H values.

    Steps:
        1. Create K holdout masks, each representing a specific time window in the data. The time windows are
           determined by `initial_window`, `step_size`, and `horizon`.
        2. Initialize the model configurations (gamma, delta, beta, H_tilde, etc.) for each holdout fold
           based on the observed data within the corresponding time window. The window size is limited by
           `max_window_size` if specified.
        3. Compute the maximum lambda_L and lambda_H values across all holdout folds.
        4. Generate a grid of lambda pairs using the maximum lambda_L and lambda_H values.
        5. For each holdout fold and lambda pair, compute the holdout loss and RMSE by fitting the model
           using the observed data within the time window and evaluating on the entire data.
        6. Compute the average RMSE across all holdout folds for each lambda pair.
        7. Select the lambda pair that yields the lowest average RMSE as the optimal regularization parameters.

    Note:
        - The binary matrix W indicates observed (0) and missing (1) entries in Y. The missing entries (1)
          are relevant for the loss computation.
        - The function uses JAX's vmap and scan operations to efficiently compute the holdout losses and
          RMSEs for multiple holdout folds and lambda pairs in parallel.
    """
    N, T = Y.shape

    def create_holdout_masks(W, initial_window, step_size, horizon, K, max_window_size):
        masks = []
        start_index = initial_window
        for _ in range(K):
            end_index = min(start_index + horizon, T)
            if max_window_size is not None:
                start_index = max(end_index - max_window_size, 0)
            mask = jnp.zeros((N, T), dtype=bool)
            mask = mask.at[:, :end_index].set(True)
            n_train = jnp.sum(W * mask)
            if n_train > 0:
                masks.append(mask)
                start_index += step_size
        return jnp.array(masks)

    holdout_masks = create_holdout_masks(W, initial_window, step_size, horizon, K, max_window_size)
    if holdout_masks.shape[0] < K:
        print("Warning: Not enough data for holdout validation. Using fewer folds.")
    if holdout_masks.shape[0] == 0:
        print("Error: No data available for holdout validation. Exiting.")
        return (jnp.array(jnp.nan), jnp.array(jnp.nan), jnp.array(jnp.nan), jnp.array(jnp.nan))
    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    def initialize_holdout(holdout_mask):
        Y_train = Y * holdout_mask
        W_train = W * holdout_mask

        (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
        ) = initialize_fixed_effects_and_H(
            Y_train, L, X_tilde, Z_tilde, V, W_train, use_unit_fe, use_time_fe, verbose=False
        )
        return (
            gamma_init,
            delta_init,
            beta_init,
            H_tilde_init,
            T_mat_init,
            in_prod_T_init,
            in_prod_init,
            lambda_L_max,
            lambda_H_max,
            holdout_mask,
        )

    holdout_configs = jax.vmap(initialize_holdout)(holdout_masks)

    max_lambda_L = jnp.max(holdout_configs[7])
    max_lambda_H = jnp.max(holdout_configs[8])

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, num_lam)

    def holdout_loss(
        gamma_init,
        delta_init,
        beta_init,
        H_tilde_init,
        T_mat_init,
        in_prod_T_init,
        in_prod_init,
        lambda_L_max,
        lambda_H_max,
        holdout_mask,
    ):
        Y_obs = Y * holdout_mask
        W_obs = W * holdout_mask

        def compute_loss(L, lambda_L_H):
            lambda_L, lambda_H = lambda_L_H
            _, L_new, _, _, _, _, loss = fit(
                Y=Y_obs,
                X_tilde=X_tilde,
                Z_tilde=Z_tilde,
                V=V,
                H_tilde=H_tilde_init,
                T_mat=T_mat_init,
                in_prod=in_prod_init,
                in_prod_T=in_prod_T_init,
                W=W_obs,
                L=L,
                gamma=gamma_init,
                delta=delta_init,
                beta=beta_init,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                Omega_inv=Omega_inv,
                niter=max_iter,
                rel_tol=tol,
            )

            def compute_rmse(loss):
                return jnp.sqrt(loss)

            def return_inf(loss):
                return jnp.inf

            holdout_rmse = jax.lax.cond(loss >= 0, compute_rmse, return_inf, loss)

            return L_new, holdout_rmse

        _, holdout_rmses = jax.lax.scan(compute_loss, L, lambda_grid)
        return holdout_rmses

    holdout_rmses = jax.vmap(holdout_loss, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(*holdout_configs)
    mean_rmses = jnp.mean(holdout_rmses, axis=0)
    min_index = jnp.argmin(mean_rmses)

    best_lambda_L, best_lambda_H = lambda_grid[min_index]

    return best_lambda_L, best_lambda_H, max_lambda_L, max_lambda_H
