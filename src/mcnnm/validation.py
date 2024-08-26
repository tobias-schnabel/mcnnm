from typing import Tuple, Optional

import jax
import jax.numpy as jnp

from .core import fit, initialize_matrices, initialize_fixed_effects_and_H
from .utils import generate_lambda_grid, extract_shortest_path
from .types import Array


def cross_validate(
    Y: Array,
    X: Array,
    Z: Array,
    V: Array,
    W: Array,
    Omega_inv: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
    num_lam: int,
    max_iter: Optional[int] = 1000,
    tol: Optional[float] = 1e-5,
    cv_ratio: Optional[float] = 0.8,
    K: Optional[int] = 5,
) -> Tuple[Array, Array]:
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
        Omega_inv (Array): The inverse of the covariance matrix of shape (T, T).
        use_unit_fe (bool): Whether to include unit fixed effects in the model.
        use_time_fe (bool): Whether to include time fixed effects in the model.
        num_lam (int): The number of lambda values to include in the lambda grid.
        max_iter (int, optional): The maximum number of iterations for model fitting. Default is 1000.
        tol (float, optional): The tolerance for convergence in model fitting. Default is 1e-5.
        cv_ratio (float, optional): The ratio of data to use for training in each fold. Default is 0.8.
        K (int, optional): The number of folds for cross-validation. Default is 5.

    Returns:
        Tuple[Array, Array]: A tuple containing the best lambda_L and lambda_H values.

    Raises:
        ValueError: If the input arrays have inconsistent shapes.

    Example:
        >>> Y = jnp.random.normal(size=(100, 50))
        >>> X = jnp.random.normal(size=(100, 10))
        >>> Z = jnp.random.normal(size=(50, 5))
        >>> V = jnp.random.normal(size=(100, 50, 3))
        >>> W = jnp.random.choice([0, 1], size=(100, 50))
        >>> Omega_inv = jnp.eye(50)
        >>> best_lambda_L, best_lambda_H = cross_validate(Y, X, Z, V, W, Omega_inv, True, True, 10)

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

    full_lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, num_lam)
    shortest_path = extract_shortest_path(full_lambda_grid)
    lambda_grid = jnp.vstack((shortest_path, jnp.array([[0.0, 0.0]])))

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
    ):  # fold_config
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
    min_index = jnp.argmin(jnp.stack(fold_rmses))

    best_lambda_L, best_lambda_H = lambda_grid[min_index]

    return best_lambda_L, best_lambda_H
