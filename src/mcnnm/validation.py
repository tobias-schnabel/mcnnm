from typing import Tuple, Optional

import jax
import jax.debug as jdb
import jax.numpy as jnp

from .core import fit
from .types import Array, Scalar


def cross_validate(
    Y: Array,
    W: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    Omega_inv: Optional[Array],
    L: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    H_tilde: Array,
    T_mat: Array,
    in_prod: Array,
    in_prod_T: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
    lambda_grid: Array,
    max_iter: Optional[int] = 1000,
    tol: Optional[float] = 1e-5,
    K: Optional[int] = 5,
) -> Tuple[Scalar, Scalar]:
    """
    Perform K-fold cross-validation to select optimal regularization parameters for the MC-NNM model.

    This function splits the data into K folds along the unit dimension, trains the model on K-1 folds,
    and evaluates it on the remaining fold. This process is repeated for all folds and all lambda pairs
    in the lambda_grid. The lambda pair that yields the lowest average loss across all folds is selected.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).
        Omega_inv (Array, optional): The autocorrelation matrix of shape (T, T). Defaults to an Identity matrix.
        L (Array): The initialized low-rank matrix of shape (N, T).
        gamma (Array): The initialized unit fixed effects vector of shape (N,).
        delta (Array): The initialized time fixed effects vector of shape (T,).
        beta (Array): The initialized unit-time specific covariate coefficients vector of shape (J,).
        H_tilde (Array): The initialized covariate coefficients matrix of shape (P+N, Q+T).
        T_mat (Array): The precomputed matrix T of shape (N * T, (P+N) * (Q+T)).
        in_prod (Array): The inner product vector of shape (N * T,).
        in_prod_T (Array): The inner product vector of T of shape ((P+N) * (Q+T),).
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.
        lambda_grid (Array): Grid of (lambda_L, lambda_H) pairs to search over.
        max_iter (int, optional): Maximum number of iterations for fitting the model in each fold. Default is 1000.
        tol (float, optional): Convergence tolerance for fitting the model in each fold. Default is 1e-5.
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
            W_train = jnp.where(mask[:, None], jnp.zeros_like(W), W)
            X_tilde_train = jnp.where(mask[:, None], jnp.zeros_like(X_tilde), X_tilde)
            V_train = jnp.where(mask[:, None, None], jnp.zeros_like(V), V)

            _, _, _, _, _, _, loss = fit(
                Y=Y_train,
                X_tilde=X_tilde_train,
                Z_tilde=Z_tilde,
                V=V_train,
                H_tilde=H_tilde,
                T_mat=T_mat,
                in_prod=in_prod,
                in_prod_T=in_prod_T,
                W=W_train,
                L=L,
                gamma=gamma,
                delta=delta,
                beta=beta,
                lambda_L=lambda_L,
                lambda_H=lambda_H,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                Omega_inv=Omega_inv,
                niter=max_iter,
                rel_tol=tol,
            )

            return loss

        def fold_loss_wrapper(i, acc):
            return acc + fold_loss(i)

        losses = jax.lax.fori_loop(0, K, fold_loss_wrapper, 0.0)

        return jnp.mean(losses)

    losses = jax.vmap(loss_fn)(lambda_grid)

    def select_best_lambda(_):
        best_idx = jnp.argmin(losses)
        return lambda_grid[best_idx]

    def use_default_lambda(_):
        mid_idx = len(lambda_grid) // 2
        jdb.print("No finite losses found. Using default lambda values.")
        return lambda_grid[mid_idx]

    best_lambda_L_H = jax.lax.cond(
        jnp.any(jnp.isfinite(losses)), select_best_lambda, use_default_lambda, operand=None
    )

    return best_lambda_L_H[0], best_lambda_L_H[1]
