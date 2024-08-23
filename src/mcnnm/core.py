from typing import Tuple, Optional

import jax.numpy as jnp
from jax import jit, lax
import jax.debug as jdb

from .core_utils import mask_observed
from .types import Array, Scalar


def initialize_coefficients(
    Y: Array, X: Array, Z: Array, V: Array
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Initialize covariate and fixed effects coefficients  for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).

    Returns:
        Tuple[Array, Array, Array, Array, Array]: A tuple containing initial values for L,
        H, gamma, delta, and beta.
    """
    N, T = Y.shape
    L = jnp.zeros_like(Y)
    gamma = jnp.zeros(N)  # unit FE coefficients
    delta = jnp.zeros(T)  # time FE coefficients

    H = jnp.zeros((X.shape[1] + N, Z.shape[1] + T))  # X and Z-covariate coefficients

    beta_shape = max(V.shape[2], 1)
    beta = jnp.zeros((beta_shape,))  # unit-time covariate coefficients

    return L, H, gamma, delta, beta


def initialize_matrices(
    Y: Array,
    X: Optional[Array],
    Z: Optional[Array],
    V: Optional[Array],
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Initialize parameters (L, H, u, v, beta) based on the input data and model configuration.
    Handle cases where covariates are not present by initializing them as zero arrays.
    Apply normalization to X and Z covariates using the normalize function.
    """
    N, T = Y.shape
    L = jnp.zeros_like(Y)
    # Initialize covariates as 0 if not used
    if X is None:
        X = jnp.zeros((N, 1))
    if Z is None:
        Z = jnp.zeros((T, 1))
    if V is None:
        V = jnp.zeros((N, T, 1))

    # Initialize unit and time fixed effects
    unit_fe = jnp.where(use_unit_fe, jnp.ones(N), jnp.zeros(N))
    time_fe = jnp.where(use_time_fe, jnp.ones(T), jnp.zeros(T))

    return L, X, Z, V, unit_fe, time_fe


@jit
def compute_svd(M: Array) -> Tuple[Array, Array, Array]:
    """
    Compute the Singular Value Decomposition (SVD) of the input matrix M.
    Return the left singular vectors (U), right singular vectors (V), and singular values (Sigma).
    """
    U, Sigma, Vt = jnp.linalg.svd(M, full_matrices=False)
    V = Vt.T
    return U, V, Sigma


@jit
def update_unit_fe(
    Y: Array, X: Array, Z: Array, H: Array, W: Array, L: Array, time_fe: Array, use_unit_fe: bool
) -> Array:
    """
    Update the unit fixed effects (unit_fe) in the coordinate descent algorithm when covariates are available.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P). TODO: padded matrices
        Z (Array): The time-specific covariates matrix of shape (T, Q). TODO: padded matrices
        H (Array): The covariate coefficients matrix of shape (P, Q). TODO: padded matrices
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        time_fe (Array): The time fixed effects vector of shape (T,).
        use_unit_fe (bool): Whether to estimate unit fixed effects.

    Returns:
        Array: The updated unit fixed effects vector of shape (N,) if use_unit_fe is True, else a zero vector.
    """
    T_ = jnp.einsum("np,pq,tq->nt", X, H, Z)
    b_ = T_ + L + time_fe - Y
    b_mask_ = b_ * W
    l = jnp.sum(W, axis=1)
    res = jnp.where(l > 0, -jnp.sum(b_mask_, axis=1) / l, 0.0)
    return jnp.where(use_unit_fe, res, jnp.zeros_like(res))


@jit
def update_time_fe(
    Y: Array, X: Array, Z: Array, H: Array, W: Array, L: Array, unit_fe: Array, use_time_fe: bool
) -> Array:
    """
    Update the time fixed effects (time_fe) in the coordinate descent algorithm when covariates are available.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P). TODO: padded matrices
        Z (Array): The time-specific covariates matrix of shape (T, Q). TODO: padded matrices
        H (Array): The covariate coefficients matrix of shape (P, Q). TODO: padded matrices
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        unit_fe (Array): The unit fixed effects vector of shape (N,).
        use_time_fe (bool): Whether to estimate time fixed effects.

    Returns:
        Array: The updated time fixed effects vector of shape (T,) if use_time_fe is True, else a zero vector.
    """
    T_ = jnp.einsum("np,pq,tq->nt", X, H, Z)
    b_ = T_ + L + jnp.expand_dims(unit_fe, axis=1) - Y
    b_mask_ = b_ * W
    l = jnp.sum(W, axis=0)
    res = jnp.where(l > 0, -jnp.sum(b_mask_, axis=0) / l, 0.0)
    return jnp.where(use_time_fe, res, jnp.zeros_like(res))


@jit
def compute_decomposition(
    L: Array,
    X: Array,
    Z: Array,
    V: Array,
    H: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Array:
    N, T = L.shape
    P = X.shape[1]
    Q = Z.shape[1]

    decomposition = L

    unit_fe_term = jnp.outer(gamma, jnp.ones(T))
    decomposition += jnp.where(use_unit_fe, unit_fe_term, jnp.zeros_like(unit_fe_term))

    time_fe_term = jnp.outer(jnp.ones(N), delta)
    decomposition += jnp.where(use_time_fe, time_fe_term, jnp.zeros_like(time_fe_term))

    decomposition += (
        X @ H[:P, :Q] @ Z.T + X @ H[:P, Q:] + H[P:, :Q] @ Z.T + jnp.einsum("ntj,j->nt", V, beta)
    )

    # XH_term = jnp.dot(X, H[:P, :Q]) TODO: cleanup
    # XH_Z_term = jnp.where(Q > 0, jnp.dot(XH_term, Z.T), jnp.zeros((N, T)))
    # decomposition += XH_Z_term
    #
    # XH_extra_term = jnp.dot(X, H[:P, Q:])
    # decomposition += XH_extra_term
    #
    # H_Z_term = jnp.where(
    #     P + N <= H.shape[0] and Q > 0, jnp.dot(H[P : P + N, :Q], Z.T), jnp.zeros((N, T))
    # )
    # decomposition += H_Z_term
    #
    # V_beta_term = jnp.einsum("ntj,j->nt", V, beta)
    # decomposition += V_beta_term

    return decomposition


def compute_objective_value(
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
    verbose: bool = False,
) -> Scalar:
    r"""
    Compute the objective value for the MC-NNM model with covariates, fixed effects,
    and time series correlation.

    The objective function is defined as:

    .. math::

        \frac{1}{|\Omega|} \sum_{(i,t) \in \Omega} \sum_{(i,s) \in \Omega}
        (Y_{it} - \hat{Y}_{it}) [\Omega^{-1}]_{ts} (Y_{is} - \hat{Y}_{is})
        + \lambda_L \|L^*\|_* + \lambda_H \|H^*\|_1

    where:
    - :math:`Y_{it}` is the observed outcome for unit :math:`i` at time :math:`t`
    - :math:`\hat{Y}_{it}` is the estimated outcome for unit :math:`i` at time
      :math:`t`, given by:

      .. math::

          \hat{Y}_{it} = L^*_{it} + \sum_{p=1}^P X_{ip} H^*_{pq} Z_{tq}
          + \sum_{q=1}^Q H^*_{(P+i)q} Z_{tq} + \sum_{p=1}^P X_{ip} H^*_{p(Q+t)}
          + \Gamma^*_i + \Delta^*_t + \sum_{j=1}^J V_{itj} \beta^*_j

    - :math:`\Omega` is the set of observed entries in the outcome matrix
    - :math:`\Omega^{-1}` is the inverse of the omega matrix, capturing the time
      series correlation
    - :math:`L^*` is the low-rank matrix of shape (N, T)
    - :math:`X` is the unit-specific covariates matrix of shape (N, P)
    - :math:`Z` is the time-specific covariates matrix of shape (T, Q)
    - :math:`V` is the unit-time-specific covariates tensor of shape (N, T, J)
    - :math:`H^*` is the covariate coefficients matrix of shape (P + N, Q + T)
    - :math:`\Gamma^*` is the unit fixed effects vector of shape (N,)
    - :math:`\Delta^*` is the time fixed effects vector of shape (T,)
    - :math:`\beta^*` is the unit-time-specific covariate coefficients vector
      of shape (J,)
    - :math:`\lambda_L` is the regularization parameter for the nuclear norm of
      :math:`L^*`
    - :math:`\lambda_H` is the regularization parameter for the element-wise L1 norm
      of :math:`H^*`

    The objective function consists of three terms:
    1. The weighted mean squared error term, which measures the discrepancy between
       the observed outcomes and the estimated outcomes, weighted by the inverse of
       the omega matrix to account for time series correlation.
    2. The nuclear norm regularization term for the low-rank matrix :math:`L^*`,
       which encourages low-rankness.
    3. The element-wise L1 norm regularization term for the covariate coefficients
       matrix :math:`H^*`, which encourages sparsity.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix :math:`X` of shape (N, P).
        Z (Array): The time-specific covariates matrix :math:`Z` of shape (T, Q).
        V (Array): The unit-time-specific covariates tensor :math:`V` of shape
            (N, T, J).
        H (Array): The covariate coefficients matrix :math:`H^*` of shape
            (P + N, Q + T).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix :math:`L^*` of shape (N, T).
        gamma (Array): The unit fixed effects vector :math:`\Gamma^*` of shape (N,).
        delta (Array): The time fixed effects vector :math:`\Delta^*` of shape (T,).
        beta (Array): The unit-time-specific covariate coefficients vector
            :math:`\beta^*` of shape (J,).
        sum_sing_vals (float): The sum of singular values of L.
        lambda_L (float): The regularization parameter for the nuclear norm of L.
        lambda_H (float): The regularization parameter for the element-wise L1 norm
            of H.
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.
        inv_omega (Optional[Array]): The inverse of the omega matrix of shape (T, T).
            If None, the identity matrix is used.

    Returns:
        Scalar: The computed objective value.
    """

    train_size = jnp.sum(W)
    norm_H = jnp.sum(jnp.abs(H))

    est_mat = compute_decomposition(L, X, Z, V, H, gamma, delta, beta, use_unit_fe, use_time_fe)
    err_mat = est_mat - Y

    if inv_omega is None:
        inv_omega = jnp.eye(Y.shape[1])

    weighted_err_mat = jnp.einsum(
        "ij,ntj,nsj->nts", inv_omega, err_mat[:, None, :], err_mat[:, :, None]
    )
    masked_weighted_err_mat = weighted_err_mat * W[:, None, :]
    obj_val = (
        (1 / train_size) * jnp.sum(masked_weighted_err_mat)
        + lambda_L * sum_sing_vals
        + lambda_H * norm_H
    )

    lax.cond(
        verbose, lambda _: jdb.print("Objective value: {ov}", ov=obj_val), lambda _: None, None
    )
    return obj_val


@jit
def initialize_fixed_effects_and_H(
    Y: Array,
    X: Array,
    Z: Array,
    V: Array,
    W: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
    niter: int = 1000,
    rel_tol: float = 1e-5,
) -> Tuple[Array, Array, Scalar, Scalar, Array, Array]:
    """
    Find the optimal unit_fe and time_fe assuming that L and H are zero. This is helpful for warm-starting
    the values of lambda_L and lambda_H. This function also outputs the smallest values of
    lambda_L and lambda_H which cause L and H to be zero.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        use_unit_fe (bool): Whether to include unit fixed effects in the model.
        use_time_fe (bool): Whether to include time fixed effects in the model.
        niter (int): The maximum number of iterations for the coordinate descent algorithm.
        rel_tol (float): The relative tolerance for convergence.

    Returns:
        Tuple[Array, Array, Scalar, Scalar, Array, Array]: A tuple containing:
            - unit_fe: The estimated unit fixed effects vector of shape (N,).
            - time_fe: The estimated time fixed effects vector of shape (T,).
            - lambda_L_max: The smallest value of lambda_L that causes L to be zero.
            - lambda_H_max: The smallest value of lambda_H that causes H to be zero.
            - T_mat: The T matrix used for computing lambda_H_max.
            - in_prod_T: The inner product of T_mat with itself.
    """
    L, X_tilde, Z_tilde, V, unit_fe, time_fe = initialize_matrices(  # TODO: check that correct
        Y, X, Z, V, use_unit_fe, use_time_fe
    )
    _, H, _, _, beta = initialize_coefficients(Y, X_tilde, Z_tilde, V)

    H_rows, H_cols = X_tilde.shape[1], Z_tilde.shape[1]

    obj_val = jnp.inf
    new_obj_val = 0.0

    def cond_fun(carry):
        iter_, obj_val, new_obj_val, *_ = carry
        rel_error = jnp.abs(new_obj_val - obj_val) / obj_val
        return jnp.logical_and(iter_ < niter, rel_error >= rel_tol)

    def body_fun(carry):
        iter_, obj_val, _, unit_fe, time_fe, L, H = carry

        unit_fe = update_unit_fe(Y, X_tilde, Z_tilde, H, W, L, time_fe, use_unit_fe)
        time_fe = update_time_fe(Y, X_tilde, Z_tilde, H, W, L, unit_fe, use_time_fe)

        new_obj_val = compute_objective_value(
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

        return iter_ + 1, new_obj_val, new_obj_val, unit_fe, time_fe, L, H

    init_carry = (0, obj_val, new_obj_val, unit_fe, time_fe, L, H)
    _, _, _, unit_fe, time_fe, L, H = lax.while_loop(cond_fun, body_fun, init_carry)

    E = compute_decomposition(
        L, X_tilde, Z_tilde, V, H, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )
    P_omega = mask_observed(Y - E, W)
    _, _, s = compute_svd(P_omega)
    lambda_L_max = 2.0 * jnp.max(s) / jnp.sum(W)

    num_train = jnp.sum(W)
    T_mat = jnp.zeros((Y.size, H_rows * H_cols))
    in_prod_T = jnp.zeros(H_rows * H_cols)

    def compute_T_mat(j, val):
        T_mat, in_prod_T = val
        for i in range(H_rows):
            out_prod = mask_observed(jnp.outer(X_tilde[:, i], Z_tilde[:, j]), W)
            index = j * H_rows + i
            T_mat = T_mat.at[:, index].set(out_prod.ravel())
            in_prod_T = in_prod_T.at[index].set(jnp.sum(T_mat[:, index] ** 2))
        return T_mat, in_prod_T

    T_mat, in_prod_T = lax.fori_loop(0, H_cols, compute_T_mat, (T_mat, in_prod_T))

    T_mat /= jnp.sqrt(num_train)
    in_prod_T /= num_train

    P_omega_resh = P_omega.ravel()
    all_Vs = jnp.dot(T_mat.T, P_omega_resh) / jnp.sqrt(num_train)
    lambda_H_max = 2 * jnp.max(jnp.abs(all_Vs))

    return unit_fe, time_fe, lambda_L_max, lambda_H_max, T_mat, in_prod_T


#
#
# def update_L(
#     M: Array, mask: Array, L: Array, u: Array, v: Array, lambda_L: Scalar
# ) -> Tuple[Array, Array]:
#     """
#     Update the low-rank matrix L using the observed matrix M, mask, current estimates of L, u, v, and regularization
#     parameter.
#     Return the updated L and its singular values.
#     """
#     # TODO: Implement the update step for L
#     pass
#
#
# def update_H(
#     M: Array,
#     X: Array,
#     Z: Array,
#     H: Array,
#     mask: Array,
#     L: Array,
#     u: Array,
#     v: Array,
#     lambda_H: Scalar,
#     to_add_ID: bool,
# ) -> Tuple[Array, Array]:
#     """
#     Update the covariate coefficient matrix H using the observed matrix M, covariates X and Z, current estimates of
#     H, L, u, v, and regularization parameter.
#     Handle the case where an identity matrix should be added to the covariates (to_add_ID).
#     Return the updated H and the inner product of the augmented covariate matrix.
#     """
#     # TODO: Implement the update step for H
#     pass
#

#
#
# def fit(
#     Y: Array,
#     W: Array,
#     X: Array,
#     Z: Array,
#     V: Array,
#     Omega: Array,
#     lambda_L: Scalar,
#     lambda_H: Scalar,
#     initial_params: Tuple[Array, Array, Array, Array, Array],
#     max_iter: int,
#     tol: Scalar,
#     use_unit_fe: bool,
#     use_time_fe: bool,
#     to_add_ID: bool,
# ) -> Tuple[Array, Array, Array, Array, Array]:
#     """
#     Fit the MC-NNM model using coordinate descent updates until convergence or maximum iterations.
#     Handle cases where covariates are not present by passing zero arrays.
#     Apply normalization to X and Z covariates using the normalize function.
#     Rescale the estimated H matrix using normalize_back_rows and normalize_back_cols functions.
#     Use the compute_obj_val and compute_obj_val_H functions to track the objective function value during the
#     optimization process.
#     Return the final estimates of L, H, u, v, and beta.
#     """
#     # TODO: Implement the model fitting process
#     pass
