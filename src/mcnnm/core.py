from typing import Tuple, Optional
from .core_utils import mask_observed
from .types import Array, Scalar

import jax.numpy as jnp
from jax import jit, lax
import jax.debug as jdb
import jax

jax.config.update("jax_enable_x64", True)


@jit
def initialize_coefficients(
    Y: Array, X_tilde: Array, Z_tilde: Array, V: Array
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Initialize covariate and fixed effects coefficients for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).

    Returns:
        Tuple[Array, Array, Array, Array, Array]: A tuple containing initial values for L,
        H_tilde, gamma, delta, and beta.
    """
    N, T = Y.shape
    L = jnp.zeros_like(Y)
    gamma = jnp.zeros(N)  # unit FE coefficients
    delta = jnp.zeros(T)  # time FE coefficients

    H_tilde = jnp.zeros(
        (X_tilde.shape[1], Z_tilde.shape[1])
    )  # X_tilde and Z_tilde-covariate coefficients

    beta_shape = max(V.shape[2], 1)
    beta = jnp.zeros((beta_shape,))  # unit-time covariate coefficients

    return L, H_tilde, gamma, delta, beta


@jit
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

    # Add identity matrices to X and Z to obtain X_tilde and Z_tilde
    X_tilde = jnp.concatenate((X, jnp.eye(N)), axis=1)
    Z_tilde = jnp.concatenate((Z, jnp.eye(T)), axis=1)

    # Initialize unit and time fixed effects
    unit_fe = jnp.where(use_unit_fe, jnp.ones(N), jnp.zeros(N))
    time_fe = jnp.where(use_time_fe, jnp.ones(T), jnp.zeros(T))

    return L, X_tilde, Z_tilde, V, unit_fe, time_fe


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
def svt(U: Array, V: Array, sing_vals: Array, threshold: Scalar) -> Array:
    """
    Perform singular value thresholding (SVT) on the given singular value decomposition.

    Args:
        U (Array): The left singular vectors matrix.
        V (Array): The right singular vectors matrix.
        sing_vals (Array): The singular values array.
        threshold (Scalar): The thresholding value.

    Returns:
        Array: The thresholded low-rank matrix.
    """
    thresholded_sing_vals = jnp.maximum(sing_vals - threshold, 0)
    return U @ jnp.diag(thresholded_sing_vals) @ V.T


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
def update_beta(
    Y: Array,
    X: Array,
    Z: Array,
    V: Array,
    H: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    time_fe: Array,
) -> Array:
    """
    Update the unit-time-specific covariate coefficients (beta) in the coordinate descent algorithm.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        H (Array): The covariate coefficients matrix of shape (P, Q).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        unit_fe (Array): The unit fixed effects vector of shape (N,).
        time_fe (Array): The time fixed effects vector of shape (T,).

    Returns:
        Array: The updated unit-time-specific covariate coefficients vector of shape (J,).
    """
    T_ = jnp.einsum("np,pq,tq->nt", X, H, Z)
    b_ = T_ + L + jnp.expand_dims(unit_fe, axis=1) + time_fe - Y
    b_mask_ = b_ * W

    V_mask_ = V * jnp.expand_dims(W, axis=-1)
    V_sum_ = jnp.sum(V_mask_, axis=(0, 1))

    V_b_prod_ = jnp.einsum("ntj,nt->j", V_mask_, b_mask_)

    return jnp.where(V_sum_ > 0, -V_b_prod_ / V_sum_, 0.0)


@jit
def compute_decomposition(
    L: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Array:
    """
    Compute the decomposition of the observed outcome matrix Y.

    This function computes the decomposition of the observed outcome matrix Y
    into its low-rank component L, covariate effects, and fixed effects (unit and time).
    The decomposition is given by:
        Y ≈ L + X @ H[:P, :Q] @ Z.T + V @ beta + gamma @ 1_T + 1_N @ delta

    Args:
        L (Array): The low-rank matrix of shape (N, T).
        X_tilde (Array): The unit-specific covariates matrix of shape (N, P).
        Z_tilde (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        H_tilde (Array): The covariate coefficients matrix of shape (P + N, Q + T).
        gamma (Array): The unit fixed effects vector of shape (N,).
        delta (Array): The time fixed effects vector of shape (T,).
        beta (Array): The unit-time-specific covariate coefficients vector of shape (J,).
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.

    Returns:
        Array: The decomposed matrix of shape (N, T).
    """
    N, T = L.shape
    P = X_tilde.shape[1]
    Q = Z_tilde.shape[1]

    decomposition = L

    unit_fe_term = jnp.outer(gamma, jnp.ones(T))
    decomposition += jnp.where(use_unit_fe, unit_fe_term, jnp.zeros_like(unit_fe_term))

    time_fe_term = jnp.outer(jnp.ones(N), delta)
    decomposition += jnp.where(use_time_fe, time_fe_term, jnp.zeros_like(time_fe_term))

    if Q > 0:
        decomposition += X_tilde @ H_tilde[:P, :Q] @ Z_tilde.T
    if H_tilde.shape[1] > Q:
        decomposition += X_tilde @ H_tilde[:P, Q:]
    if P + N <= H_tilde.shape[0] and Q > 0:
        decomposition += H_tilde[P : P + N, :Q] @ Z_tilde.T
    decomposition += jnp.einsum("ntj,j->nt", V, beta)

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
) -> Tuple[Array, Array, Scalar, Scalar, Array, Array, Array]:
    """
    Initialize fixed effects and the matrix H for the MC-NNM model.

    This function initializes the fixed effects (unit and time) and the matrix H
    using an iterative coordinate descent algorithm. It also computes the maximum
    regularization parameters for the low-rank matrix L and the covariate coefficients
    matrix H.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.
        niter (int, optional): The maximum number of iterations for the coordinate descent algorithm. Default is 1000.
        rel_tol (float, optional): The relative tolerance for convergence. Default is 1e-5.

    Returns:
        Tuple[Array, Array, Scalar, Scalar, Array, Array]: A tuple containing:
            - unit_fe (Array): The unit fixed effects vector of shape (N,).
            - time_fe (Array): The time fixed effects vector of shape (T,).
            - lambda_L_max (Scalar): The maximum regularization parameter for the nuclear norm of L.
            - lambda_H_max (Scalar): The maximum regularization parameter for the element-wise L1 norm of H.
            - T_mat (Array): The matrix T used for computing the regularization parameter lambda_H_max.
            - in_prod_T (Array): The inner product of T_mat used for computing lambda_H_max.
            - in_prod (Array): The inner product vector used for updating H. Initialized as zeros.
    """
    N, T = Y.shape
    num_train = jnp.sum(W)
    in_prod = jnp.zeros_like(W)
    L, X_tilde, Z_tilde, V, unit_fe, time_fe = initialize_matrices(
        Y, X, Z, V, use_unit_fe, use_time_fe
    )
    _, H, _, _, beta = initialize_coefficients(Y, X_tilde, Z_tilde, V)

    H_rows, H_cols = X_tilde.shape[1], Z_tilde.shape[1]

    def body_fun(carry):
        obj_val, unit_fe, time_fe = carry
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

        rel_error = jnp.abs(new_obj_val - obj_val) / (jnp.abs(obj_val) + 1e-10)
        obj_val = jnp.where(rel_error < rel_tol, obj_val, new_obj_val)
        return obj_val, unit_fe, time_fe

    init_val = (jnp.inf, unit_fe, time_fe)
    _, unit_fe, time_fe = lax.fori_loop(0, niter, lambda i, x: body_fun(x), init_val)

    E = compute_decomposition(
        L, X_tilde, Z_tilde, V, H, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )
    P_omega = mask_observed(Y - E, W)
    s = jnp.linalg.svd(P_omega, compute_uv=False)
    lambda_L_max = 2.0 * jnp.max(s) / num_train

    T_mat = jnp.zeros((Y.size, H_rows * H_cols))

    def compute_T_mat(j, T_mat):
        out_prod = mask_observed(jnp.outer(X_tilde[:, j // H_rows], Z_tilde[:, j % H_cols]), W)
        return T_mat.at[:, j].set(out_prod.ravel())

    T_mat = lax.fori_loop(0, H_rows * H_cols, compute_T_mat, T_mat)
    T_mat /= jnp.sqrt(num_train)

    in_prod_T = jnp.sum(T_mat**2, axis=0)

    P_omega_resh = P_omega.ravel()
    all_Vs = jnp.dot(T_mat.T, P_omega_resh) / jnp.sqrt(num_train)
    lambda_H_max = 2 * jnp.max(jnp.abs(all_Vs))

    return unit_fe, time_fe, lambda_L_max, lambda_H_max, T_mat, in_prod_T, in_prod


@jit
def update_H(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    T_mat: Array,
    in_prod: Array,
    in_prod_T: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    time_fe: Array,
    beta: Array,
    lambda_H: Scalar,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Tuple[Array, Array]:
    """
    Update the covariate coefficients matrix H_tilde in the coordinate descent algorithm.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        H_tilde (Array): The covariate coefficients matrix of shape (P+N, Q+T).
        T_mat (Array): The precomputed matrix T of shape (N * T, (P+N) * (Q+T)).
        in_prod (Array): The inner product vector of shape (N * T,).
        in_prod_T (Array): The inner product vector of T of shape ((P+N) * (Q+T),).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        unit_fe (Array): The unit fixed effects vector of shape (N,).
        time_fe (Array): The time fixed effects vector of shape (T,).
        beta (Array): The unit-time-specific covariate coefficients vector of shape (J,).
        lambda_H (Scalar): The regularization parameter for the element-wise L1 norm of H_tilde.
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.

    Returns:
        Tuple[Array, Array]: A tuple containing the updated covariate coefficients matrix H_tilde and the updated inner
        product vector in_prod.
    """
    H_tilde_rows, H_tilde_cols = X_tilde.shape[1], Z_tilde.shape[1]
    num_train = jnp.sum(W)

    M_hat = compute_decomposition(
        L, X_tilde, Z_tilde, V, H_tilde, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )
    residual = (Y - M_hat) * W / jnp.sqrt(num_train)
    residual_flat = residual.ravel()

    H_tilde_flat = H_tilde.ravel()
    updated_in_prod = in_prod.copy()

    X_cols, Z_cols = X_tilde.shape[1] - Y.shape[0], Z_tilde.shape[1] - Y.shape[1]

    def update_H_elem(carry, idx):
        updated_in_prod, H_tilde_flat = carry
        col, row = jnp.divmod(idx, H_tilde_rows)
        elem_idx = col * H_tilde_rows + row
        in_prod_T_elem = in_prod_T[elem_idx]
        H_tilde_new = jnp.where(
            in_prod_T_elem != 0,
            0.5
            * (
                jnp.maximum(
                    (
                        2
                        * jnp.dot(
                            residual_flat
                            - updated_in_prod
                            + T_mat[:, elem_idx] * H_tilde_flat[elem_idx],
                            T_mat[:, elem_idx],
                        )
                        - lambda_H
                    )
                    / in_prod_T_elem,
                    0,
                )
                - jnp.maximum(
                    (
                        -2
                        * jnp.dot(
                            residual_flat
                            - updated_in_prod
                            + T_mat[:, elem_idx] * H_tilde_flat[elem_idx],
                            T_mat[:, elem_idx],
                        )
                        - lambda_H
                    )
                    / in_prod_T_elem,
                    0,
                )
            ),
            H_tilde_flat[elem_idx],
        )
        updated_in_prod += (H_tilde_new - H_tilde_flat[elem_idx]) * T_mat[:, elem_idx]
        H_tilde_flat = H_tilde_flat.at[elem_idx].set(H_tilde_new)
        return (updated_in_prod, H_tilde_flat), None

    if Z_cols > 0:
        (updated_in_prod, H_tilde_flat), _ = jax.lax.scan(
            update_H_elem, (updated_in_prod, H_tilde_flat), jnp.arange(Z_cols * H_tilde_rows)
        )

    if X_cols > 0:
        (updated_in_prod, H_tilde_flat), _ = jax.lax.scan(
            update_H_elem,
            (updated_in_prod, H_tilde_flat),
            jnp.arange(Z_cols * H_tilde_rows, H_tilde_cols * H_tilde_rows),
        )

    H_tilde_updated = H_tilde_flat.reshape(H_tilde_rows, H_tilde_cols)

    return H_tilde_updated, updated_in_prod


@jit
def update_L(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    time_fe: Array,
    beta: Array,
    lambda_L: Scalar,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Tuple[Array, Array]:
    """
    Update the low-rank matrix L in the coordinate descent algorithm.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        H_tilde (Array): The covariate coefficients matrix of shape (P+N, Q+T).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        unit_fe (Array): The unit fixed effects vector of shape (N,).
        time_fe (Array): The time fixed effects vector of shape (T,).
        beta (Array): The unit-time-specific covariate coefficients vector of shape (J,).
        lambda_L (Scalar): The regularization parameter for the nuclear norm of L.
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.

    Returns:
        Tuple[Array, Array]: A tuple containing the updated low-rank matrix L and the singular values.
    """
    num_train = jnp.sum(W)
    P_mat = compute_decomposition(
        L, X_tilde, Z_tilde, V, H_tilde, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )
    P_omega = Y - P_mat
    masked_P_omega = mask_observed(P_omega, W)
    proj = masked_P_omega + L

    U, S, Vt = jnp.linalg.svd(proj, full_matrices=False)
    V = Vt.T

    L_upd = svt(U, V, S, lambda_L * num_train / 2)

    return L_upd, S
