from typing import Tuple, Optional

import jax.numpy as jnp
from jax import jit

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
    unit_fe = jnp.ones(N) if use_unit_fe else jnp.zeros(N)
    time_fe = jnp.ones(T) if use_time_fe else jnp.zeros(T)

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
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        H (Array): The covariate coefficients matrix of shape (P, Q).
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
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        H (Array): The covariate coefficients matrix of shape (P, Q).
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
    r"""
    Compute the matrix decomposition :math:`L^* + X H^* Z^T + H^*_Z Z^T + X H^*_X + \Gamma^* 1_T^T + 1_N (\Delta^*)^T +
    [V_{it} \beta^*]_{it} + \varepsilon`.

    Args:
        L (Array): The low-rank matrix :math:`L^*` of shape (N, T).
        X (Array): The unit-specific covariates matrix :math:`X` of shape (N, P).
        Z (Array): The time-specific covariates matrix :math:`Z` of shape (T, Q).
        V (Array): The unit-time-specific covariates tensor :math:`V` of shape (N, T, J).
        H (Array): The covariate coefficients matrix :math:`H^*` of shape (P + N, Q + T).
        gamma (Array): The unit fixed effects vector :math:`\Gamma^*` of shape (N,).
        delta (Array): The time fixed effects vector :math:`\Delta^*` of shape (T,).
        beta (Array): The unit-time-specific covariate coefficients vector :math:`\beta^*` of shape (J,).
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.

    Returns:
        Array: The computed matrix decomposition of shape (N, T).
    """
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

    return obj_val
