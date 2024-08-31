from typing import Tuple, Optional, cast

import jax
import jax.debug as jdb
import jax.numpy as jnp
from jax import jit, lax

from .core_utils import mask_observed, element_wise_l1_norm, is_positive_definite
from .types import Array, Scalar

jax.config.update("jax_enable_x64", True)


@jit
def initialize_coefficients(
    Y: Array, X_tilde: Array, Z_tilde: Array, V: Array
) -> Tuple[Array, Array, Array, Array]:
    """
    Initialize covariate and fixed effects coefficients for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing initial values for:
            - H_tilde: Initial value for the interactive fixed effects matrix.
            - gamma: Initial value for unit fixed effects.
            - delta: Initial value for time fixed effects.
            - beta: Initial value for covariate coefficients.
    """
    N, T = Y.shape
    gamma = jnp.zeros(N)  # unit FE coefficients
    delta = jnp.zeros(T)  # time FE coefficients

    H_tilde = jnp.zeros(
        (X_tilde.shape[1], Z_tilde.shape[1])
    )  # X_tilde and Z_tilde covariate coefficients

    beta_shape = max(V.shape[2], 0)
    beta = jnp.zeros((beta_shape,))  # unit-time covariate coefficients

    return H_tilde, gamma, delta, beta


@jit
def initialize_matrices(
    Y: Array,
    X: Optional[Array],
    Z: Optional[Array],
    V: Optional[Array],
) -> Tuple[Array, Array, Array, Array]:
    """
    Initialize the matrices for the MC-NNM model.

    This function initializes the low-rank matrix L and the covariate matrices X_tilde and Z_tilde.
    If the covariate matrices X, Z, or V are not provided, they are initialized to zero matrices/tensors.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Optional[Array]): The unit-specific covariates matrix of shape (N, P). If None, initialized to zeros.
        Z (Optional[Array]): The time-specific covariates matrix of shape (T, Q). If None, initialized to zeros.
        V (Optional[Array]): The unit-time-specific covariates tensor of shape (N, T, J). If None, initialized to zeros.

    Returns:
        Tuple[Array, Array, Array, Array]: A tuple containing:
            - L (Array): The low-rank matrix of shape (N, T).
            - X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
            - Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
            - V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
    """
    N, T = Y.shape
    P = X.shape[1] if X is not None else 0
    Q = Z.shape[1] if Z is not None else 0
    J = V.shape[2] if V is not None else 0

    # Initialize X, Z, and V to zero matrices if None
    X = jnp.zeros((N, P)) if X is None else X
    Z = jnp.zeros((T, Q)) if Z is None else Z
    V = jnp.zeros((N, T, J)) if V is None else V

    # Add identity matrices to X and Z to obtain X_tilde and Z_tilde
    X_tilde = jnp.concatenate((X, jnp.eye(N)), axis=1)
    Z_tilde = jnp.concatenate((Z, jnp.eye(T)), axis=1)

    # Initialize L to a zero matrix
    L = jnp.zeros((N, T))

    return L, X_tilde, Z_tilde, V


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
    Perform soft singular value thresholding (SVT) on the given singular value decomposition.

    Args:
        U (Array): The left singular vectors matrix.
        V (Array): The right singular vectors matrix.
        sing_vals (Array): The singular values array.
        threshold (Scalar): The thresholding value.

    Returns:
        Array: The thresholded low-rank matrix.
    """
    soft_thresholded_sing_vals = jnp.maximum(sing_vals - threshold, 0)
    return U @ jnp.diag(soft_thresholded_sing_vals) @ V.T


@jit
def update_unit_fe(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    time_fe: Array,
    use_unit_fe: bool,
) -> Array:
    """
    Update the unit fixed effects in the coordinate descent algorithm when covariates are available.

    This function calculates the unit fixed effects by minimizing the squared error of the model
    predictions with respect to the observed outcomes. It accounts for the covariate effects,
    low-rank matrix, and time fixed effects in the calculation.

    The update process involves the following steps:
    1. Compute the covariate contribution to the predicted outcomes.
    2. Calculate the total predicted outcomes (including L and time fixed effects).
    3. Compute the residuals (observed - predicted).
    4. Apply the observation mask to the residuals.
    5. Calculate the average residual for each unit, accounting for missing observations.
    6. Return the negative of these averages as the updated unit fixed effects.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        H_tilde (Array): The augmented covariate coefficients matrix of shape (P+N, Q+T).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        time_fe (Array): The time fixed effects vector of shape (T,).
        use_unit_fe (bool): Whether to estimate unit fixed effects.

    Returns:
        Array: The updated unit fixed effects vector of shape (N,) if use_unit_fe is True, else a zero vector.
    """
    # Compute the covariate contribution to the predicted outcomes
    covariate_contribution = jnp.einsum("np,pq,tq->nt", X_tilde, H_tilde, Z_tilde)

    # Calculate the total predicted outcomes (without unit fixed effects)
    Y_hat = covariate_contribution + L + time_fe

    # Compute the residuals (observed - predicted)
    residuals = Y_hat - Y

    # Apply the observation mask to the residuals
    masked_residuals = mask_observed(residuals, W)

    # Count the number of observed entries for each unit
    num_observed_per_unit = jnp.sum(W, axis=1)

    # Calculate the average residual for each unit, avoiding division by zero
    average_residuals = jnp.where(
        num_observed_per_unit > 0,
        -jnp.sum(masked_residuals, axis=1) / (num_observed_per_unit + 1e-8),
        0.0,
    )

    # Return the updated unit fixed effects if use_unit_fe is True, else return zeros
    return jnp.where(use_unit_fe, average_residuals, jnp.zeros_like(average_residuals))


@jit
def update_time_fe(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    use_time_fe: bool,
) -> Array:
    """
    Update the time fixed effects in the coordinate descent algorithm when covariates are available.

    This function calculates the time fixed effects by minimizing the squared error of the model
    predictions with respect to the observed outcomes. It accounts for the covariate effects,
    low-rank matrix, and unit fixed effects in the calculation.

    The update process involves the following steps:
    1. Compute the covariate contribution to the predicted outcomes.
    2. Calculate the total predicted outcomes (including L and unit fixed effects).
    3. Compute the residuals (observed - predicted).
    4. Apply the observation mask to the residuals.
    5. Calculate the average residual for each time period, accounting for missing observations.
    6. Return the negative of these averages as the updated time fixed effects.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        H_tilde (Array): The augmented covariate coefficients matrix of shape (P+N, Q+T).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        unit_fe (Array): The unit fixed effects vector of shape (N,).
        use_time_fe (bool): Whether to estimate time fixed effects.

    Returns:
        Array: The updated time fixed effects vector of shape (T,) if use_time_fe is True, else a zero vector.
    """
    # Compute the covariate contribution to Y hat
    covariate_contribution = jnp.einsum("np,pq,tq->nt", X_tilde, H_tilde, Z_tilde)

    # Calculate the total predicted outcomes (without time fixed effects)
    Y_hat = covariate_contribution + L + jnp.expand_dims(unit_fe, axis=1)

    # Compute the residuals (observed - predicted)
    residuals = Y_hat - Y

    # Apply the observation mask to the residuals
    masked_residuals = mask_observed(residuals, W)

    # Count the number of observed entries for each time period
    num_observed_per_time = jnp.sum(W, axis=0)

    # Calculate the average residual for each time period, avoiding division by zero
    average_residuals = jnp.where(
        num_observed_per_time > 0,
        -jnp.sum(masked_residuals, axis=0) / (num_observed_per_time + 1e-8),
        0.0,
    )

    # Return the updated time fixed effects if use_time_fe is True, else return zeros
    return jnp.where(use_time_fe, average_residuals, jnp.zeros_like(average_residuals))


@jit
def update_beta(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    unit_fe: Array,
    time_fe: Array,
) -> Array:
    """
    Update the unit-time-specific covariate coefficients (beta) in the coordinate descent algorithm.

    This function calculates the coefficients for unit-time-specific covariates by minimizing
    the squared error of the model predictions with respect to the observed outcomes. It accounts
    for the other components of the model (covariate effects, low-rank matrix, unit and time fixed effects)
    in the calculation.

    The update process involves the following steps:
    1. Compute the covariate contribution to the predicted outcomes.
    2. Calculate the total predicted outcomes (including all components except V*beta).
    3. Compute the residuals (observed - predicted).
    4. Apply the observation mask to the residuals and unit-time-specific covariates.
    5. Calculate the updated beta coefficients using the masked residuals and covariates.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        H_tilde (Array): The augmented covariate coefficients matrix of shape (P+N, Q+T).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix of shape (N, T).
        unit_fe (Array): The unit fixed effects vector of shape (N,).
        time_fe (Array): The time fixed effects vector of shape (T,).

    Returns:
        Array: The updated unit-time-specific covariate coefficients vector of shape (J,).
    """
    # Compute the covariate contribution to Y hat
    covariate_contribution = jnp.einsum("np,pq,tq->nt", X_tilde, H_tilde, Z_tilde)

    # Calculate Y hat (without V*beta)
    Y_hat = covariate_contribution + L + jnp.expand_dims(unit_fe, axis=1) + time_fe

    # Compute the residuals (observed - predicted)
    residuals = Y_hat - Y

    # Apply the observation mask to the residuals
    masked_residuals = mask_observed(residuals, W)

    # Apply the observation mask to the unit-time-specific covariates
    masked_V = V * jnp.expand_dims(W, axis=-1)

    # Sum of masked unit-time-specific covariates for each covariate
    V_sums = jnp.sum(masked_V, axis=(0, 1))

    # Compute the product of masked residuals and masked unit-time-specific covariates
    V_residual_products = jnp.einsum("ntj,nt->j", masked_V, masked_residuals)

    # Calculate updated beta coefficients, avoiding division by zero
    updated_beta = jnp.where(V_sums > 0, -V_residual_products / (V_sums + 1e-8), 0.0)

    return updated_beta


@jit
def compute_Y_hat(
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

    .. math::
        \\hat{Y} ≈ L + X @ H[:P, :Q] @ Z.T + V @ beta + gamma @ 1_T + 1_N @ delta

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
        Array: The estimated matrix \\(\\hat{Y}\\) of shape (N, T).
    """
    N, T = L.shape
    P = X_tilde.shape[1]  # Number of unit-specific covariates
    Q = Z_tilde.shape[1]  # Number of time-specific covariates

    # Start with the low-rank component
    Y_hat = L

    # Add unit FEs
    unit_fe_term = jnp.outer(gamma, jnp.ones(T))
    Y_hat += jnp.where(use_unit_fe, unit_fe_term, jnp.zeros_like(unit_fe_term))

    # Add time FEs
    time_fe_term = jnp.outer(jnp.ones(N), delta)
    Y_hat += jnp.where(use_time_fe, time_fe_term, jnp.zeros_like(time_fe_term))

    if Q > 0:
        Y_hat += X_tilde @ H_tilde[:P, :Q] @ Z_tilde.T
    if H_tilde.shape[1] > Q:
        Y_hat += X_tilde @ H_tilde[:P, Q:]
    if P + N <= H_tilde.shape[0] and Q > 0:
        Y_hat += H_tilde[P : P + N, :Q] @ Z_tilde.T
    V_beta_term = jnp.einsum("ntj,j->nt", V, beta)
    Y_hat += V_beta_term

    return Y_hat


def compute_objective_value(
    Y: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    W: Array,
    L: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    sum_sing_vals: Scalar,
    lambda_L: Scalar,
    lambda_H: Scalar,
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
    - :math:`\widehat{Y}_{it}` is the estimated outcome for unit :math:`i` at time :math:`t`, given by:

      .. math::

          \hat{Y}_{it} = L^*_{it} + \sum_{p=1}^{P+N} \tilde{X}_{ip} \tilde{H}^*_{pq} \tilde{Z}_{tq}
          + \Gamma^*_i + \Delta^*_t + \sum_{j=1}^J V_{itj} \beta^*_j

    - :math:`\Omega` is the set of observed entries in the outcome matrix
    - :math:`\Omega^{-1}` is the inverse of the omega matrix, capturing the time
      series correlation
    - :math:`\hat{L}` is the low-rank matrix of shape (N, T)
    - :math:`\tilde{X}` is the augmented unit-specific covariates matrix of shape (N, P+N)
    - :math:`\tilde{Z}` is the augmented time-specific covariates matrix of shape (T, Q+T)
    - :math:`V` is the unit-time-specific covariates tensor of shape (N, T, J)
    - :math:`\tilde{H}` is the augmented covariate coefficients matrix of shape (P+N, Q+T)
    - :math:`\Gamma` is the unit fixed effects vector of shape (N,)
    - :math:`\Delta` is the time fixed effects vector of shape (T,)
    - :math:`\beta` is the unit-time-specific covariate coefficients vector
      of shape (J,)
    - :math:`\lambda_L` is the regularization parameter for the nuclear norm of
      :math:`L`
    - :math:`\lambda_H` is the regularization parameter for the element-wise L1 norm
      of :math:`\tilde{H}`

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix :math:`\tilde{X}` of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix :math:`\tilde{Z}` of shape (T, Q+T).
        V (Array): The unit-time-specific covariates tensor :math:`V` of shape
            (N, T, J).
        H_tilde (Array): The augmented covariate coefficients matrix :math:`\tilde{H}` of shape
            (P+N, Q+T).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The low-rank matrix :math:`L` of shape (N, T).
        gamma (Array): The unit fixed effects vector :math:`\Gamma` of shape (N,).
        delta (Array): The time fixed effects vector :math:`\Delta` of shape (T,).
        beta (Array): The unit-time-specific covariate coefficients vector
            :math:`\beta` of shape (J,).
        sum_sing_vals (Scalar): The sum of singular values of L.
        lambda_L (Scalar): The regularization parameter for the nuclear norm of L.
        lambda_H (Scalar): The regularization parameter for the element-wise L1 norm
            of H.
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.
        inv_omega (Optional[Array]): The inverse of the omega matrix of shape (T, T).
            If None, the identity matrix is used.

    Returns:
        Scalar: The computed objective value.
    """
    train_size = jnp.sum(W)
    norm_H = element_wise_l1_norm(H_tilde)

    Y_hat = compute_Y_hat(
        L, X_tilde, Z_tilde, V, H_tilde, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    error_matrix = Y_hat - Y

    if inv_omega is None:
        inv_omega = jnp.eye(Y.shape[1])

    lax.cond(
        is_positive_definite(inv_omega),
        lambda _: None,
        lambda _: jdb.print("WARNING: inv_omega is not positive definite"),
        None,
    )

    error_mask = mask_observed(error_matrix, W)  # mask the error matrix
    weighted_error_term = (1 / train_size) * jnp.trace(error_mask @ inv_omega @ error_mask.T)

    L_regularization_term = lambda_L * sum_sing_vals

    H_regularization_term = lambda_H * norm_H

    obj_val = weighted_error_term + L_regularization_term + H_regularization_term

    return obj_val


@jit
def initialize_fixed_effects_and_H(
    Y: Array,
    L: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    W: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
    niter: int = 1000,
    rel_tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Scalar, Scalar]:
    """
    Initialize fixed effects and the matrix H_tilde for the MC-NNM model.

    This function initializes the fixed effects (unit and time) and the matrix H_tilde
    using an iterative coordinate descent algorithm. It also computes the maximum
    regularization parameters for the low-rank matrix L and the covariate coefficients
    matrix H_tilde.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        L (Array): The zero-initialised low-rank matrix of shape (N, T).
        X_tilde (Array): The unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.
        niter (int, optional): The maximum number of iterations for the coordinate descent algorithm. Default is 1000.
        rel_tol (float, optional): The relative tolerance for convergence. Default is 1e-5.
        verbose (bool, optional): Whether to print the objective value after initialization. Default is False.
    Returns:
        Tuple[Array, Array, Scalar, Scalar, Array, Array]: A tuple containing:
            - gamma (Array): The unit fixed effects vector of shape (N,).
            - delta (Array): The time fixed effects vector of shape (T,).
            - beta (Array): The unit-time-specific covariate coefficients vector of shape (J,).
            - H_tilde (Array): The covariate coefficients matrix of shape (P+N, Q+T).
            - T_mat (Array): The matrix T used for computing the regularization parameter lambda_H_max.
            - in_prod_T (Array): The inner product of T_mat used for computing lambda_H_max.
            - in_prod (Array): The inner product vector used for updating H_tilde. Initialized as zeros.
            - lambda_L_max (Scalar): The maximum regularization parameter for the nuclear norm of L.
            - lambda_H_max (Scalar): The maximum regularization parameter for the element-wise L1 norm of H_tilde.
    """
    num_train = jnp.sum(W)
    in_prod = jnp.zeros_like(W)

    H_tilde, gamma, delta, beta = initialize_coefficients(Y, X_tilde, Z_tilde, V)

    H_rows, H_cols = X_tilde.shape[1], Z_tilde.shape[1]

    def cond_fun(carry):
        obj_val, prev_obj_val, _, _, i = carry
        rel_error = jnp.where(
            jnp.isfinite(prev_obj_val),
            (obj_val - prev_obj_val) / (jnp.abs(prev_obj_val) + 1e-8),
            jnp.inf,
        )
        return lax.cond(
            ((rel_error < rel_tol) & (rel_error > 0)),
            lambda _: False,
            lambda _: i < niter,
            None,
        )

    def body_fun(carry):
        obj_val, prev_obj_val, gamma, delta, i = carry
        gamma = update_unit_fe(Y, X_tilde, Z_tilde, H_tilde, W, L, delta, use_unit_fe)
        delta = update_time_fe(Y, X_tilde, Z_tilde, H_tilde, W, L, gamma, use_time_fe)

        new_obj_val = compute_objective_value(
            Y,
            X_tilde,
            Z_tilde,
            V,
            H_tilde,
            W,
            L,
            gamma,
            delta,
            beta,
            0.0,
            0.0,
            0.0,
            use_unit_fe,
            use_time_fe,
        )

        return new_obj_val, obj_val, gamma, delta, i + 1

    init_val = (1e10, 1e10, gamma, delta, 0)
    obj_val, _, gamma, delta, _ = lax.while_loop(cond_fun, body_fun, init_val)

    Y_hat = compute_Y_hat(
        L, X_tilde, Z_tilde, V, H_tilde, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    masked_error_matrix = mask_observed(Y - Y_hat, W)
    s = jnp.linalg.svd(masked_error_matrix, compute_uv=False)
    lambda_L_max = 2.0 * jnp.max(s) / num_train
    lambda_L_max = cast(Scalar, lambda_L_max)  # type: ignore[assignment]

    T_mat = jnp.zeros((Y.size, H_rows * H_cols))

    def compute_T_mat(j, T_mat):
        out_prod = mask_observed(jnp.outer(X_tilde[:, j // H_rows], Z_tilde[:, j % H_cols]), W)
        return T_mat.at[:, j].set(out_prod.ravel())

    T_mat = lax.fori_loop(0, H_rows * H_cols, compute_T_mat, T_mat)
    T_mat /= jnp.sqrt(num_train)

    in_prod_T = jnp.sum(T_mat**2, axis=0)

    P_omega_resh = masked_error_matrix.ravel()
    all_Vs = jnp.dot(T_mat.T, P_omega_resh) / jnp.sqrt(num_train)
    lambda_H_max = 2 * jnp.max(jnp.abs(all_Vs))
    lambda_H_max = cast(Scalar, lambda_H_max)  # type: ignore[assignment]

    # Truncate the value to 5 decimal places for printing
    truncated_ov = jnp.round(obj_val, decimals=5)

    lax.cond(
        verbose,
        lambda _: jdb.print("Initialization complete, objective value: {ov}", ov=truncated_ov),
        lambda _: None,
        None,
    )

    return gamma, delta, beta, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max


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

    This function implements the update step for the covariate coefficients matrix (H_tilde) in the matrix
    completion algorithm. It uses a soft-thresholding operator to update each element of the coefficient matrix,
    taking into account the regularization parameter and the current residuals.

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
    # Get dimensions of the covariate coefficients matrix
    coeff_rows, coeff_cols = X_tilde.shape[1], Z_tilde.shape[1]
    num_observed = jnp.sum(W)

    # Compute the current predicted outcomes
    Y_hat = compute_Y_hat(
        L, X_tilde, Z_tilde, V, H_tilde, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )

    # Compute the residuals
    error_matrix = Y - Y_hat
    masked_error_matrix = mask_observed(error_matrix, W)
    residuals = masked_error_matrix / jnp.sqrt(num_observed)
    residuals_flat = residuals.ravel()

    # Flatten the covariate coefficients matrix
    coeff_flat = H_tilde.ravel()
    updated_in_prod = in_prod.copy()

    # Get the number of unit-specific and time-specific covariates
    num_unit_covariates = X_tilde.shape[1] - Y.shape[0]
    num_time_covariates = Z_tilde.shape[1] - Y.shape[1]

    def update_coefficient(carry, idx):
        """Update a single coefficient using soft-thresholding."""
        current_in_prod, current_coeffs = carry
        U = in_prod_T[idx]
        current_in_prod_flat = current_in_prod.ravel()

        # Compute the update using soft-thresholding
        new_coeff = jnp.where(
            U != 0,
            0.5
            * (
                jnp.maximum(
                    (
                        2
                        * jnp.dot(
                            residuals_flat
                            - current_in_prod_flat
                            + T_mat[:, idx] * current_coeffs[idx],
                            T_mat[:, idx],
                        )
                        - lambda_H
                    )
                    / U,
                    0,
                )
                - jnp.maximum(
                    (
                        -2
                        * jnp.dot(
                            residuals_flat
                            - current_in_prod_flat
                            + T_mat[:, idx] * current_coeffs[idx],
                            T_mat[:, idx],
                        )
                        - lambda_H
                    )
                    / U,
                    0,
                )
            ),
            0,
        )

        # Update the inner product
        current_in_prod += (new_coeff - current_coeffs[idx]) * T_mat[:, idx].reshape(
            current_in_prod.shape
        )
        current_coeffs = current_coeffs.at[idx].set(new_coeff)
        return (current_in_prod, current_coeffs), None

    # Update time-specific covariates
    if num_time_covariates > 0:
        (updated_in_prod, coeff_flat), _ = jax.lax.scan(
            update_coefficient,
            (updated_in_prod, coeff_flat),
            jnp.arange(num_time_covariates * coeff_rows),
        )

    # Update unit-specific covariates
    if num_unit_covariates > 0:
        (updated_in_prod, coeff_flat), _ = jax.lax.scan(
            update_coefficient,
            (updated_in_prod, coeff_flat),
            jnp.arange(num_time_covariates * coeff_rows, coeff_cols * coeff_rows),
        )

    # Reshape the updated coefficients back to a matrix
    updated_H_tilde = coeff_flat.reshape(coeff_rows, coeff_cols)

    return updated_H_tilde, updated_in_prod


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
    # Count the number of observed entries
    num_train = jnp.sum(W)

    # Compute the current predicted outcomes

    Y_hat = compute_Y_hat(
        L, X_tilde, Z_tilde, V, H_tilde, unit_fe, time_fe, beta, use_unit_fe, use_time_fe
    )

    # Calculate the residuals (observed - predicted)
    residuals = Y - Y_hat

    # Apply the observation mask to the residuals
    masked_residuals = mask_observed(residuals, W)

    # Add the current L to the masked residuals
    svd_input = masked_residuals + L

    # Perform Singular Value Decomposition (SVD)
    U, singular_values, Vt = jnp.linalg.svd(svd_input, full_matrices=False)
    V = Vt.T

    # Compute the singular value thresholding parameter
    svt_threshold = 0.25 * lambda_L * num_train

    # Apply singular value thresholding to update L
    L_updated = svt(U, V, singular_values, svt_threshold)

    return L_updated, singular_values


@jit
def fit(
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
    gamma: Array,
    delta: Array,
    beta: Array,
    lambda_L: Scalar,
    lambda_H: Scalar,
    use_unit_fe: bool,
    use_time_fe: bool,
    Omega_inv: Optional[Array] = None,
    niter: int = 1000,
    rel_tol: float = 1e-5,
    verbose: bool = False,
    print_iters: bool = False,
) -> Tuple[Array, Array, Array, Array, Array, Array, Scalar]:
    """
    Perform cyclic coordinate descent to estimate the MC-NNM model parameters.

    This function implements the core estimation algorithm for the Matrix Completion with Nuclear Norm Minimization \
    (MC-NNM) model. It uses a cyclic coordinate descent approach to iteratively update the model parameters, including \
    the low-rank matrix L, the covariate coefficients matrix H_tilde, fixed effects vectors (gamma and delta), \
    and unit-time-specific covariate coefficients (beta).

    The algorithm minimizes an objective function that balances the fit to the observed data with regularization terms.\
    It uses nuclear norm regularization for the low-rank matrix L and element-wise L1 regularization for the covariate \
    coefficients matrix H_tilde. The function handles both unit and time fixed effects, as well as various types of \
    covariates (unit-specific, time-specific, and unit-time-specific).

    The estimation process continues until convergence is reached (determined by the relative change in the objective \
    function) or the maximum number of iterations is exceeded. The function can optionally provide verbose output to \
    track the optimization progress.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X_tilde (Array): The augmented unit-specific covariates matrix of shape (N, P+N).
        Z_tilde (Array): The augmented time-specific covariates matrix of shape (T, Q+T).
        V (Array): The unit-time-specific covariates tensor of shape (N, T, J).
        H_tilde (Array): The initial covariate coefficients matrix of shape (P+N, Q+T).
        T_mat (Array): The precomputed matrix T of shape (N * T, (P+N) * (Q+T)).
        in_prod (Array): The inner product vector of shape (N * T,).
        in_prod_T (Array): The inner product vector of T of shape ((P+N) * (Q+T),).
        W (Array): The mask matrix indicating observed entries of shape (N, T).
        L (Array): The initial low-rank matrix of shape (N, T).
        gamma (Array): The initial unit fixed effects vector of shape (N,).
        delta (Array): The initial time fixed effects vector of shape (T,).
        beta (Array): The initial unit-time-specific covariate coefficients vector of shape (J,).
        lambda_L (Scalar): The regularization parameter for the nuclear norm of L.
        lambda_H (Scalar): The regularization parameter for the element-wise L1 norm of H_tilde.
        use_unit_fe (bool): Whether to include unit fixed effects in the decomposition.
        use_time_fe (bool): Whether to include time fixed effects in the decomposition.
        Omega_inv (Optional[Array]): The inverse of the omega matrix of shape (T, T). If None, the identity matrix is
        niter (int, optional): The maximum number of iterations for the coordinate descent algorithm. Default is 1000.
        rel_tol (float, optional): The relative tolerance for convergence. Default is 1e-5.
        verbose (bool, optional): Whether to print the objective value at each iteration. Default is False.
        print_iters (bool, optional): Whether to print in each iteration. Default is False.

    Returns:
        Tuple[Array, Array, Array, Array, Array, Array]:
        A tuple containing:
            - The updated covariate coefficient matrix H_tilde
            - The updated low-rank matrix L
            - The updated unit fixed effects vector
            - The updated time fixed effects vector
            - The updated unit-time-specific covariate vector
            - The updated in_prod vector
            - The final objective value

    Note:
    This function is computationally intensive and forms the core of the MC-NNM estimation procedure. It is designed to\
    handle large matrices efficiently, leveraging JAX for potential GPU acceleration. The convergence speed and final\
    estimates can be sensitive to the choice of regularization parameters (lambda_L and lambda_H) and the initial \
    values of the matrices and vectors.
    """
    obj_val = jnp.inf

    _, singular_values, _ = jnp.linalg.svd(L, full_matrices=False)
    sum_sigma = jnp.sum(singular_values)

    obj_val = compute_objective_value(  # type: ignore[assignment]
        Y,
        X_tilde,
        Z_tilde,
        V,
        H_tilde,
        W,
        L,
        gamma,
        delta,
        beta,
        sum_sigma,
        lambda_L,
        lambda_H,
        use_unit_fe,
        use_time_fe,
        inv_omega=Omega_inv,
    )

    def cond_fun(carry):
        obj_val, prev_obj_val, *_ = carry
        # rel_error = (obj_val - prev_obj_val) / (jnp.abs(prev_obj_val))
        rel_error = jnp.where(
            jnp.isfinite(obj_val),
            (obj_val - prev_obj_val) / (jnp.abs(prev_obj_val) + 1e-8),
            jnp.inf,
        )
        return lax.cond(
            (rel_error < rel_tol)
            & (rel_error > -0.5),  # Allow for slightly negative relative error
            lambda _: False,
            lambda _: carry[-1] < niter,
            None,
        )

    def body_fun(carry):
        obj_val, prev_obj_val, gamma, delta, beta, L, H_tilde, in_prod, i = carry
        gamma = update_unit_fe(Y, X_tilde, Z_tilde, H_tilde, W, L, delta, use_unit_fe)
        delta = update_time_fe(Y, X_tilde, Z_tilde, H_tilde, W, L, gamma, use_time_fe)
        beta = update_beta(Y, X_tilde, Z_tilde, V, H_tilde, W, L, gamma, delta)
        H_tilde, in_prod = update_H(
            Y,
            X_tilde,
            Z_tilde,
            V,
            H_tilde,
            T_mat,
            in_prod,
            in_prod_T,
            W,
            L,
            gamma,
            delta,
            beta,
            lambda_H,
            use_unit_fe,
            use_time_fe,
        )

        L, singular_values = update_L(
            Y,
            X_tilde,
            Z_tilde,
            V,
            H_tilde,
            W,
            L,
            gamma,
            delta,
            beta,
            lambda_L,
            use_unit_fe,
            use_time_fe,
        )

        sum_sigma = jnp.sum(singular_values)

        new_obj_val = compute_objective_value(
            Y,
            X_tilde,
            Z_tilde,
            V,
            H_tilde,
            W,
            L,
            gamma,
            delta,
            beta,
            sum_sigma,
            lambda_L,
            lambda_H,
            use_unit_fe,
            use_time_fe,
            inv_omega=Omega_inv,
        )

        lax.cond(
            print_iters,
            lambda _: jax.debug.print("Iteration {i}: {ov}", i=i, ov=new_obj_val),
            lambda _: None,
            operand=None,
        )
        return new_obj_val, obj_val, gamma, delta, beta, L, H_tilde, in_prod, i + 1

    init_val = (
        2 * obj_val,
        obj_val,
        gamma,
        delta,
        beta,
        L,
        H_tilde,
        in_prod,
        0,
    )  # TODO: improve initialization
    obj_val, _, gamma, delta, beta, L, H, in_prod, term_iter = lax.while_loop(
        cond_fun, body_fun, init_val
    )

    # lax.cond(
    #     term_iter == niter,
    #     lambda _: jax.debug.print("WARNING: Did not converge"),
    #     lambda _: None,
    #     None,
    # )

    lax.cond(
        verbose,
        lambda _: jax.debug.print(
            "Terminated at iteration {term_iter}: for lambda_L= {lam_L}, lambda_H= {lam_H}, "
            "objective function value= {obj_val}",
            term_iter=term_iter,
            lam_L=lambda_L,
            lam_H=lambda_H,
            obj_val=obj_val,
        ),
        lambda _: None,
        operand=None,
    )

    return H, L, gamma, delta, beta, in_prod, obj_val
