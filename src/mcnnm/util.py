from datetime import datetime
from typing import Optional, Tuple, Dict, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.numpy.linalg import norm

from .types import Array, Scalar


def p_o(A: Array, mask: Array) -> Array:
    """
    Projects the matrix A onto the observed entries specified by the binary mask.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        Array: The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.
    """
    if A.shape != mask.shape:
        raise ValueError("Shapes of A and mask must match.")
    return jnp.where(mask, A, jnp.zeros_like(A))


def p_perp_o(A: Array, mask: Array) -> Array:
    """
    Projects the matrix A onto the unobserved entries specified by the binary mask.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        Array: The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.
    """
    if A.shape != mask.shape:
        raise ValueError("Shapes of A and mask must match.")
    return jnp.where(mask, jnp.zeros_like(A), A)


def shrink_lambda(A: Array, lambda_: Scalar) -> Array:
    """
    Applies the soft-thresholding operator to the singular values of a matrix A.

    Args:
        A: The input matrix.
        lambda_: The shrinkage parameter.

    Returns:
        Array: The matrix with soft-thresholded singular values.
    """
    u, s, vt = jnp.linalg.svd(A, full_matrices=False)
    s_shrunk = jnp.maximum(s - lambda_, 0)
    return u @ jnp.diag(s_shrunk) @ vt


def frobenius_norm(A: Array) -> Scalar:
    """
    Computes the Frobenius norm of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        Scalar: The Frobenius norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return norm(A, ord="fro")


def nuclear_norm(A: Array) -> Scalar:
    """
    Computes the nuclear norm (sum of singular values) of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        Scalar: The nuclear norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    _, s, _ = jnp.linalg.svd(A, full_matrices=False)
    return jnp.sum(s)


def element_wise_l1_norm(A: Array) -> Scalar:
    """
    Computes the element-wise L1 norm of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        Scalar: The element-wise L1 norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return jnp.sum(jnp.abs(A))


def propose_lambda(proposed_lambda: Optional[Scalar] = None, n_lambdas: int = 6) -> Array:
    """
    Creates a log-spaced list of proposed lambda values around a given value.

    Args:
        proposed_lambda: The proposed lambda value. If None, the default sequence is used.
        n_lambdas: The number of lambda values to generate.

    Returns:
        Array: The sequence of proposed lambda values.
    """

    def generate_sequence(log_min: Scalar, log_max: Scalar) -> Array:
        return jnp.logspace(log_min, log_max, n_lambdas)

    def default_sequence(_):
        return generate_sequence(-3, 0)

    def custom_sequence(lambda_val):
        log_lambda = jnp.log10(jnp.maximum(lambda_val, 1e-10))
        return generate_sequence(log_lambda - 2, log_lambda + 2)

    return jax.lax.cond(
        proposed_lambda is None,
        default_sequence,
        custom_sequence,
        operand=jnp.array(proposed_lambda if proposed_lambda is not None else 0.0),
    )


def initialize_params(
    Y: Array, X: Array, Z: Array, V: Array
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Initialize parameters for the MC-NNM model.

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
    H = jnp.zeros((X.shape[1] + N, Z.shape[1] + T))
    gamma = jnp.zeros(N)
    delta = jnp.zeros(T)

    # Calculate beta_shape as a concrete value
    beta_shape = max(V.shape[2], 1)
    beta = jnp.zeros((beta_shape,))

    # Use JAX conditional to initialize beta correctly
    beta = jax.lax.cond(
        V.shape[2] > 0, lambda _: beta, lambda _: jnp.zeros_like(beta), operand=None
    )

    return L, H, gamma, delta, beta


def check_inputs(
    Y: Array,
    W: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array]:
    """
    Check and preprocess input arrays for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Optional[Array]): The unit-specific covariates matrix of shape (N, P). Default is None.
        Z (Optional[Array]): The time-specific covariates matrix of shape (T, Q). Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor of shape (N, T, J). Default is None.
        Omega (Optional[Array]): The autocorrelation matrix of shape (T, T). Default is None.

    Returns:
        Tuple: A tuple containing preprocessed X, Z, V, and Omega arrays.

    Raises:
        ValueError: If the shape of W does not match the shape of Y.
    """
    N, T = Y.shape
    if W.shape != (N, T):
        raise ValueError("The shape of W must match the shape of Y.")
    X = jnp.zeros((N, 0)) if X is None else X
    Z = jnp.zeros((T, 0)) if Z is None else Z
    V = (
        jnp.zeros((N, T, 1)) if V is None else V
    )  # Add a dummy dimension if V is None, otherwise causes issues
    Omega = jnp.eye(T) if Omega is None else Omega
    return X, Z, V, Omega


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


def generate_data(
    nobs: int = 500,
    nperiods: int = 100,
    treatment_probability: float = 0.5,
    rank: int = 5,
    treatment_effect: float = 1.0,
    unit_fe: bool = True,
    time_fe: bool = True,
    X_cov: bool = True,
    Z_cov: bool = True,
    V_cov: bool = True,
    fixed_effects_scale: float = 0.1,
    covariates_scale: float = 0.1,
    noise_scale: float = 0.1,
    assignment_mechanism: Literal[
        "staggered", "block", "single_treated_period", "single_treated_unit", "last_periods"
    ] = "staggered",
    treated_fraction: float = 0.2,
    last_treated_periods: int = 10,
    autocorrelation: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic data for testing the MC-NNM model with various treatment assignment mechanisms and
    autocorrelated errors.

    Args:
        nobs: Number of observations (units).
        nperiods: Number of time periods.
        treatment_probability: The probability of a unit being treated (for staggered adoption).
        rank: The rank of the low-rank matrix L.
        treatment_effect: The true treatment effect.
        unit_fe: Whether to include unit fixed effects.
        time_fe: Whether to include time fixed effects.
        X_cov: Whether to include unit-specific covariates.
        Z_cov: Whether to include time-specific covariates.
        V_cov: Whether to include unit-time specific covariates.
        fixed_effects_scale: The scale of the fixed effects.
        covariates_scale: The scale of the covariates and their coefficients.
        noise_scale: The scale of the noise.
        assignment_mechanism: The treatment assignment mechanism to use.
            - 'staggered': Staggered adoption (default)
            - 'block': Block structure
            - 'single_treated_period': Single treated period
            - 'single_treated_unit': Single treated unit
            - 'last_periods': All units treated for the last few periods
        treated_fraction: Fraction of units to be treated (for block and single_treated_period).
        last_treated_periods: Number of periods to treat all units at the end (for last_periods mechanism).
        autocorrelation: The autocorrelation coefficient for the error term (0 <= autocorrelation < 1).
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - pandas DataFrame with the generated data
        - Dictionary with true parameter values
    """
    if seed is not None:
        np.random.seed(seed)

    if not 0 <= autocorrelation < 1:
        raise ValueError("Autocorrelation must be between 0 and 1 (exclusive).")

    # Generate basic structure
    unit = np.arange(1, nobs + 1)
    period = np.arange(1, nperiods + 1)
    data = pd.DataFrame([(u, t) for u in unit for t in period], columns=["unit", "period"])

    # Generate low-rank matrix L
    U = np.random.normal(0, 1, (nobs, rank))
    V = np.random.normal(0, 1, (nperiods, rank))
    L = U @ V.T

    # Generate fixed effects
    unit_fe_values = np.random.normal(0, fixed_effects_scale, nobs) if unit_fe else np.zeros(nobs)
    time_fe_values = (
        np.random.normal(0, fixed_effects_scale, nperiods) if time_fe else np.zeros(nperiods)
    )

    # Generate covariates and their coefficients
    X = np.random.normal(0, covariates_scale, (nobs, 2)) if X_cov else np.zeros((nobs, 0))
    X_coef = np.random.normal(0, covariates_scale, 2) if X_cov else np.array([])
    Z = np.random.normal(0, covariates_scale, (nperiods, 2)) if Z_cov else np.zeros((nperiods, 0))
    Z_coef = np.random.normal(0, covariates_scale, 2) if Z_cov else np.array([])
    V = (
        np.random.normal(0, covariates_scale, (nobs, nperiods, 2))
        if V_cov
        else np.zeros((nobs, nperiods, 0))
    )
    V_coef = np.random.normal(0, covariates_scale, 2) if V_cov else np.array([])

    # Generate autocorrelated errors
    errors = np.zeros((nobs, nperiods))
    for i in range(nobs):
        errors[i, 0] = np.random.normal(0, noise_scale)
        for t in range(1, nperiods):
            errors[i, t] = autocorrelation * errors[i, t - 1] + np.random.normal(
                0, noise_scale * np.sqrt(1 - autocorrelation**2)
            )

    # Generate outcome
    Y = (
        L
        + np.outer(unit_fe_values, np.ones(nperiods))
        + np.outer(np.ones(nobs), time_fe_values)
        + np.repeat(X @ X_coef, nperiods).reshape(nobs, nperiods)
        + np.tile((Z @ Z_coef).reshape(1, -1), (nobs, 1))
        + np.sum(V * V_coef, axis=2)
        + errors
    )

    # Generate treatment assignment based on the specified mechanism
    if assignment_mechanism == "staggered":
        treat = np.zeros((nobs, nperiods), dtype=int)
        adoption_times = np.random.geometric(p=treatment_probability, size=nobs)
        for i in range(nobs):
            if adoption_times[i] <= nperiods:
                treat[i, adoption_times[i] - 1 :] = 1
    elif assignment_mechanism == "block":
        treated_units = np.random.choice(nobs, size=int(nobs * treated_fraction), replace=False)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_units, nperiods // 2 :] = 1
    elif assignment_mechanism == "single_treated_period":
        treated_units = np.random.choice(nobs, size=int(nobs * treated_fraction), replace=False)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_units, -1] = 1
    elif assignment_mechanism == "single_treated_unit":
        treated_unit = np.random.choice(nobs)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_unit, nperiods // 2 :] = 1
    elif assignment_mechanism == "last_periods":
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[:, -last_treated_periods:] = 1
    else:
        raise ValueError("Invalid assignment mechanism specified.")

    Y += treat * treatment_effect

    data["y"] = Y.flatten()
    data["treat"] = treat.flatten()

    true_params = {
        "L": L,
        "unit_fe": unit_fe_values,
        "time_fe": time_fe_values,
        "X": X,
        "X_coef": X_coef,
        "Z": Z,
        "Z_coef": Z_coef,
        "V": V,
        "V_coef": V_coef,
        "treatment_effect": treatment_effect,
        "noise_scale": noise_scale,
        "assignment_mechanism": assignment_mechanism,
        "autocorrelation": autocorrelation,
    }

    return data, true_params


def print_with_timestamp(message: str) -> None:
    """
    Print a message with a human-readable timestamp (hhmmss) prefix.

    Args:
        message (str): The message to be printed.

    Returns:
        None
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
