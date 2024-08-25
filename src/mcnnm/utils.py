from typing import Optional, Tuple, List, Literal, Dict
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .types import Array, Scalar

jax.config.update("jax_enable_x64", True)


def convert_inputs(
    Y: pd.DataFrame,
    W: pd.DataFrame,
    X: Optional[pd.DataFrame] = None,
    Z: Optional[pd.DataFrame] = None,
    V: Optional[List[pd.DataFrame]] = None,
    Omega: Optional[pd.DataFrame] = None,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
]:
    """
    Convert input DataFrames to JAX arrays for the MC-NNM model.
    """
    Y = jnp.array(Y.values)
    W = jnp.array(W.values)
    N, T = Y.shape

    X_arr: Optional[jnp.ndarray] = None
    Z_arr: Optional[jnp.ndarray] = None
    V_arr: Optional[jnp.ndarray] = None
    Omega_arr: Optional[jnp.ndarray] = None

    if X is not None:
        X_arr = jnp.array(X.values)
        if X_arr.shape[0] != N:
            raise ValueError(
                f"The first dimension of X ({X_arr.shape[0]}) must match the first dimension of Y ({N})."
            )

    if Z is not None:
        Z_arr = jnp.array(Z.values)
        if Z_arr.shape[0] != T:
            raise ValueError(
                f"The first dimension of Z ({Z_arr.shape[0]}) must match the second dimension of Y ({T})."
            )

    if V is not None:
        if len(V) == 0:
            raise ValueError("V cannot be an empty list.")
        V_list = [jnp.array(df.values) for df in V]
        V_arr = jnp.stack(V_list, axis=2)
        if V_arr.shape[:2] != (N, T):
            raise ValueError(f"The shape of V must match the shape of Y ({N}, {T}).")

    if Omega is not None:
        Omega_arr = jnp.array(Omega.values)
        if Omega_arr.shape != (T, T):
            raise ValueError(f"The shape of Omega ({Omega_arr.shape}) must be ({T}, {T}).")

    return Y, W, X_arr, Z_arr, V_arr, Omega_arr


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
        ValueError: If the input array dimensions are mismatched or invalid.
    """
    N, T = Y.shape

    if W.shape != (N, T):
        raise ValueError(f"The shape of W ({W.shape}) must match the shape of Y ({Y.shape}).")

    if X is not None:
        if X.shape[0] != N:
            raise ValueError(
                f"The first dimension of X ({X.shape[0]}) must match the first dimension of Y ({N})."
            )
    else:
        X = jnp.zeros((N, 0))

    if Z is not None:
        if Z.shape[0] != T:
            raise ValueError(
                f"The first dimension of Z ({Z.shape[0]}) must match the second dimension of Y ({T})."
            )
    else:
        Z = jnp.zeros((T, 0))

    if V is not None:
        if V.shape[:2] != (N, T):
            raise ValueError(
                f"The first two dimensions of V ({V.shape[:2]}) must match the shape of Y ({Y.shape})."
            )
    else:
        V = jnp.zeros((N, T, 1))  # Add a dummy dimension if V is None, otherwise causes issues

    if Omega is not None:
        if Omega.shape != (T, T):
            raise ValueError(f"The shape of Omega ({Omega.shape}) must be ({T}, {T}).")
    else:
        Omega = jnp.eye(T)

    return X, Z, V, Omega


def generate_data(
    nobs: int = 500,
    nperiods: int = 100,
    Y_mean: float = 10.0,
    treatment_probability: float = 0.5,
    rank: int = 5,
    treatment_effect: float = 5.0,
    unit_fe: bool = True,
    time_fe: bool = True,
    X_cov: bool = True,
    Z_cov: bool = True,
    V_cov: bool = True,
    fixed_effects_scale: float = 0.5,
    covariates_scale: float = 0.5,
    noise_scale: float = 1,
    assignment_mechanism: Literal[
        "staggered", "block", "single_treated_period", "single_treated_unit", "last_periods"
    ] = "last_periods",
    treated_fraction: float = 0.2,
    last_treated_periods: int = 2,
    autocorrelation: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Dict,
]:
    if seed is not None:
        np.random.seed(seed)

    if not 0 <= autocorrelation < 1:
        raise ValueError("Autocorrelation must be between 0 and 1 (exclusive).")

    # Generate low-rank matrix L
    U = np.random.normal(0, 1, (nobs, rank))
    V = np.random.normal(0, 1, (nperiods, rank))
    L = U @ V.T

    # Generate fixed effects
    unit_fe_values = np.random.normal(0, fixed_effects_scale, nobs) if unit_fe else np.zeros(nobs)
    time_fe_values = (
        np.random.normal(0, fixed_effects_scale, nperiods) if time_fe else np.zeros(nperiods)
    )

    # generate random offsets to y_mean
    X_mean = Y_mean - np.random.normal(2, 1)
    Z_mean = Y_mean + np.random.normal(2, 1)
    V_mean = Y_mean - np.random.normal(4, 1)

    # Generate covariates and their coefficients
    X = jnp.array(np.random.normal(X_mean, covariates_scale, (nobs, 2))) if X_cov else None
    X_coef = np.random.normal(0, covariates_scale, 2) if X_cov else np.array([])
    Z = jnp.array(np.random.normal(Z_mean, covariates_scale, (nperiods, 2))) if Z_cov else None
    Z_coef = np.random.normal(0, covariates_scale, 2) if Z_cov else np.array([])
    V = (
        jnp.array(np.random.normal(V_mean, covariates_scale, (nobs, nperiods, 2)))  # type: ignore
        if V_cov
        else None
    )  # type: ignore
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
        + Y_mean
        + np.outer(unit_fe_values, np.ones(nperiods))
        + np.outer(np.ones(nobs), time_fe_values)
        + (np.repeat(X @ X_coef, nperiods).reshape(nobs, nperiods) if X_cov else 0)  # type: ignore
        + (np.tile((Z @ Z_coef).reshape(1, -1), (nobs, 1)) if Z_cov else 0)  # type: ignore
        + (np.sum(V * V_coef, axis=2) if V_cov else 0)
        + errors
    )

    # Generate treatment assignment based on the specified mechanism
    if assignment_mechanism == "staggered":
        treat = np.zeros((nobs, nperiods), dtype=int)
        adoption_times = np.random.geometric(p=treatment_probability, size=nobs)
        for i in range(nobs):  # pragma: no  cover
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

    Y_0 = Y.copy()  # untreated potential outcome
    Y += treat * treatment_effect

    # Convert Y, W, and covariates to JAX arrays
    Y = jnp.array(Y)
    W = jnp.array(treat, dtype=jnp.float64)
    X = jnp.array(X) if X is not None else None
    Z = jnp.array(Z) if Z is not None else None
    V = jnp.array(V) if V is not None else None  # type: ignore[assignment]

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
        "Y(0)": jnp.array(Y_0),
    }

    return Y, W, X, Z, V, true_params  # type: ignore


def propose_lambda(
    max_lambda: Scalar, min_lambda: Optional[Scalar] = None, n_lambdas: int = 6
) -> Array:
    """
    Creates a log-spaced list of proposed lambda values between max_lambda and min_lambda.

    Args:
        max_lambda: The maximum lambda value.
        min_lambda: The minimum lambda value. If None, it is set to max_lambda - 3 (in log10 scale).
        n_lambdas: The number of lambda values to generate.

    Returns:
        Array: The sequence of proposed lambda values.

    Raises:
        ValueError: If max_lambda is smaller than the default minimum lambda value (1e-10).
    """
    min_log_lambda = jnp.log10(max_lambda) - 3 if min_lambda is None else jnp.log10(min_lambda)
    max_log_lambda = jnp.log10(max_lambda)

    if min_lambda and max_lambda < min_lambda:
        raise ValueError("max_lambda must be greater than or equal to min_lambda.")

    if max_log_lambda < -10:
        raise ValueError(
            f"max_lambda ({max_lambda}) is too small. It should be greater than or equal to 1e-10."
        )
    if n_lambdas < 2:
        raise ValueError("n_lambdas must be greater than or equal to 2.")
    # Ensure min_log_lambda is not smaller than a small positive value to avoid zero or negative lambdas
    min_log_lambda = jnp.maximum(min_log_lambda, -10)

    return jnp.logspace(min_log_lambda, max_log_lambda, n_lambdas)


def generate_lambda_grid(max_lambda_L, max_lambda_H, n_lambda):
    """
    Generates a grid of lambda values for the MC-NNM model.

    This function creates a 2D grid of lambda values by generating log-spaced sequences
    for both lambda_L and lambda_H, and then forming a meshgrid from these sequences.

    Args:
        max_lambda_L (float): The maximum lambda value for the L dimension.
        max_lambda_H (float): The maximum lambda value for the H dimension.
        n_lambda (int): The number of lambda values to generate for both dimensions.

    Returns:
        jnp.ndarray: A 2D array where each row represents a pair of lambda values (lambda_L, lambda_H).
    """
    lambda_L_values = propose_lambda(max_lambda_L, n_lambdas=n_lambda)
    lambda_H_values = propose_lambda(max_lambda_H, n_lambdas=n_lambda)

    lambda_grid = jnp.array(jnp.meshgrid(lambda_L_values, lambda_H_values)).T.reshape(-1, 2)
    return lambda_grid


def extract_shortest_path(lambda_grid):
    """
    Extracts the shortest path along the edges of a 2D lambda grid.

    This function traverses the edges of the given lambda grid starting from the top-left corner
    (highest lambda_L and lambda_H), moving down to the bottom-left corner (lowest lambda_L, highest lambda_H),
    and then right to the bottom-right corner (lowest lambda_L and lambda_H).

    Args:
        lambda_grid (jnp.ndarray): A 2D array representing the flattened lambda grid.

    Returns:
        jnp.ndarray: A 2D array containing the lambda pairs along the shortest path.
    """
    # Infer n_lambda_L and n_lambda_H from the grid shape
    total_points = lambda_grid.shape[0]
    n_lambda_L = int(jnp.sqrt(total_points))
    n_lambda_H = n_lambda_L  # Assuming a square grid

    shortest_path = []

    # Add the top edge of the grid (fixed highest lambda_L, decreasing lambda_H)
    for j in range(n_lambda_H - 1, -1, -1):
        shortest_path.append(lambda_grid[(n_lambda_L - 1) * n_lambda_H + j])

    # Add the left edge of the grid (decreasing lambda_L, fixed lowest lambda_H)
    for i in range(n_lambda_L - 2, -1, -1):
        shortest_path.append(lambda_grid[i * n_lambda_H])

    return jnp.array(shortest_path)
