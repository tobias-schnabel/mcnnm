from typing import Optional, Tuple, Dict, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .types import Array


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
    y_mean: float = 10.0,
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
        y_mean: The mean of the outcome variable.
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
        + y_mean
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

    Y_0 = Y.flatten()  # untreated potential outcome
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
        "Y(0)": Y_0,
    }

    return data, true_params
