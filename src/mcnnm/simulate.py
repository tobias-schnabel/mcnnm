import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Literal


# def generate_data(
#         nobs: int = 500,
#         nperiods: int = 100,
#         treatment_probability: float = 0.5,
#         rank: int = 5,
#         treatment_effect: float = 1.0,
#         unit_fe: bool = True,
#         time_fe: bool = True,
#         X_cov: bool = True,
#         Z_cov: bool = True,
#         V_cov: bool = True,
#         fixed_effects_scale: float = 0.1,
#         covariates_scale: float = 0.1,
#         noise_scale: float = 0.1,
#         seed: Optional[int] = None
# ) -> Tuple[pd.DataFrame, Dict]:
#     """
#     Generate synthetic data for testing the MC-NNM model.
#
#     Args:
#         nobs: Number of observations (units).
#         nperiods: Number of time periods.
#         treatment_probability: The probability of a unit being treated.
#         rank: The rank of the low-rank matrix L.
#         treatment_effect: The true treatment effect.
#         unit_fe: Whether to include unit fixed effects.
#         time_fe: Whether to include time fixed effects.
#         X_cov: Whether to include unit-specific covariates.
#         Z_cov: Whether to include time-specific covariates.
#         V_cov: Whether to include unit-time specific covariates.
#         fixed_effects_scale: The scale of the fixed effects.
#         covariates_scale: The scale of the covariates and their coefficients.
#         noise_scale: The scale of the noise.
#         seed: Random seed for reproducibility.
#
#     Returns:
#         A tuple containing:
#         - pandas DataFrame with the generated data
#         - Dictionary with true parameter values
#     """
#     if seed is not None:
#         np.random.seed(seed)
#
#     # Generate basic structure
#     unit = np.arange(1, nobs + 1)
#     period = np.arange(1, nperiods + 1)
#     data = pd.DataFrame([(u, t) for u in unit for t in period], columns=['unit', 'period'])
#
#     # Generate low-rank matrix L
#     U = np.random.normal(0, 1, (nobs, rank))
#     V = np.random.normal(0, 1, (nperiods, rank))
#     L = U @ V.T
#
#     # Generate fixed effects
#     unit_fe_values = np.random.normal(0, fixed_effects_scale, nobs) if unit_fe else np.zeros(nobs)
#     time_fe_values = np.random.normal(0, fixed_effects_scale, nperiods) if time_fe else np.zeros(nperiods)
#
#     # Generate covariates and their coefficients
#     X = np.random.normal(0, covariates_scale, (nobs, 2)) if X_cov else np.zeros((nobs, 0))
#     X_coef = np.random.normal(0, covariates_scale, 2) if X_cov else np.array([])
#     Z = np.random.normal(0, covariates_scale, (nperiods, 2)) if Z_cov else np.zeros((nperiods, 0))
#     Z_coef = np.random.normal(0, covariates_scale, 2) if Z_cov else np.array([])
#     V = np.random.normal(0, covariates_scale, (nobs, nperiods, 2)) if V_cov else np.zeros((nobs, nperiods, 0))
#     V_coef = np.random.normal(0, covariates_scale, 2) if V_cov else np.array([])
#
#     # Generate outcome
#     Y = (L +
#          np.outer(unit_fe_values, np.ones(nperiods)) +
#          np.outer(np.ones(nobs), time_fe_values) +
#          np.repeat(X @ X_coef, nperiods).reshape(nobs, nperiods) +
#          np.tile((Z @ Z_coef).reshape(1, -1), (nobs, 1)) +
#          np.sum(V * V_coef, axis=2) +
#          np.random.normal(0, noise_scale, (nobs, nperiods)))
#
#     # Generate treatment assignment
#     treat = np.random.binomial(1, treatment_probability, (nobs, nperiods))
#     Y += treat * treatment_effect
#
#     data['y'] = Y.flatten()
#     data['treat'] = treat.flatten()
#
#     true_params = {
#         'L': L,
#         'unit_fe': unit_fe_values,
#         'time_fe': time_fe_values,
#         'X': X,
#         'X_coef': X_coef,
#         'Z': Z,
#         'Z_coef': Z_coef,
#         'V': V,
#         'V_coef': V_coef,
#         'treatment_effect': treatment_effect,
#         'noise_scale': noise_scale
#     }
#
#     return data, true_params

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
        assignment_mechanism: Literal['staggered', 'block', 'single_treated_period', 'single_treated_unit', 'last_periods'] = 'staggered',
        treated_fraction: float = 0.2,
        last_treated_periods: int = 10,
        seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic data for testing the MC-NNM model with various treatment assignment mechanisms.

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
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - pandas DataFrame with the generated data
        - Dictionary with true parameter values
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate basic structure
    unit = np.arange(1, nobs + 1)
    period = np.arange(1, nperiods + 1)
    data = pd.DataFrame([(u, t) for u in unit for t in period], columns=['unit', 'period'])

    # Generate low-rank matrix L
    U = np.random.normal(0, 1, (nobs, rank))
    V = np.random.normal(0, 1, (nperiods, rank))
    L = U @ V.T

    # Generate fixed effects
    unit_fe_values = np.random.normal(0, fixed_effects_scale, nobs) if unit_fe else np.zeros(nobs)
    time_fe_values = np.random.normal(0, fixed_effects_scale, nperiods) if time_fe else np.zeros(nperiods)

    # Generate covariates and their coefficients
    X = np.random.normal(0, covariates_scale, (nobs, 2)) if X_cov else np.zeros((nobs, 0))
    X_coef = np.random.normal(0, covariates_scale, 2) if X_cov else np.array([])
    Z = np.random.normal(0, covariates_scale, (nperiods, 2)) if Z_cov else np.zeros((nperiods, 0))
    Z_coef = np.random.normal(0, covariates_scale, 2) if Z_cov else np.array([])
    V = np.random.normal(0, covariates_scale, (nobs, nperiods, 2)) if V_cov else np.zeros((nobs, nperiods, 0))
    V_coef = np.random.normal(0, covariates_scale, 2) if V_cov else np.array([])

    # Generate outcome
    Y = (L +
         np.outer(unit_fe_values, np.ones(nperiods)) +
         np.outer(np.ones(nobs), time_fe_values) +
         np.repeat(X @ X_coef, nperiods).reshape(nobs, nperiods) +
         np.tile((Z @ Z_coef).reshape(1, -1), (nobs, 1)) +
         np.sum(V * V_coef, axis=2) +
         np.random.normal(0, noise_scale, (nobs, nperiods)))

    # Generate treatment assignment based on the specified mechanism
    if assignment_mechanism == 'staggered':
        treat = np.zeros((nobs, nperiods), dtype=int)
        adoption_times = np.random.geometric(p=treatment_probability, size=nobs)
        for i in range(nobs):
            if adoption_times[i] <= nperiods:
                treat[i, adoption_times[i]-1:] = 1
    elif assignment_mechanism == 'block':
        treated_units = np.random.choice(nobs, size=int(nobs * treated_fraction), replace=False)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_units, nperiods // 2:] = 1
    elif assignment_mechanism == 'single_treated_period':
        treated_units = np.random.choice(nobs, size=int(nobs * treated_fraction), replace=False)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_units, -1] = 1
    elif assignment_mechanism == 'single_treated_unit':
        treated_unit = np.random.choice(nobs)
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[treated_unit, nperiods // 2:] = 1
    elif assignment_mechanism == 'last_periods':
        treat = np.zeros((nobs, nperiods), dtype=int)
        treat[:, -last_treated_periods:] = 1
    else:
        raise ValueError("Invalid assignment mechanism specified.")

    Y += treat * treatment_effect

    data['y'] = Y.flatten()
    data['treat'] = treat.flatten()

    true_params = {
        'L': L,
        'unit_fe': unit_fe_values,
        'time_fe': time_fe_values,
        'X': X,
        'X_coef': X_coef,
        'Z': Z,
        'Z_coef': Z_coef,
        'V': V,
        'V_coef': V_coef,
        'treatment_effect': treatment_effect,
        'noise_scale': noise_scale,
        'assignment_mechanism': assignment_mechanism
    }

    return data, true_params