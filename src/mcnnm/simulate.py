import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


def generate_data_factor(
        nobs: int = 500,
        nperiods: int = 100,
        treated_period: int = 50,
        rank: int = 5,
        treatment_effect: float = 1.0,
        fixed_effects_scale: float = 0.1,
        covariates_scale: float = 0.1,
        noise_scale: float = 0.1,
        seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic data for testing the MC-NNM model.

    Args:
        nobs: Number of observations (units).
        nperiods: Number of time periods.
        treated_period: The period when treatment starts.
        rank: The rank of the low-rank matrix L.
        treatment_effect: The true treatment effect.
        fixed_effects_scale: The scale of the fixed effects.
        covariates_scale: The scale of the covariates and their coefficients.
        noise_scale: The scale of the noise.
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
    unit_fe = np.random.normal(0, fixed_effects_scale, nobs)
    time_fe = np.random.normal(0, fixed_effects_scale, nperiods)

    # Generate covariates and their coefficients
    X = np.random.normal(0, covariates_scale, (nobs, 2))
    X_coef = np.random.normal(0, covariates_scale, 2)
    Z = np.random.normal(0, covariates_scale, (nperiods, 2))
    Z_coef = np.random.normal(0, covariates_scale, 2)
    V = np.random.normal(0, covariates_scale, (nobs, nperiods, 2))
    V_coef = np.random.normal(0, covariates_scale, 2)

    # Generate outcome
    Y = (L +
         np.outer(unit_fe, np.ones(nperiods)) +
         np.outer(np.ones(nobs), time_fe) +
         np.repeat(X @ X_coef, nperiods).reshape(nobs, nperiods) +
         np.tile((Z @ Z_coef).reshape(1, -1), (nobs, 1)) +
         np.sum(V * V_coef, axis=2) +
         np.random.normal(0, noise_scale, (nobs, nperiods)))

    # Add treatment effect
    treat = np.zeros((nobs, nperiods))
    treat[:, treated_period:] = 1
    Y += treat * treatment_effect

    data['y'] = Y.flatten()
    data['treat'] = treat.flatten()

    true_params = {
        'L': L,
        'unit_fe': unit_fe,
        'time_fe': time_fe,
        'X': X,
        'X_coef': X_coef,
        'Z': Z,
        'Z_coef': Z_coef,
        'V': V,
        'V_coef': V_coef,
        'treatment_effect': treatment_effect,
        'noise_scale': noise_scale
    }

    return data, true_params