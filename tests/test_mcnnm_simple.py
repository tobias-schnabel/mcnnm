import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple, Dict
from mcnnm.main import fit

def generate_data(
    nobs: int = 500,
    nperiods: int = 100,
    treated_period: int = 50,
    include_unit_fe: bool = True,
    include_time_fe: bool = True,
    include_unit_covariates: bool = True,
    include_time_covariates: bool = True,
    include_unit_time_covariates: bool = True,
    treatment_effect: float = 5.0,
    noise_std: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic data for testing the MC-NNM model.

    Args:
        nobs: Number of observations (units).
        nperiods: Number of time periods.
        treated_period: The period when treatment starts.
        include_unit_fe: Whether to include unit fixed effects.
        include_time_fe: Whether to include time fixed effects.
        include_unit_covariates: Whether to include unit-specific covariates.
        include_time_covariates: Whether to include time-specific covariates.
        include_unit_time_covariates: Whether to include unit-time specific covariates.
        treatment_effect: The true treatment effect.
        noise_std: Standard deviation of the noise.
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

    # Initialize outcome
    data['y'] = 0.0

    # Generate and add fixed effects
    true_params = {}
    if include_unit_fe:
        unit_fe = np.random.normal(0, 1, nobs)
        data['y'] += np.repeat(unit_fe, nperiods)
        true_params['unit_fe'] = unit_fe

    if include_time_fe:
        time_fe = np.random.normal(0, 1, nperiods)
        data['y'] += np.tile(time_fe, nobs)
        true_params['time_fe'] = time_fe

    # Generate and add covariates
    if include_unit_covariates:
        X = np.random.normal(0, 1, (nobs, 2))  # 2 unit-specific covariates
        X_coef = np.array([0.5, -0.5])
        data['y'] += np.repeat(X.dot(X_coef), nperiods)
        true_params['X'] = X
        true_params['X_coef'] = X_coef

    if include_time_covariates:
        Z = np.random.normal(0, 1, (nperiods, 2))  # 2 time-specific covariates
        Z_coef = np.array([0.3, -0.3])
        data['y'] += np.tile(Z.dot(Z_coef), nobs)
        true_params['Z'] = Z
        true_params['Z_coef'] = Z_coef

    if include_unit_time_covariates:
        V = np.random.normal(0, 1, (nobs, nperiods, 2))  # 2 unit-time specific covariates
        V_coef = np.array([0.2, -0.2])
        data['y'] += V.reshape(-1, 2).dot(V_coef)
        true_params['V'] = V
        true_params['V_coef'] = V_coef

    # Add treatment
    data['treat'] = (data['period'] > treated_period).astype(int)
    data['y'] += data['treat'] * treatment_effect
    true_params['treatment_effect'] = treatment_effect

    # Add noise
    data['y'] += np.random.normal(0, noise_std, nobs * nperiods)
    true_params['noise_std'] = noise_std

    print(f"Proportion of treated observations: {data['treat'].mean()}")

    return data, true_params

def test_mcnnm_accuracy():
    nobs, nperiods = 500, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)

    Y = data.pivot(index='unit', columns='period', values='y').values
    W = data.pivot(index='unit', columns='period', values='treat').values

    # Prepare covariates if they exist in the data
    X = true_params.get('X')
    Z = true_params.get('Z')
    V = true_params.get('V')

    print(f"Proportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {np.mean(Y)}")
    print(f"Std of Y: {np.std(Y)}")

    # Use fit function with cross-validation
    results = fit(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True)
    tau, lambda_L, L, Y_completed, gamma, delta, beta, H = results

    # Check treatment effect
    print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {tau}")
    print(f"Chosen lambda_L: {lambda_L}")

    # Compare true and estimated fixed effects
    if 'unit_fe' in true_params:
        print(f"Unit fixed effects correlation: {np.corrcoef(gamma, true_params['unit_fe'])[0, 1]}")
        print(f"True unit FE mean: {np.mean(true_params['unit_fe'])}, Estimated: {np.mean(gamma)}")
        print(f"True unit FE std: {np.std(true_params['unit_fe'])}, Estimated: {np.std(gamma)}")

    if 'time_fe' in true_params:
        print(f"Time fixed effects correlation: {np.corrcoef(delta, true_params['time_fe'])[0, 1]}")
        print(f"True time FE mean: {np.mean(true_params['time_fe'])}, Estimated: {np.mean(delta)}")
        print(f"True time FE std: {np.std(true_params['time_fe'])}, Estimated: {np.std(delta)}")

    # Compare true and estimated covariate coefficients
    if 'X_coef' in true_params:
        estimated_X_coef = H[:X.shape[1], :Z.shape[1]]
        print(f"True X coefficients: {true_params['X_coef']}, Estimated: {estimated_X_coef}")

    if 'Z_coef' in true_params:
        estimated_Z_coef = H[:X.shape[1], :Z.shape[1]].T
        print(f"True Z coefficients: {true_params['Z_coef']}, Estimated: {estimated_Z_coef}")

    if 'V_coef' in true_params:
        print(f"True V coefficients: {true_params['V_coef']}, Estimated: {beta}")

    # Check completed matrix
    mse = np.mean((Y - Y_completed)**2)
    print(f"Mean Squared Error of completed matrix: {mse}")

    # Simple difference in means
    treated_mean = np.mean(Y[W == 1])
    control_mean = np.mean(Y[W == 0])
    print(f"Simple diff-in-means estimate: {treated_mean - control_mean}")

    assert np.allclose(tau, true_params['treatment_effect'], atol=1e-1), \
        f"Estimated effect {tau} not close to true effect {true_params['treatment_effect']}"

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_mcnnm_accuracy()

