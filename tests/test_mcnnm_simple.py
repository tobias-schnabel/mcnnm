import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Optional, Tuple, Dict
from mcnnm.main import fit, MCNNMResults

def generate_data(
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

    # print(f"Proportion of treated observations: {data['treat'].mean()}")

    return data, true_params


def test_mcnnm_accuracy():
    nobs, nperiods = 500, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)

    Y = data.pivot(index='unit', columns='period', values='y').values
    W = data.pivot(index='unit', columns='period', values='treat').values

    X, Z, V = true_params['X'], true_params['Z'], true_params['V']

    print(f"\nProportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {np.mean(Y)}")
    print(f"Std of Y: {np.std(Y)}")

    results = fit(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True, verbose=True)

    # Print all results except the L matrix
    print("\nResults:")
    for field in results._fields:
        if field != 'L':
            value = getattr(results, field)
            if value is not None:
                if isinstance(value, (float, int)):
                    print(f"{field}: {value:.4f}")
                elif isinstance(value, (np.ndarray, jnp.ndarray)):
                    print(f"{field}: shape {value.shape}, mean {np.mean(value):.4f}, std {np.std(value):.4f}")
                else:
                    print(f"{field}: {value}")

    print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
    print(f"Chosen lambda_L: {results.lambda_L:.4f}")
    print(f"Chosen lambda_H: {results.lambda_H:.4f}")

    print(f"Unit fixed effects correlation: {np.corrcoef(results.gamma, true_params['unit_fe'])[0, 1]:.4f}")
    print(f"Time fixed effects correlation: {np.corrcoef(results.delta, true_params['time_fe'])[0, 1]:.4f}")

    print(f"True X coefficients: {true_params['X_coef']}")
    print(f"Estimated X coefficients: {results.H[:X.shape[1], :Z.shape[1]]}")
    print(f"True Z coefficients: {true_params['Z_coef']}")
    print(f"Estimated Z coefficients: {results.H[:X.shape[1], :Z.shape[1]].T}")
    print(f"True V coefficients: {true_params['V_coef']}")
    print(f"Estimated V coefficients: {results.beta}")

    mse = np.mean((Y - results.Y_completed)**2)
    print(f"Mean Squared Error of completed matrix: {mse:.4f}")

    treated_mean = np.mean(Y[W == 1])
    control_mean = np.mean(Y[W == 0])
    print(f"Simple diff-in-means estimate: {treated_mean - control_mean:.4f}")

def test_mcnnm_accuracy_no_covariates():
    nobs, nperiods = 500, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42,
                                      covariates_scale=0.0)  # Set covariates_scale to 0 to effectively remove covariates

    Y = data.pivot(index='unit', columns='period', values='y').values
    W = data.pivot(index='unit', columns='period', values='treat').values

    print(f"\nProportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {np.mean(Y)}")
    print(f"Std of Y: {np.std(Y)}")

    results = fit(Y, W, return_fixed_effects=True, verbose=True)

    # Print all results except the L matrix
    print("\nResults:")
    for field in results._fields:
        if field != 'L':
            value = getattr(results, field)
            if value is not None:
                if isinstance(value, (float, int)):
                    print(f"{field}: {value:.4f}")
                elif isinstance(value, (np.ndarray, jnp.ndarray)):
                    print(f"{field}: shape {value.shape}, mean {np.mean(value):.4f}, std {np.std(value):.4f}")
                else:
                    print(f"{field}: {value}")

    print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
    print(f"Chosen lambda_L: {results.lambda_L:.4f}")

    print(f"Unit fixed effects correlation: {np.corrcoef(results.gamma, true_params['unit_fe'])[0, 1]:.4f}")
    print(f"Time fixed effects correlation: {np.corrcoef(results.delta, true_params['time_fe'])[0, 1]:.4f}")

    mse = np.mean((Y - results.Y_completed)**2)
    print(f"Mean Squared Error of completed matrix: {mse:.4f}")

    treated_mean = np.mean(Y[W == 1])
    control_mean = np.mean(Y[W == 0])
    print(f"Simple diff-in-means estimate: {treated_mean - control_mean:.4f}")

    # Compare the estimated L with the true L
    L_correlation = np.corrcoef(results.L.flatten(), true_params['L'].flatten())[0, 1]
    print(f"Correlation between true and estimated L: {L_correlation:.4f}")

    # Compute and print the rank of the estimated L
    _, s, _ = np.linalg.svd(results.L)
    estimated_rank = np.sum(s > 1e-10)  # Count singular values above a small threshold
    print(f"Estimated rank of L: {estimated_rank}")
    print(f"True rank of L: {true_params['L'].shape[1]}")

if __name__ == "__main__":
    test_mcnnm_accuracy()