import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Optional, Tuple, Dict
import pytest
from mcnnm.main import estimate, MCNNMResults
import jax


def assert_close(true_value, estimated_value, tolerance, message):
    assert np.abs(true_value - estimated_value) < tolerance, f"{message}: true={true_value:.4f}, estimated={estimated_value:.4f}, difference={np.abs(true_value - estimated_value):.4f}"


jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)


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


@pytest.mark.timeout(120)
def test_mcnnm_accuracy_no_covariates(tolerance=0.1):
    nobs, nperiods = 500, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42,
                                      covariates_scale=0.0)  # Set covariates_scale to 0 to effectively remove covariates

    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

    print(f"\nProportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {jnp.mean(Y)}")
    print(f"Std of Y: {jnp.std(Y)}")

    results = estimate(Y, W, return_fixed_effects=True)

    print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
    assert_close(true_params['treatment_effect'], results.tau, tolerance, "Estimated treatment effect")

    print(f"Chosen lambda_L: {results.lambda_L:.4f}")

    print("\nFixed Effects Comparison:")
    print("Unit Fixed Effects:")
    print(f"True mean:      {jnp.mean(true_params['unit_fe']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.gamma):.4f}")
    assert_close(jnp.mean(true_params['unit_fe']), jnp.mean(results.gamma), tolerance, "Estimated unit fixed effects mean")

    print("\nTime Fixed Effects:")
    print(f"True mean:      {jnp.mean(true_params['time_fe']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.delta):.4f}")
    assert_close(jnp.mean(true_params['time_fe']), jnp.mean(results.delta), tolerance, "Estimated time fixed effects mean")

    print(f"\nTrue L mean: {jnp.mean(true_params['L']):.4f}")
    print(f"Estimated L mean: {jnp.mean(results.L):.4f}")
    assert_close(jnp.mean(true_params['L']), jnp.mean(results.L), tolerance, "Estimated L mean")


@pytest.mark.timeout(600)
def test_mcnnm_accuracy(tolerance=0.2):
    nobs, nperiods = 500, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)

    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])

    print(f"\nProportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {jnp.mean(Y)}")
    print(f"Std of Y: {jnp.std(Y)}")

    results = estimate(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True)

    print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
    assert_close(true_params['treatment_effect'], results.tau, tolerance, "Estimated treatment effect")

    print(f"Chosen lambda_L: {results.lambda_L:.4f}")
    print(f"Chosen lambda_H: {results.lambda_H:.4f}")

    print("\nFixed Effects Comparison:")
    print("Unit Fixed Effects:")
    print(f"True mean:      {jnp.mean(true_params['unit_fe']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.gamma):.4f}")
    assert_close(jnp.mean(true_params['unit_fe']), jnp.mean(results.gamma), tolerance, "Estimated unit fixed effects mean")

    print("\nTime Fixed Effects:")
    print(f"True mean:      {jnp.mean(true_params['time_fe']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.delta):.4f}")
    assert_close(jnp.mean(true_params['time_fe']), jnp.mean(results.delta), tolerance, "Estimated time fixed effects mean")

    print("\nCovariate Coefficients Comparison:")
    print("X Coefficients:")
    print(f"True mean:      {jnp.mean(true_params['X_coef']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.H[:X.shape[1], :Z.shape[1]]):.4f}")
    assert_close(jnp.mean(true_params['X_coef']), jnp.mean(results.H[:X.shape[1], :Z.shape[1]]), tolerance, "Estimated X coefficients mean")

    print("\nZ Coefficients:")
    print(f"True mean:      {jnp.mean(true_params['Z_coef']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.H[:X.shape[1], :Z.shape[1]].T):.4f}")
    assert_close(jnp.mean(true_params['Z_coef']), jnp.mean(results.H[:X.shape[1], :Z.shape[1]].T), tolerance, "Estimated Z coefficients mean")

    print("\nV Coefficients:")
    print(f"True mean:      {jnp.mean(true_params['V_coef']):.4f}")
    print(f"Estimated mean: {jnp.mean(results.beta):.4f}")
    assert_close(jnp.mean(true_params['V_coef']), jnp.mean(results.beta), tolerance, "Estimated V coefficients mean")
