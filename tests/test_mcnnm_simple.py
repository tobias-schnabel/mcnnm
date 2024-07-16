import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest
from mcnnm.main import fit, MCNNMResults
from mcnnm.simulate import generate_data_factor
import jax


def assert_close(true_value, estimated_value, tolerance, message):
    assert np.abs(true_value - estimated_value) < tolerance, f"{message}: true={true_value:.4f}, estimated={estimated_value:.4f}, difference={np.abs(true_value - estimated_value):.4f}"


jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)


@pytest.mark.timeout(120)
def test_mcnnm_accuracy_no_covariates(tolerance=0.1):
    nobs, nperiods = 500, 100
    data, true_params = generate_data_factor(nobs=nobs, nperiods=nperiods, seed=42,
                                             covariates_scale=0.0)  # Set covariates_scale to 0 to effectively remove covariates

    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

    print(f"\nProportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {jnp.mean(Y)}")
    print(f"Std of Y: {jnp.std(Y)}")

    results = fit(Y, W, return_fixed_effects=True)

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
    data, true_params = generate_data_factor(nobs=nobs, nperiods=nperiods, seed=42)

    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])

    print(f"\nProportion of treated observations: {W.mean()}")
    print(f"Mean of Y: {jnp.mean(Y)}")
    print(f"Std of Y: {jnp.std(Y)}")

    results = fit(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True)

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
