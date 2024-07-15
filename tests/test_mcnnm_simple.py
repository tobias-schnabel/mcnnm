import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np
from mcnnm.main import fit

def generate_data(nobs=500, nperiods=100, nobsgroups=50, treated_period=20):
    key = random.PRNGKey(0)

    # Generate unit data
    unit_key, group_key, fe_key = random.split(key, 3)
    unit = pd.DataFrame({
        'unit': range(1, nobs + 1),
        'obsgroup': random.choice(group_key, nobsgroups, shape=(nobs,)) + 1,
        'unit_fe': random.normal(fe_key, shape=(nobs,)) * 0.5
    })

    # Assign treatment status
    shuffled_groups = np.random.permutation(unit['obsgroup'].unique())
    half = len(shuffled_groups) // 2
    unit['group'] = np.where(unit['obsgroup'].isin(shuffled_groups[:half]), treated_period, nperiods + 1)
    unit['evertreated'] = np.where(unit['group'] == treated_period, 1, 0)
    unit['avg_te'] = np.where(unit['group'] == treated_period, 1, 0)
    unit['te'] = unit['avg_te'] + random.normal(random.PRNGKey(1), shape=(nobs,)) * 0.2

    # Generate period data
    period_key = random.PRNGKey(2)
    period = pd.DataFrame({
        'period': range(1, nperiods + 1),
        'period_fe': random.normal(period_key, shape=(nperiods,)) * 0.5
    })

    # Combine unit and period data
    data = unit.merge(period, how='cross')
    data['error'] = random.normal(random.PRNGKey(3), shape=(len(data),)) * 0.5
    data['treat'] = np.where((data['evertreated'] == 1) & (data['period'] > treated_period), 1, 0)
    data['t_eff'] = np.where(data['treat'] == 1, data['te'], 0)
    data['y'] = data['unit_fe'] + data['period_fe'] + data['t_eff'] + data['error']
    data['group'] = np.where(data['group'] == nperiods + 1, 0, data['group'])

    return data

def test_mcnnm_simple():
    # Generate data
    data = generate_data()

    # Prepare inputs for MCNNM
    Y = data.pivot(index='unit', columns='period', values='y').values
    W = data.pivot(index='unit', columns='period', values='treat').values

    # Set lambda values
    lambda_L = 0.1
    lambda_H = 0.1

    # Fit MCNNM
    results = fit(Y, W, lambda_L=lambda_L, lambda_H=lambda_H)

    # Check results
    assert len(results) == 4, "Expected 4 return values from fit function"
    tau, lambda_L_out, L, Y_completed = results

    print("tau:", tau)
    print("lambda_L:", lambda_L_out)
    print("L shape:", L.shape)
    print("Y_completed shape:", Y_completed.shape)

    assert isinstance(tau, (float, jnp.ndarray)), f"tau should be a float or JAX array, got {type(tau)}"
    if isinstance(tau, jnp.ndarray):
        assert tau.shape == (), f"tau should be a scalar, got shape {tau.shape}"
    assert isinstance(lambda_L_out, (float, jnp.ndarray)), f"lambda_L should be a float or JAX array, got {type(lambda_L_out)}"
    assert isinstance(L, jnp.ndarray), f"L should be a JAX array, got {type(L)}"
    assert isinstance(Y_completed, jnp.ndarray), f"Y_completed should be a JAX array, got {type(Y_completed)}"

    assert L.shape == Y.shape, f"L shape {L.shape} should match Y shape {Y.shape}"
    assert Y_completed.shape == Y.shape, f"Y_completed shape {Y_completed.shape} should match Y shape {Y.shape}"

    print("All assertions passed!")

if __name__ == "__main__":
    test_mcnnm_simple()