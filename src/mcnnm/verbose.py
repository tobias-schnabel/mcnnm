# Description: Example of how to use the MC-NNM model with verbose output
# import numpy as np
import jax
import jax.numpy as jnp
from mcnnm.main import estimate
from mcnnm.simulate import generate_data_factor

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)

# Generate sample data
nobs, nperiods = 500, 100
data, true_params = generate_data_factor(nobs=nobs, nperiods=nperiods, seed=42)
print("data generated")

Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])
print("data converted, begin fitting")
# Fit the MC-NNM model
results = estimate(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True, verbose=True)

print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
print(f"Chosen lambda_L: {results.lambda_L:.4f}")
print(f"Chosen lambda_H: {results.lambda_H:.4f}")
