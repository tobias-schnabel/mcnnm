# Description: Example of how to use the MC-NNM model with verbose output
# import numpy as np
import jax
import jax.numpy as jnp
from mcnnm.main import estimate
from mcnnm.simulate import generate_data_factor
from mcnnm.util import print_with_timestamp

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)

# Generate sample data
nobs, nperiods = 1000, 100
data, true_params = generate_data_factor(nobs=nobs, nperiods=nperiods, seed=42)

Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])
print_with_timestamp("Data converted, begin fitting")
# Fit the MC-NNM model
results = estimate(Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True, verbose=True)
print_with_timestamp("Fitting completed")
print(f"\nTrue effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
print(f"Chosen lambda_L: {results.lambda_L:.4f}")
print(f"Chosen lambda_H: {results.lambda_H:.4f}")
print("\nFixed Effects Comparison:")
print("Unit Fixed Effects:")
print(f"True mean:      {jnp.mean(true_params['unit_fe']):.4f}")
print(f"Estimated mean: {jnp.mean(results.gamma):.4f}")

print("\nTime Fixed Effects:")
print(f"True mean:      {jnp.mean(true_params['time_fe']):.4f}")
print(f"Estimated mean: {jnp.mean(results.delta):.4f}")

print("\nCovariate Coefficients Comparison:")
print("X Coefficients:")
print(f"True mean:      {jnp.mean(true_params['X_coef']):.4f}")
print(f"Estimated mean: {jnp.mean(results.H[:X.shape[1], :Z.shape[1]]):.4f}")

print("\nZ Coefficients:")
print(f"True mean:      {jnp.mean(true_params['Z_coef']):.4f}")
print(f"Estimated mean: {jnp.mean(results.H[:X.shape[1], :Z.shape[1]].T):.4f}")

print("\nV Coefficients:")
print(f"True mean:      {jnp.mean(true_params['V_coef']):.4f}")
print(f"Estimated mean: {jnp.mean(results.beta):.4f}")
