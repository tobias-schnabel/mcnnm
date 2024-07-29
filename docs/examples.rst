Examples
========

This page provides examples of how to use the lightweight-mcnnm package for various scenarios.

Basic Usage
-----------

Here's a basic example of how to use lightweight-mcnnm:

.. code-block:: python

   import jax.numpy as jnp
   from mcnnm import estimate, generate_data

   # Generate some sample data
   data, true_params = generate_data(nobs=500, nperiods=100, seed=42)

   Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
   W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

   # Estimate the MC-NNM model
   results = estimate(Y, W)

   # Print the estimated treatment effect
   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")

Example 1: Staggered Adoption with Cross-Validation
---------------------------------------------------

In this example, we generate a dataset with covariates and staggered adoption treatment assignment and use the default cross-validation method for selecting the regularization parameters.

.. code-block:: python

   data, true_params = generate_data(nobs=500, nperiods=100, seed=42, assignment_mechanism='staggered',
                                     X_cov=True, Z_cov=True, V_cov=True)

   Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
   W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
   X = jnp.array(true_params['X'])
   Z = jnp.array(true_params['Z'])
   V = jnp.array(true_params['V'])

   results = estimate(Y, W, X=X, Z=Z, V=V)

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 2: Block Assignment with Holdout Validation
---------------------------------------------------

This example demonstrates how to use block treatment assignment and holdout validation for selecting the regularization parameters.

.. code-block:: python

   data, true_params = generate_data(nobs=1000, nperiods=50, seed=123, assignment_mechanism='block',
                                     treated_fraction=0.4, X_cov=False, Z_cov=False, V_cov=False)

   Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
   W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

   results = estimate(Y, W, validation_method='holdout')

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 3: Single Treated Unit with Covariates
----------------------------------------------

This example shows how to handle a dataset with a single treated unit and include covariates in the estimation.

.. code-block:: python

   data, true_params = generate_data(nobs=100, nperiods=200, seed=456, assignment_mechanism='single_treated_unit',
                                     X_cov=True, Z_cov=True, V_cov=True)

   Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
   W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
   X = jnp.array(true_params['X'])
   Z = jnp.array(true_params['Z'])
   V = jnp.array(true_params['V'])

   results = estimate(Y, W, X=X, Z=Z, V=V)

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.4f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Matrix Completion
-----------------

If you're interested in just completing the matrix without estimating the treatment effect, you can use the `complete_matrix` function:

.. code-block:: python

   from mcnnm import complete_matrix

   Y_completed, lambda_L, lambda_H = complete_matrix(Y, W, X=X, Z=Z, V=V)

   print(f"Chosen lambda_L: {lambda_L:.4f}, lambda_H: {lambda_H:.4f}")
   print(f"Completed matrix shape: {Y_completed.shape}")

These examples demonstrate various use cases of the lightweight-mcnnm package. You can adjust the parameters and data generation process to fit your specific needs.
