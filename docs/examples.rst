Examples
========

This page provides examples of how to use the mcnnm package for various scenarios. The estimation results and execution timing of these examples can be found
`on Colab <https://colab.research.google.com/github/tobias-schnabel/mcnnm/blob/main/Example.ipynb>`_

Note: The examples used in this documentation are meant to demonstrate functionality, not showcase estimation accuracy. The synthetic data generated for these examples may not reflect real-world scenarios accurately. The purpose is to illustrate how to use the `estimate` function with different settings and validation methods.

Basic Usage
-----------

Here's a basic example of how to use mcnnm:

::

   import jax.numpy as jnp
   from mcnnm.wrappers import estimate
   from mcnnm.utils import generate_data

   # Generate some sample data
   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=True,
       Z_cov=True,
       V_cov=True,
       seed=2024,
       noise_scale=0.1,
       autocorrelation=0.0,
       assignment_mechanism="staggered",
       treatment_probability=0.1,
   )

   # Estimate the MC-NNM model
   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='cv',
       K=3,
       n_lambda=30,
   )

   # Print the estimated treatment effect
   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.3f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 1: Staggered Adoption with Cross-Validation
---------------------------------------------------

In this example, we generate a dataset with covariates and staggered adoption treatment assignment and use the default cross-validation method for selecting the regularization parameters.

::

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=True,
       Z_cov=True,
       V_cov=True,
       seed=2024,
       noise_scale=0.1,
       autocorrelation=0.0,
       assignment_mechanism="staggered",
       treatment_probability=0.1,
   )

   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='cv',
       K=3,
       n_lambda=30,
   )

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.3f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 2: Block Assignment with Cross-Validation
-------------------------------------------------

This example demonstrates how to use block treatment assignment and cross-validation for selecting the regularization parameters.

::

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=False,
       Z_cov=False,
       V_cov=False,
       seed=2024,
       noise_scale=0.1,
       autocorrelation=0.0,
       assignment_mechanism="block",
       treated_fraction=0.1,
   )

   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='cv',
       K=2,
       n_lambda=10,
   )

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.3f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 3: Single Treated Unit with Covariates
----------------------------------------------

This example shows how to handle a dataset with a single treated unit and include covariates in the estimation.

::

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=True,
       Z_cov=True,
       V_cov=True,
       seed=2024,
       noise_scale=0.1,
       assignment_mechanism="single_treated_unit",
   )

   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='cv',
       K=3,
       n_lambda=20,
   )

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.3f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 4: Estimation with Holdout Validation
---------------------------------------------

In this example, we generate data without covariates and staggered adoption treatment assignment and use holdout validation for selecting the regularization parameters.

::

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=100,
       unit_fe=True,
       time_fe=True,
       X_cov=False,
       Z_cov=False,
       V_cov=False,
       seed=2024,
       noise_scale=0.1,
       assignment_mechanism="staggered",
       treatment_probability=0.1,
   )

   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='holdout',
       initial_window=50,
       max_window_size=80,
       step_size=10,
       horizon=5,
       K=5,
       n_lambda=30,
   )

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.3f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Example 5: Estimation with Pre-specified Lambda Values
------------------------------------------------------

This example shows how to estimate the model using pre-specified lambda values.

::

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=True,
       Z_cov=True,
       V_cov=True,
       seed=2024,
       noise_scale=0.1,
       autocorrelation=0.0,
       assignment_mechanism="staggered",
       treatment_probability=0.1,
   )

   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=0.001,
       lambda_H=0.01,
   )

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.2f}")

Example 6: Estimation with Autocorrelation
------------------------------------------

In this example, we generate data with autocorrelation and use a custom Omega matrix in estimation.

::

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=True,
       Z_cov=True,
       V_cov=True,
       seed=2024,
       noise_scale=0.1,
       autocorrelation=0.5,
       assignment_mechanism="last_periods",
       last_treated_periods=5,
   )

   # Create custom Omega matrix with AR(1) structure
   rho = 0.5
   T = Y.shape[1]
   Omega = jnp.power(rho, jnp.abs(jnp.arange(T)[:, None] - jnp.arange(T)))

   results = estimate(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=Omega,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='holdout',
       initial_window=2,
       max_window_size=None,
       step_size=1,
       horizon=1,
       K=3,
       n_lambda=30,
   )

   print(f"True effect: {true_params['treatment_effect']}, Estimated effect: {results.tau:.12f}")
   print(f"Chosen lambda_L: {results.lambda_L:.4f}, lambda_H: {results.lambda_H:.4f}")

Matrix Completion
-----------------

If you're interested in just completing the matrix without estimating the treatment effect, you can use the `complete_matrix` function:

::

   from mcnnm.wrappers import complete_matrix

   Y, W, X, Z, V, true_params = generate_data(
       nobs=50,
       nperiods=10,
       unit_fe=True,
       time_fe=True,
       X_cov=False,
       Z_cov=False,
       V_cov=False,
       seed=2024,
       noise_scale=0.1,
       autocorrelation=0.0,
       assignment_mechanism="block",
       treated_fraction=0.1,
   )

   Y_completed, opt_lambda_L, opt_lambda_H = complete_matrix(
       Y=Y,
       Mask=W,
       X=X,
       Z=Z,
       V=V,
       Omega=None,
       use_unit_fe=True,
       use_time_fe=True,
       lambda_L=None,
       lambda_H=None,
       validation_method='cv',
       K=2,
       n_lambda=10,
   )

   print(f"Chosen lambda_L: {opt_lambda_L:.4f}, lambda_H: {opt_lambda_H:.4f}")
   print(f"Mean absolute error of imputation: {jnp.mean(jnp.abs(Y - Y_completed)):.4f}")
   print(f"Mean squared error of imputation: {jnp.mean(jnp.square(Y - Y_completed)):.4f}")
   print(f"Mean of Y: {jnp.mean(Y):.4f}, Mean of Y_completed: {jnp.mean(Y_completed):.4f}")

Covariates
----------

The `generate_data` function allows you to include three types of covariates in the generated dataset:

1. **Unit-specific covariates (X):** These are characteristics or features that vary across units but remain constant over time.
2. **Time-specific covariates (Z):** These are factors that change over time but are the same for all units at each time point.
3. **Unit-time specific covariates (V):** These are covariates that vary both across units and over time.

In the `generate_data` function, you can control the inclusion of these covariates using the boolean flags X_cov, Z_cov, and V_cov. Setting these flags to True incorporates the respective type of covariates into the generated dataset, while setting them to False excludes them.

These examples demonstrate various use cases of the mcnnm package. You can adjust the parameters and data generation process to fit your specific needs.
