Usage
=====

Basic Usage
-----------
Here's a basic example of how to use lightweight-mcnnm:

.. code-block:: python

   import jax.numpy as jnp
   from lightweight_mcnnm import estimate

   # Generate some sample data
   Y = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   W = jnp.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]])

   # Fit the MC-NNM model
   results = estimate(Y, W)

   # Print the estimated treatment effect
   print(f"Estimated treatment effect: {results.tau}")

Input Data
----------
The `estimate` function expects two main inputs:

- `Y`: A matrix of observed outcomes
- `W`: A matrix of treatment assignments

Both matrices should be provided as JAX NumPy arrays.

Generating Synthetic Data
-------------------------
For testing and demonstration purposes, you can use the `generate_data` function:

.. code-block:: python

   from mcnnm import generate_data

   data, true_params = generate_data(nobs=500, nperiods=100, seed=42,
                                     assignment_mechanism='staggered')

   Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
   W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)

Advanced Usage
--------------

Including Covariates
^^^^^^^^^^^^^^^^^^^^
lightweight-mcnnm supports three types of covariates:

1. Unit-specific covariates (X)
2. Time-specific covariates (Z)
3. Unit-time specific covariates (V)

Here's an example of how to include covariates in your estimation:

.. code-block:: python

   results = fit(Y, W, X=X, Z=Z, V=V)

Choosing Validation Method
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can choose between cross-validation (the default) and holdout validation:

.. code-block:: python

   results = estimate(Y, W, validation_method='holdout')

Interpreting Results
--------------------
The `estimate` function returns a results object with the following attributes:

- `tau`: The estimated treatment effect
- `lambda_L`: The chosen regularization parameter for the low-rank component
- `lambda_H`: The chosen regularization parameter for the high-rank component

For more detailed examples, please refer to the :doc:`examples` page.