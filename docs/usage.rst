Usage
=====

JIT Compilation
---------------
By default, this package uses JAX's JIT compilation for better performance in typical use cases. If you want to disable JIT compilation, you can add the following line at the top of your script:

.. code-block:: python

   jax.config.update('jax_disable_jit', True)

Note that disabling JIT may impact performance depending on your specific use case. I have found leaving JIT enabled to be the best option for most use cases. An example use case where disabling JIT may be sensible is calling estimate() multiple times on datasets of different sizes, which triggers recompilation any time the input data shape changes.



Comprehensive Example
---------------------
For a comprehensive example of using lightweight-mcnnm, please refer to the following Colab notebook:

https://colab.research.google.com/github/tobias-schnabel/mcnnm/blob/main/Example.ipynb


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

Both matrices should be provided as JAX NumPy arrays. The package offers a convenience function convert_inputs that can be used to convert pandas DataFrames to JAX NumPy arrays.

Generating Synthetic Data
-------------------------
For testing and demonstration purposes, you can use the `generate_data` function:

.. code-block:: python

   from mcnnm import generate_data, estimate

   Y, W, X, Z, V, true_params = generate_data(
        nobs=50,
        nperiods=10,
        unit_fe=True,
        time_fe=True,
        X_cov=True,
        Z_cov=True,
        V_cov=True,
        seed=2024,
        noise_scale=0.2,
        autocorrelation=0.0,
        assignment_mechanism="last_periods",
        treated_fraction=0.4,
        last_treated_periods=3,
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
    K=10,
    n_lambda=12,
    max_iter=1e5,
    tol=1e-5,
    )

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

   results = estimate(Y, W, X=X, Z=Z, V=V)

Choosing Validation Method
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can choose between cross-validation (the default) and holdout validation:

.. code-block:: python

   results = estimate(Y, W, validation_method='holdout')


Interpreting Results
--------------------
The `estimate` function returns a results object with the following main attributes:

- `tau`: The estimated treatment effect
- `Y_completed`: The imputed matrix of outcomes
- `lambda_L`: The chosen regularization parameter for the low-rank component
- `lambda_H`: The chosen regularization parameter for the high-rank component

For more detailed examples, please refer to the :doc:`examples` page.
