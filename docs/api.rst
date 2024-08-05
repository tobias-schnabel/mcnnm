API Reference
=============

This page provides detailed information about the main functions in lightweight-mcnnm.

Main Functions
--------------

estimate
^^^^^^^^
.. autofunction:: mcnnm.estimate

Parameters:
    - Y (Array): The observed outcome matrix.
    - W (Array): The binary treatment matrix.
    - X (Optional[Array]): The unit-specific covariates matrix. Default is None.
    - Z (Optional[Array]): The time-specific covariates matrix. Default is None.
    - V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
    - Omega (Optional[Array]): The autocorrelation matrix. Default is None.
    - lambda_L (Optional[float]): The regularization parameter for L. If None, it will be selected via validation.
    - lambda_H (Optional[float]): The regularization parameter for H. If None, it will be selected via validation.
    - n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
    - n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
    - return_tau (bool): Whether to return the estimated treatment effect. Default is True.
    - return_lambda (bool): Whether to return the selected lambda values. Default is True.
    - return_completed_L (bool): Whether to return the completed low-rank matrix. Default is True.
    - return_completed_Y (bool): Whether to return the completed outcome matrix. Default is True.
    - return_fixed_effects (bool): Whether to return the estimated fixed effects. Default is False.
    - return_covariate_coefficients (bool): Whether to return the estimated covariate coefficients. Default is False.
    - max_iter (int): Maximum number of iterations for fitting. Default is 1000.
    - tol (float): Convergence tolerance for fitting. Default is 1e-4.
    - verbose (bool): Whether to print progress messages. Default is False.
    - validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
    - K (int): Number of folds for cross-validation. Default is 5.
    - window_size (Optional[int]): Size of the rolling window for time-based validation. Default is None.
    - expanding_window (bool): Whether to use an expanding window for time-based validation. Default is False.
    - max_window_size (Optional[int]): Maximum size of the expanding window for time-based validation. Default is None.

Returns:
    MCNNMResults: A named tuple containing the estimation results.

complete_matrix
^^^^^^^^^^^^^^^
.. autofunction:: mcnnm.complete_matrix

Parameters:
    - Y (Array): The observed outcome matrix.
    - W (Array): The binary treatment matrix.
    - X (Optional[Array]): The unit-specific covariates matrix. Default is None.
    - Z (Optional[Array]): The time-specific covariates matrix. Default is None.
    - V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
    - Omega (Optional[Array]): The autocorrelation matrix. Default is None.
    - lambda_L (Optional[float]): The regularization parameter for L. If None, it will be selected via validation.
    - lambda_H (Optional[float]): The regularization parameter for H. If None, it will be selected via validation.
    - n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
    - n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
    - max_iter (int): Maximum number of iterations for fitting. Default is 1000.
    - tol (float): Convergence tolerance for fitting. Default is 1e-4.
    - verbose (bool): Whether to print progress messages. Default is False.
    - validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
    - K (int): Number of folds for cross-validation. Default is 5.
    - window_size (Optional[int]): Size of the rolling window for time-based validation. Default is None.
    - expanding_window (bool): Whether to use an expanding window for time-based validation. Default is False.
    - max_window_size (Optional[int]): Maximum size of the expanding window for time-based validation. Default is None.

Returns:
    Tuple[Array, float, float]: A tuple containing the completed outcome matrix, optimal lambda_L, and optimal lambda_H.

generate_data
^^^^^^^^^^^^^
.. autofunction:: mcnnm.generate_data

Parameters:
    - nobs (int): Number of units. Default is 500.
    - nperiods (int): Number of time periods. Default is 100.
    - treatment_probability (float): The probability of a unit being treated (for staggered adoption). Default is 0.5.
    - rank (int): The rank of the low-rank matrix L. Default is 5.
    - treatment_effect (float): The true treatment effect. Default is 1.0.
    - unit_fe (bool): Whether to include unit fixed effects. Default is True.
    - time_fe (bool): Whether to include time fixed effects. Default is True.
    - X_cov (bool): Whether to include unit-specific covariates. Default is True.
    - Z_cov (bool): Whether to include time-specific covariates. Default is True.
    - V_cov (bool): Whether to include unit-time specific covariates. Default is True.
    - fixed_effects_scale (float): The scale of the fixed effects. Default is 0.1.
    - covariates_scale (float): The scale of the covariates and their coefficients. Default is 0.1.
    - noise_scale (float): The scale of the noise. Default is 0.1.
    - assignment_mechanism (str): The treatment assignment mechanism to use. Default is 'staggered'.
    - treated_fraction (float): Fraction of units to be treated (for block and single_treated_period). Default is 0.2.
    - last_treated_periods (int): Number of periods to treat all units at the end (for last_periods mechanism). Default is 10.
    - autocorrelation (float): The autocorrelation coefficient for the error term. Default is 0.0.
    - seed (Optional[int]): Random seed for reproducibility. Default is None.

Returns:
    Tuple[pd.DataFrame, Dict]: A tuple containing the generated data as a DataFrame and a dictionary of true parameter values.

Classes
-------

MCNNMResults
^^^^^^^^^^^^
.. autoclass:: mcnnm.MCNNMResults
   :members:
   :noindex:

Attributes:
    - tau (Optional[float]): The estimated average treatment effect. :noindex:
    - lambda_L (Optional[float]): The selected regularization parameter for L. :noindex:
    - lambda_H (Optional[float]): The selected regularization parameter for H. :noindex:
    - L (Optional[Array]): The estimated low-rank matrix. :noindex:
    - Y_completed (Optional[Array]): The completed outcome matrix. :noindex:
    - gamma (Optional[Array]): The estimated unit fixed effects. :noindex:
    - delta (Optional[Array]): The estimated time fixed effects. :noindex:
    - beta (Optional[Array]): The estimated unit-time specific covariate coefficients. :noindex:
    - H (Optional[Array]): The estimated covariate coefficient matrix. :noindex:

Internal Functions
------------------
Function Hierarchy and Workflow
-------------------------------
All equations, sections, tables referenced hereafter refer to:
`Athey et al. (2021) <https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924>`_

When the `estimate()` function is called, the following sequence of operations occurs:

1. Input Checking and Preprocessing:
   - `check_inputs()` is called to validate and preprocess the input data.

2. Lambda Selection (if not provided):
   - If `lambda_L` or `lambda_H` are not provided, the function enters a lambda selection process:
   - `propose_lambda()` generates a grid of lambda values.
   - Depending on the `validation_method`:
   - For 'cv': `cross_validate()` is called, which in turn calls `compute_cv_loss()` for each fold.
   - For 'holdout': `time_based_validate()` is called, which performs time-based validation.

3. Model Fitting:
   - `initialize_params()` is called to set initial values for L, H, gamma, delta, and beta.
   - `fit()` is called, which iteratively calls `fit_step()` until convergence or max iterations:
   - `update_L()` updates the low-rank matrix L (corresponds to Equation (7) in Athey et al.)
   - `update_H()` updates the covariate coefficient matrix H (corresponds to Equation (8))
   - `update_gamma_delta_beta()` updates fixed effects and unit-time specific coefficients

4. Treatment Effect Computation (if requested):
   - If `return_tau` is True, `compute_treatment_effect()` is called to estimate the average treatment effect.

5. Results Compilation:
   - The function compiles the requested outputs into an `MCNNMResults` object.

Key Function Descriptions
-------------------------

estimate():
  The main function that orchestrates the entire MC-NNM estimation process. It implements the overall algorithm described in Sections 4.2/8.1 of Athey et al.

fit():
  Fits the MC-NNM model using an iterative process. This function implements the core of the algorithm described in Sections 4.2/8.1.

update_L():
  Updates the low-rank matrix L using soft-thresholding. This corresponds to Equation (10).

update_H():
  Updates the covariate coefficient matrix H. This step is part of the iterative algorithm described in Section 8.2.

update_gamma_delta_beta():
  Updates the fixed effects (gamma for unit, delta for time) and unit-time specific covariate coefficients (beta). This step is part of the iterative algorithm described in Section 8.2.

cross_validate():
  Performs K-fold cross-validation to select optimal regularization parameters. This implements the cross-validation procedure mentioned in Section 4.3 of Athey et al.

time_based_validate():
  Performs time-based validation to select optimal regularization parameters. This is an alternative to cross-validation that respects the temporal structure of the data.

compute_treatment_effect():
  Computes the average treatment effect using the completed matrix. This implements the treatment effect computation described in Section 2.

shrink_lambda():
  Applies soft-thresholding to the singular values of a matrix. This corresponds to Equation (9).

nuclear_norm():
  Computes the nuclear norm of a matrix, which is the sum of its singular values. See Table 1.

frobenius_norm():
  Computes the Frobenius norm of a matrix, See Table 1.

propose_lambda():
  Generates a sequence of lambda values for grid search in hyperparameter tuning. This function is used for both L and H.

initialize_params():
  Initializes the model parameters (L, H, gamma, delta, beta) before the iterative fitting process begins.

generate_data():
  Generates synthetic data for testing the MC-NNM model. While not part of the estimation process, this function is useful for simulation studies and testing the estimator under various conditions.

This section provides detailed information about the internal functions used in lightweight-mcnnm.

Type Aliases
^^^^^^^^^^^^

lightweight-mcnnm uses the following type aliases for clarity and consistency:

- ``Array``: Alias for ``jax.Array``, representing a JAX array.
- ``Scalar``: Union type of ``float``, ``int``, or ``Array``, representing a scalar value or a single-element array.

These aliases are used to improve code readability and to allow for easy switching between different array implementations if needed in the future.

Functions from estimate.py
^^^^^^^^^^^^^^^^^^^^^^^^^^

update_L
~~~~~~~~
.. autofunction:: mcnnm.estimate.update_L

update_H
~~~~~~~~
.. autofunction:: mcnnm.estimate.update_H

update_gamma_delta_beta
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.estimate.update_gamma_delta_beta

compute_beta
~~~~~~~~~~~~
.. autofunction:: mcnnm.estimate.compute_beta

fit_step
~~~~~~~~
.. autofunction:: mcnnm.estimate.fit_step

fit
~~~
.. autofunction:: mcnnm.estimate.fit

compute_cv_loss
~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.estimate.compute_cv_loss

cross_validate
~~~~~~~~~~~~~~
.. autofunction:: mcnnm.estimate.cross_validate

time_based_validate
~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.estimate.time_based_validate

compute_treatment_effect
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.estimate.compute_treatment_effect

Functions from util.py
^^^^^^^^^^^^^^^^^^^^^^

p_o
~~~
.. autofunction:: mcnnm.util.p_o

p_perp_o
~~~~~~~~
.. autofunction:: mcnnm.util.p_perp_o

shrink_lambda
~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.shrink_lambda

frobenius_norm
~~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.frobenius_norm

nuclear_norm
~~~~~~~~~~~~
.. autofunction:: mcnnm.util.nuclear_norm

element_wise_l1_norm
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.element_wise_l1_norm

propose_lambda
~~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.propose_lambda

initialize_params
~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.initialize_params

check_inputs
~~~~~~~~~~~~
.. autofunction:: mcnnm.util.check_inputs

generate_time_based_validate_defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.generate_time_based_validate_defaults

print_with_timestamp
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.util.print_with_timestamp
