API Reference
=============

This page provides detailed information about the main functions in lightweight-mcnnm.

Main Functions
--------------

estimate
^^^^^^^^
.. autofunction:: lightweight_mcnnm.estimate

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
.. autofunction:: lightweight_mcnnm.complete_matrix

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
.. autofunction:: lightweight_mcnnm.generate_data

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
.. autoclass:: lightweight_mcnnm.MCNNMResults
   :members:

Attributes:
    - tau (Optional[float]): The estimated average treatment effect.
    - lambda_L (Optional[float]): The selected regularization parameter for L.
    - lambda_H (Optional[float]): The selected regularization parameter for H.
    - L (Optional[Array]): The estimated low-rank matrix.
    - Y_completed (Optional[Array]): The completed outcome matrix.
    - gamma (Optional[Array]): The estimated unit fixed effects.
    - delta (Optional[Array]): The estimated time fixed effects.
    - beta (Optional[Array]): The estimated unit-time specific covariate coefficients.
    - H (Optional[Array]): The estimated covariate coefficient matrix.