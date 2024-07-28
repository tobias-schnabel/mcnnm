Validation Methods
==================

This package supports two validation methods for selecting optimal regularization parameters:

1. Cross-Validation
-------------------
Cross-validation is implemented in the `cross_validate` function. This method performs K-fold cross-validation to select optimal regularization parameters (lambda_L and lambda_H).

The process works as follows:
1. The data is randomly split into K folds.
2. For each pair of lambda values in the provided grid:
   a. The model is trained K times, each time using K-1 folds for training and the remaining fold for validation.
   b. The loss is computed for each validation fold.
   c. The average loss across all valid folds is calculated.
3. The lambda pair that results in the lowest average loss is selected.

Key features:

• Uses jax.random.bernoulli for random splitting of data.
• Skips folds with no treated units in the test set.
• Handles cases where no valid folds are found.

When to use Cross-Validation:

• Random or quasi-random treatment assignment: If treatments are assigned randomly or in a pattern that doesn't heavily depend on time, cross-validation is appropriate.
• Balanced treatment across time: When the proportion of treated units is relatively stable across different time periods.
• Small to medium-sized datasets: Cross-validation is generally more efficient for smaller datasets where time-based splitting might result in too little data for reliable estimation.
• No strong temporal trends: If your data doesn't exhibit strong time trends or seasonality that might affect the treatment effect estimation.


2. Holdout Validation
---------------------
Holdout validation is implemented in the `time_based_validate` function. This method uses a time-based holdout strategy to select optimal regularization parameters.

The process works as follows:
1. The data is split into training and test sets based on time periods.
2. For each pair of lambda values in the provided grid:
   a. The model is trained on the training set.
   b. The loss is computed on the test set.
   c. This process is repeated for multiple time periods, creating a series of train-test splits.
3. The average loss across all valid periods is calculated for each lambda pair.
4. The lambda pair that results in the lowest average loss is selected.

Key features:
• Supports both fixed window size and expanding window for training data.
• Allows for multiple folds within each time-based split.
• Handles cases where no valid folds are found.

When to use Holdout Validation:

• Time-dependent treatment assignment: If treatments are assigned based on time (e.g., policy changes that affect all units from a certain date), holdout validation is more appropriate.
• Staggered adoption: When units adopt the treatment at different times, holdout validation can capture this temporal structure.
• Large datasets with many time periods: Holdout validation can be more efficient for very large datasets, especially those with many time periods.
• Presence of temporal trends: If your data exhibits strong time trends or seasonality, holdout validation can help account for these patterns.
• Forecasting applications: If your goal includes predicting future outcomes, holdout validation mimics this task more closely.

Choosing between methods:

• Data structure: Consider the temporal structure of your data. If time plays a crucial role in treatment assignment or outcome dynamics, lean towards holdout validation.
• Sample size: For smaller samples, cross-validation might be more robust. For larger samples, especially with many time periods, holdout validation can be more computationally efficient.
• Treatment mechanism: If the treatment mechanism is closely tied to time (like policy changes), holdout validation is often more appropriate.
• Stability of effects: If you suspect the treatment effect might change over time, holdout validation can help capture this, especially with the expanding window option.

In practice, if computational resources allow, it can be informative to try both methods and compare results. Significant differences between the two might indicate important temporal structures in your data that deserve closer examination.

Proposing Lambda Values
=======================
The `propose_lambda` function in the `util.py` file is used to generate a sequence of lambda values for grid search. It works as follows:

1. If no `proposed_lambda` is provided:
   - Returns a logarithmically spaced sequence of `n_lambdas` values between 10^-3 and 10^0.

2. If a `proposed_lambda` is provided:
   - Creates a logarithmically spaced sequence of `n_lambdas` values centered around the `proposed_lambda`.
   - The range spans from `10^(log10(proposed_lambda) - 2)` to `10^(log10(proposed_lambda) + 2)`.

Usage:
- When called without arguments, it provides a default range of lambda values.
- When called with a specific lambda value, it provides a range of values around that lambda for fine-tuning.

Customizing Validation in estimate()
====================================
The `estimate` function in `estimate.py` allows for customization of the validation process through several parameters:

1. `validation_method` (str): Choose between 'cv' for cross-validation or 'holdout' for time-based holdout validation.

2. `lambda_L` and `lambda_H` (Optional[float]): If provided, these values are used directly. If None, they are selected via validation.

3. `n_lambda_L` and `n_lambda_H` (int): Number of lambda values to consider in the grid search for lambda_L and lambda_H respectively.

4. `K` (int): Number of folds for cross-validation (default is 5).

5. `window_size` (Optional[int]): Size of the rolling window for time-based validation.

6. `expanding_window` (bool): Whether to use an expanding window for time-based validation.

7. `max_window_size` (Optional[int]): Maximum size of the expanding window for time-based validation.

8. `max_iter` (int) and `tol` (float): Maximum number of iterations and convergence tolerance for fitting.

9. `verbose` (bool): Whether to print progress messages during validation.

These parameters allow users to fine-tune the validation process according to their specific needs and data characteristics.