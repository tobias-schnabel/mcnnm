Validation Methods
==================

This package supports two validation methods for selecting optimal regularization parameters:

1. Cross-Validation
-------------------
This is the default used in the `estimate` function: `validation_method: str = 'cv'`
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
Holdout validation can be used in the `estimate` function by setting `validation_method='holdout'`. This method uses a time-based holdout strategy to select optimal regularization parameters.

The process works as follows:

1. The data is split into training and test sets based on time periods.
2. For each pair of lambda values in the provided grid:
   a. The model is trained on the training set.
   b. The loss is computed on the test set.
   c. This process is repeated for multiple time periods, creating a series of train-test splits.
3. The average loss across all valid periods is calculated for each lambda pair.
4. The lambda pair that results in the lowest average loss is selected.

Key features:

- Supports both fixed window size and expanding window for training data:
    a. Fixed Window (default): Use this if you believe only recent data is relevant for predicting the near future.
    b. Expanding Window: Use this if you believe all historical data is useful, but you want to limit how far back you go.
- Allows for multiple folds within each time-based split.
- Handles cases where no valid folds are found.

When to use Holdout Validation:

- Large datasets with many time periods
- Presence of temporal trends or seasonality
- Forecasting applications where predicting future outcomes is the goal

Configuring Holdout Validation in `estimate()`:

To use holdout validation, set the following parameters in the `estimate()` function:

- `validation_method='holdout'`: Activates holdout validation (default is 'cv').
- `window_size`: Set the size of the training window (default is 80% of total time periods).
- `expanding_window`: Set to True for expanding window, False for fixed window (default is False).
- `max_window_size`: Set the maximum window size when using expanding window (default is None).
- `n_folds`: Set the number of folds for time-based validation (default is 5).

Example:
```python
results = estimate(Y, W, X=X, Z=Z, V=V,
                   validation_method='holdout',
                   window_size=50,
                   expanding_window=True,
                   max_window_size=80,
                   n_folds=3)
```

This call uses time-based holdout validation with an expanding window, starting with a window size of 50, expanding up to 80, and using 3 folds for validation.
Notes:

- The `window_size` must be less than the total number of time periods.
- If `expanding_window` is True and max_window_size is not set, it defaults to window_size.
- The actual number of folds used might be less than `n_folds` if there aren't enough time periods.
- Time-based validation requires at least 5 time periods in total.

Choosing between Cross-Validation and Holdout Validation:
Consider:

- Data structure and temporal importance
- Sample size and computational efficiency
- Treatment mechanism's relation to time
- Stability of effects over time

If resources allow, trying both methods can provide insights into temporal structures in your data.

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

1. `validation_method` (str): Choose between 'cv' for cross-validation (the default) or 'holdout' for time-based holdout validation.

2. `lambda_L` and `lambda_H` (Optional[float]): If provided, these values are used as the starting point for the grid search.

3. `n_lambda_L` and `n_lambda_H` (int): Number of lambda values to consider in the grid search for lambda_L and lambda_H respectively. If both lambda values are provided and `n_lambda_L` and `n_lambda_H` are set to 1, no grid search is performed.

4. `K` (int): Number of folds for cross-validation (default is 5).

5. `window_size` (Optional[int]): Size of the rolling window for time-based validation.

6. `expanding_window` (bool): Whether to use an expanding window for time-based validation.

7. `max_window_size` (Optional[int]): Maximum size of the expanding window for time-based validation.

8. `max_iter` (int) and `tol` (float): Maximum number of iterations and convergence tolerance for fitting.

9. `verbose` (bool): Whether to print progress messages during validation.

These parameters allow users to fine-tune the validation process according to their specific needs and data characteristics.