Validation Methods
==================

This package supports two validation methods for selecting optimal regularization parameters:

1. Cross-Validation
-------------------
This is the default method used in the `estimate` function: `validation_method: str = 'cv'`
Cross-validation is implemented in the `cross_validate` function. This method performs K-fold cross-validation to select optimal regularization parameters (lambda_L and lambda_H).

The process works as follows:
1. The data is randomly split into K folds.
2. For each pair of lambda values in the provided grid:

a. The model is trained K times, each time using K-1 folds for training and the remaining fold for validation.
b. The loss is computed for each validation fold.
c. The average loss across all valid folds is calculated.

3. The lambda pair that results in the lowest average loss is selected.

Key features:

- Uses jax.random.bernoulli for random splitting of data.
- Skips folds with no treated units in the test set.
- Handles cases where no valid folds are found.

Key Parameters:

1. `K` (int):
   This parameter sets the number of folds for cross-validation. It is user-selectable, with a default value of 5.

   Example: If `K=10`, the data will be split into 10 folds, and the model will be trained and validated 10 times.

   Why it's useful: A higher K value provides a more thorough evaluation but increases computation time. A lower K value is faster but may be less robust.

2. `n_lambda_L` and `n_lambda_H` (int):
   These parameters determine the number of lambda values to consider in the grid search for lambda_L and lambda_H respectively.

   Example: If `n_lambda_L=10` and `n_lambda_H=10`, the grid search will consider 100 different combinations of lambda values.

Example Usage:

::

   results = estimate(Y, W, X=X, Z=Z, V=V,
                   validation_method='cv',
                   K=10,
                   n_lambda_L=8,
                   n_lambda_H=8)

This configuration would:

- Use 10-fold cross-validation
- Consider 8 different values for lambda_L and 8 for lambda_H, resulting in 64 lambda pairs to evaluate

Choosing Parameters:

- `K`: Common choices are 5 or 10. Higher values provide more thorough validation but increase computation time. For smaller datasets, you might use leave-one-out cross-validation by setting K equal to the number of observations.
- `n_lambda_L` and `n_lambda_H`: These control the granularity of your lambda search. Higher values provide a more comprehensive search but increase computation time. Start with moderate values (e.g., 5-10) and adjust based on your computational resources and the sensitivity of your results to lambda values.

When to use Cross-Validation:

- Random or quasi-random treatment assignment: If treatments are assigned randomly or in a pattern that doesn't heavily depend on time, cross-validation is appropriate.
- Balanced treatment across time: When the proportion of treated units is relatively stable across different time periods.
- Small to medium-sized datasets: Cross-validation is generally more efficient for smaller datasets where time-based splitting might result in too little data for reliable estimation.
- No strong temporal trends: If your data doesn't exhibit strong time trends or seasonality that might affect the treatment effect estimation.

While cross-validation is often a good default choice, it's important to consider the structure of your data and the nature of your research question when deciding between cross-validation and time-based validation methods.

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

Key Parameters:

1. `initial_window` (int):
   This parameter sets the number of initial time periods to use for the first training set.
   It determines how much historical data is used for the initial model training.

   Example: If `initial_window=50` and you have 100 time periods, the first training set will use periods 0-49.

   Why it's useful: This allows you to control how much historical data is considered relevant for prediction.

2. `step_size` (int):
   This parameter determines how many time periods to move forward for each subsequent split.
   It controls the granularity of your validation process.

   Example: If `step_size=10`, after the initial training, the next split will start at period 10, then 20, and so on.

   Why it's useful: Smaller step sizes provide more validation points but increase computation time. Larger step sizes are faster but may miss important temporal patterns.

3. `horizon` (int):
   This sets the number of future time periods to predict (forecast horizon).
   It determines how far into the future the model is expected to predict accurately.

   Example: If `horizon=5`, each validation step will predict 5 time periods ahead.

   Why it's useful: This allows you to tailor the validation to your specific forecasting needs. A longer horizon tests the model's long-term predictive power, while a shorter horizon focuses on immediate future predictions.

4. `K` (int):
   This parameter sets the number of folds (splits) to use in the time-based validation.
   It determines how many train-test splits are created and evaluated.

   Example: If `K=5`, the function will create 5 different train-test splits to evaluate the model.

   Why it's useful: More folds provide a more robust evaluation but increase computation time. Fewer folds are faster but may be less reliable.

5. `max_window_size` (Optional[int]):
   This parameter sets the maximum size of the window to consider. If None, all data is used.
   It effectively limits how far back in time the model will look for training data.

   Example: If `max_window_size=80` and you have 100 time periods, only the most recent 80 periods will be used for any training set.

   Why it's useful: This can be helpful if you believe that very old data is no longer relevant to current predictions, or if you want to limit computational resources.

Example Usage:

::

   results = estimate(Y, W, X=X, Z=Z, V=V,
                      validation_method='holdout',
                      initial_window=50,
                      step_size=10,
                      horizon=5,
                      K=5,
                      max_window_size=80)

This configuration would:

• Start with an initial training window of 50 time periods
• Move forward by 10 periods for each subsequent split
• Predict 5 periods into the future for each validation step
• Create 5 different train-test splits for validation
• Use at most the 80 most recent time periods for any training set

Choosing Parameters:

1. `initial_window`: Set this based on how much historical data you believe is necessary to train a good initial model. If your data has strong seasonality, consider setting this to at least one full cycle.
2. `step_size`: Smaller values provide more granular validation but increase computation time. A good starting point might be 5-10% of your total time periods.
3. `horizon`: Set this to match your forecasting needs. If you're interested in short-term predictions, a small horizon (1-5 periods) might be appropriate. For long-term forecasting, consider larger values.
4. `K`: More folds generally provide more robust results but increase computation time. 5-10 folds are common choices.
5. `max_window_size`: If you believe very old data might not be relevant, set this to limit the historical data used. Otherwise, leaving it as None allows the model to use all available data.

These parameters allow for flexible time-based validation strategies. You can create a rolling window approach by setting step_size equal to horizon, or an expanding window approach by setting step_size smaller than horizon. The max_window_size parameter allows you to implement a sliding window approach if desired.

When to use Holdout Validation:

• Large datasets with many time periods
• Presence of temporal trends or seasonality
• When you want to explicitly test the model's predictive performance over time
• When you believe recent data is more relevant for prediction than older data
• When you want to simulate real-world forecasting scenarios in your validation process

The optimal configuration may depend on your specific dataset and prediction task. It's often beneficial to experiment with different parameter settings to find what works best for your particular case.

Proposing Lambda Values
-----------------------
The internal `propose_lambda` function in the `util.py` file is used to generate a sequence of lambda values for grid search. It works as follows:

1. If no `proposed_lambda` is provided:

   • Returns a logarithmically spaced sequence of `n_lambdas` values between 10^-3 and 10^0.

2. If a `proposed_lambda` is provided:

   • Creates a logarithmically spaced sequence of `n_lambdas` values centered around the `proposed_lambda`.
   • The range spans from `10^(log10(proposed_lambda) - 2)` to `10^(log10(proposed_lambda) + 2)`.

Usage:

• When called without arguments, it provides a default range of lambda values.
• When called with a specific lambda value, it provides a range of values around that lambda for fine-tuning.

Customizing Validation in estimate()
------------------------------------
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
