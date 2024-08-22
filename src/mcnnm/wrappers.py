from .types import Array, Scalar
from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp


def compute_treatment_effect(
    Y: Array,
    L: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    H: Array,
    X: Array,
    W: Array,
    Z: Array,
    V: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Scalar:
    """
    Compute the average treatment effect using the MC-NNM model estimates.

    This function calculates the difference between the observed outcomes and the
    completed (counterfactual) outcomes for treated units, then averages this
    difference to estimate the average treatment effect.

    Args:
        Y (Array): The observed outcome matrix.
        L (Array): The estimated low-rank matrix.
        gamma (Array): The estimated unit fixed effects.
        delta (Array): The estimated time fixed effects.
        beta (Array): The estimated unit-time specific covariate coefficients.
        H (Array): The estimated covariate coefficient matrix.
        X (Array): The unit-specific covariates matrix.
        W (Array): The binary treatment matrix.
        Z (Array): The time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.

    Returns:
        Scalar: The estimated average treatment effect.
    """
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))
    Y_completed = L + jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))

    def add_unit_fe(y):
        return y + jnp.outer(gamma, jnp.ones(T))

    def add_time_fe(y):
        return y + jnp.outer(jnp.ones(N), delta)

    Y_completed = jax.lax.cond(use_unit_fe, add_unit_fe, lambda y: y, Y_completed)

    Y_completed = jax.lax.cond(use_time_fe, add_time_fe, lambda y: y, Y_completed)

    Y_completed = jax.lax.cond(
        V.shape[2] > 0,
        lambda y: y + jnp.sum(V * beta[None, None, :], axis=2),
        lambda y: y,
        Y_completed,
    )

    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    return tau


class MCNNMResults(NamedTuple):
    """
    A named tuple containing the results of the MC-NNM (Matrix Completion with Nuclear Norm Minimization) estimation.

    This class encapsulates all the key outputs from the MC-NNM model, including
    the estimated treatment effect, selected regularization parameters, and
    various estimated matrices and vectors.

    Attributes:
        tau (Optional[Scalar]): The estimated average treatment effect.
        lambda_L (Optional[Scalar]): The selected regularization parameter for the low-rank matrix L.
        lambda_H (Optional[Scalar]): The selected regularization parameter for the covariate coefficient matrix H.
        L (Optional[Array]): The estimated low-rank matrix.
        Y_completed (Optional[Array]): The completed outcome matrix (including counterfactuals).
        gamma (Optional[Array]): The estimated unit fixed effects.
        delta (Optional[Array]): The estimated time fixed effects.
        beta (Optional[Array]): The estimated unit-time specific covariate coefficients.
        H (Optional[Array]): The estimated covariate coefficient matrix.

    All attributes are optional and initialized to None by default.
    """

    tau: Optional[Scalar] = None
    lambda_L: Optional[Scalar] = None
    lambda_H: Optional[Scalar] = None
    L: Optional[Array] = None
    Y_completed: Optional[Array] = None
    gamma: Optional[Array] = None
    delta: Optional[Array] = None
    beta: Optional[Array] = None
    H: Optional[Array] = None


def estimate(
    Y: Array,
    W: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
    use_unit_fe: bool = True,
    use_time_fe: bool = True,
    lambda_L: Optional[Scalar] = None,
    lambda_H: Optional[Scalar] = None,
    n_lambda_L: int = 10,
    n_lambda_H: int = 10,
    return_tau: bool = True,
    return_lambda: bool = True,
    return_completed_L: bool = True,
    return_completed_Y: bool = True,
    return_fixed_effects: bool = False,
    return_covariate_coefficients: bool = False,
    max_iter: int = 1000,
    tol: Scalar = 1e-4,
    validation_method: str = "cv",
    K: int = 5,
    initial_window: Optional[int] = None,
    step_size: Optional[int] = None,
    horizon: Optional[int] = None,
    max_window_size: Optional[int] = None,
) -> MCNNMResults:
    """
    Estimate the parameters of the MC-NNM (Matrix Completion with Nuclear Norm Minimization) model.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        X (Optional[Array]): The unit-specific covariates matrix. Default is None.
        Z (Optional[Array]): The time-specific covariates matrix. Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor. Default is None.
        Omega (Optional[Array]): The autocorrelation matrix. Default is None.
        use_unit_fe (bool): Whether to use unit fixed effects. Default is True.
        use_time_fe (bool): Whether to use time fixed effects. Default is True.
        lambda_L (Optional[Scalar]): The regularization parameter for L. If None, it will be selected via validation.
        lambda_H (Optional[Scalar]): The regularization parameter for H. If None, it will be selected via validation.
        n_lambda_L (int): Number of lambda_L values to consider in grid search. Default is 10.
        n_lambda_H (int): Number of lambda_H values to consider in grid search. Default is 10.
        return_tau (bool): Whether to return the estimated average treatment effect. Default is True.
        return_lambda (bool): Whether to return the selected regularization parameters. Default is True.
        return_completed_L (bool): Whether to return the estimated low-rank matrix L. Default is True.
        return_completed_Y (bool): Whether to return the completed outcome matrix. Default is True.
        return_fixed_effects (bool): Whether to return the estimated unit and time fixed effects. Default is False.
        return_covariate_coefficients (bool): Whether to return the estimated covariate coefficients. Default is False.
        max_iter (int): Maximum number of iterations for fitting. Default is 1000.
        tol (Scalar): Convergence tolerance for fitting. Default is 1e-4.
        validation_method (str): Method for selecting lambda values. Either 'cv' or 'holdout'. Default is 'cv'.
        K (int): Number of folds for cross-validation or time-based validation. Default is 5.
        initial_window (Optional[int]): Number of initial time periods to use for first training set in holdout
        validation. Only used when validation_method='holdout'. If None, defaults to 80% of total time periods.
        step_size (Optional[int]): Number of time periods to move forward for each split in holdout validation.
                                   Only used when validation_method='holdout'.
                                   If None, defaults to (T - initial_window) // K.
        horizon (Optional[int]): Number of future time periods to predict (forecast horizon) in holdout validation.
                                 Only used when validation_method='holdout'. If None, defaults to step_size.
        max_window_size (Optional[int]): Maximum size of the window to consider in holdout validation.
                                         Only used when validation_method='holdout'. If None, use all data.

    Returns:
        MCNNMResults: A named tuple containing the results of the MC-NNM estimation.
    """
    # TODO: Implement this function
    return MCNNMResults()


def complete_matrix():
    pass  # TODO: Implement this function
