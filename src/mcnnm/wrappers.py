from .core_utils import is_positive_definite
from .types import Array, Scalar
from typing import NamedTuple, Optional, Literal, cast, Tuple
import jax.numpy as jnp

from .core import compute_Y_hat, initialize_matrices, initialize_fixed_effects_and_H
from .validation import cross_validate, holdout_validate, final_fit
from .utils import validate_holdout_config


def compute_treatment_effect(
    Y: Array,
    W: Array,
    L: Array,
    X_tilde: Array,
    Z_tilde: Array,
    V: Array,
    H_tilde: Array,
    gamma: Array,
    delta: Array,
    beta: Array,
    use_unit_fe: bool,
    use_time_fe: bool,
) -> Scalar:
    """
    Compute the average treatment effect using the MC-NNM model estimates.
    Thin wrapper around the `compute_Y_hat` function.

    This function calculates the difference between the observed outcomes and the
    completed (counterfactual) outcomes for treated units, then averages this
    difference to estimate the average treatment effect.

    Args:
        Y (Array): The observed outcome matrix.
        W (Array): The binary treatment matrix.
        L (Array): The estimated low-rank matrix.
        X_tilde (Array): The augmented unit-specific covariates matrix.
        Z_tilde (Array): The augmented time-specific covariates matrix.
        V (Array): The unit-time specific covariates tensor.
        H_tilde (Array): The augmented covariate coefficient matrix.
        gamma (Array): The estimated unit fixed effects.
        delta (Array): The estimated time fixed effects.
        beta (Array): The estimated unit-time specific covariate coefficients.
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.

    Returns:
        Scalar: The estimated average treatment effect.
    """
    Y_completed = compute_Y_hat(
        L, X_tilde, Z_tilde, V, H_tilde, gamma, delta, beta, use_unit_fe, use_time_fe
    )
    W = 1 - W
    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    tau = cast(Scalar, tau.item())  # type: ignore
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
    Mask: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
    use_unit_fe: bool = True,
    use_time_fe: bool = True,
    lambda_L: Optional[Scalar] = None,
    lambda_H: Optional[Scalar] = None,
    n_lambda: Optional[int] = 10,
    max_iter: Optional[Scalar] = 1e4,
    tol: Optional[Scalar] = 1e-4,
    validation_method: Literal["cv", "holdout"] = "cv",
    K: int = 5,
    initial_window: Optional[int] = None,
    step_size: Optional[int] = None,
    horizon: Optional[int] = None,
    max_window_size: Optional[int] = None,
) -> MCNNMResults:
    """
    Estimate the Matrix Completion with Nuclear Norm Minimization (MC-NNM) model.

    This function performs the complete estimation process for the MC-NNM model, including
    parameter selection, model fitting, and treatment effect computation. It handles various
    input configurations and validation methods to provide a comprehensive analysis of the
    given data.

    Detailed Process:

    1. Input Validation and Initialization:
       - Checks the dimensions of input matrices Y and W.
       - Initializes the inverse of the Omega matrix (Omega_inv) if provided, otherwise uses an identity matrix.
       - Validates that Omega_inv is positive definite.
       - Initializes matrices L, X_tilde, Z_tilde, and V using the initialize_matrices function.
       - Initializes fixed effects and H matrix using the initialize_fixed_effects_and_H function.

    2. Lambda Selection:
       - If lambda_L or lambda_H are not provided, performs validation to select optimal values:

         a. Cross-validation (CV):
            - Uses the cross_validate function to find optimal lambda values.
            - Performs K-fold cross-validation on the data.
         b. Holdout validation:
            - Uses the holdout_validate function to find optimal lambda values.
            - Requires initial_window, step_size, and horizon parameters.
            - Generates default values for these parameters if not provided.
            - Validates the holdout configuration using validate_holdout_config function.

       - If lambda_L and lambda_H are provided, uses these values directly.

    3. Model Fitting:
       - Calls the final_fit function to fit the MC-NNM model using the selected lambda values.
       - Uses a warm-start approach, fitting the model along a path of lambda values.

    4. Results Computation:
       - Computes the completed outcome matrix using the compute_Y_hat function.
       - Calculates the average treatment effect using the compute_treatment_effect function.

    5. Return Results:
       - Returns an MCNNMResults object containing the estimated parameters, completed matrix, and treatment effect.

    Parameters:
        Y (Array): The observed outcome matrix of shape (N, T), where N is the number of units
                   and T is the number of time periods.
        Mask (Array): The treatment assignment matrix of shape (N, T). Binary values where 1 indicates
                   treatment and 0 indicates control.
        X (Array, optional): Unit-specific covariates of shape (N, P), where P is the number of
                             unit-specific covariates.
        Z (Array, optional): Time-specific covariates of shape (T, Q), where Q is the number of
                             time-specific covariates.
        V (Array, optional): Unit-time specific covariates of shape (N, T, J), where J is the
                             number of unit-time specific covariates.
        Omega (Array, optional): The covariance matrix of shape (T, T) representing the time
                                 series correlation structure. If not provided, an identity
                                 matrix is used.
        use_unit_fe (bool): Whether to include unit fixed effects in the model. Default is True.
        use_time_fe (bool): Whether to include time fixed effects in the model. Default is True.
        lambda_L (Scalar, optional): The regularization parameter for the nuclear norm of L.
                                     If not provided, it will be selected via validation.
        lambda_H (Scalar, optional): The regularization parameter for the nuclear norm of H.
                                     If not provided, it will be selected via validation.
        n_lambda (int): The number of lambda values to consider in the validation grid. Default is 10.
        max_iter (Scalar, optional): Maximum number of iterations for the optimization algorithm. Default is 10,000.
        tol (Scalar, optional): Tolerance for convergence of the optimization algorithm. Default is 1e-4.
        validation_method (str): The method to use for selecting lambda values. Must be either
                                 'cv' for cross-validation or 'holdout' for holdout validation.
                                 Default is 'cv'.
        K (int): The number of folds to use in cross-validation. Default is 5.
        initial_window (int, optional): The size of the initial window for holdout validation.
                                        Required if validation_method is 'holdout'.
        step_size (int, optional): The step size for moving the window in holdout validation.
                                   Required if validation_method is 'holdout'.
        horizon (int, optional): The size of the holdout horizon in holdout validation.
                                 Required if validation_method is 'holdout'.
        max_window_size (int, optional): The maximum window size for holdout validation.
                                         If not provided, no maximum is imposed.

    Returns:
        MCNNMResults: An object containing the following attributes:
            - tau: The estimated average treatment effect.
            - lambda_L: The final lambda_L value used in the model.
            - lambda_H: The final lambda_H value used in the model.
            - L: The estimated low-rank matrix.
            - Y_completed: The completed outcome matrix.
            - gamma: The estimated unit fixed effects (if use_unit_fe is True).
            - delta: The estimated time fixed effects (if use_time_fe is True).
            - beta: The estimated coefficients for covariates.
            - H: The estimated interactive fixed effects matrix.

    Raises:
        ValueError: If Omega_inv is not positive definite, if an invalid validation method is
                    specified, or if required parameters for holdout validation are missing.

    Notes:
        - The function uses JAX for numerical computations, which allows for automatic
          differentiation and potential GPU acceleration.
        - The estimation process involves several steps of initialization, validation, and
          optimization, which can be computationally intensive for large datasets.
        - The choice of validation method and associated parameters can significantly affect
          the final estimates and computational time.
        - The function handles missing data in the outcome matrix Y through the treatment
          assignment matrix W.
        - The implementation supports various types of fixed effects and covariates, allowing
          for flexible model specifications.

    Example:
        >>> from mcnnm.wrappers import estimate
        >>> from mcnnm.utils import generate_data
        >>> N, T = 100, 50  # Number of units and time periods
        >>> noise_scale = 1.0 # Scale of the noise in the data
        >>> Y, W, X, Z, V, true_params = generate_data(
        ...     nobs=N,
        ...     nperiods=T,
        ...     unit_fe=True,
        ...     time_fe=True,
        ...     X_cov=True,
        ...     Z_cov=True,
        ...     V_cov=True,
        ...     seed=2024,
        ...     noise_scale=noise_scale,
        ... )
        >>> results = estimate(Y, W, X, Z, V, validation_method='cv', K=5)
        >>> print(results.tau)  # Print the estimated average treatment effect
        >>> print(f"True ATE: {true_params['treatment_effect']}")  # Compare with the true ATE

    See Also:
        - :func:`.validation.cross_validate`: Function used for cross-validation.
        - :func:`validation.holdout_validate`: Function used for holdout validation.
        - :func:`validation.final_fit`: Function used for the final model fitting.
        - :func:`.core.compute_Y_hat`: Function used to compute the completed outcome matrix.
        - :func:`.core.compute_treatment_effect`: Function used to compute the average treatment effect.
    """
    W = 1 - Mask  # get inverse mask where 1 indicates no treatment

    # Convert max_iter to int
    if max_iter is not None:  # pragma: no cover
        try:
            max_iter = int(max_iter)
        except ValueError:  # pragma: no cover
            raise ValueError(f"max_iter must be convertible to int, got {max_iter}")
    else:  # pragma: no cover
        max_iter = 10_000

    # Convert n_lambda to int
    if n_lambda is not None:  # pragma: no cover
        try:  # pragma: no cover
            n_lambda = int(n_lambda)
        except ValueError:
            raise ValueError(f"n_lambda must be convertible to int, got {n_lambda}")
    else:  # pragma: no cover
        n_lambda = 10

    # Convert lambdas to jnp arrays if specified
    if lambda_L is not None:
        lambda_L = jnp.array([lambda_L])
    if lambda_H is not None:
        lambda_H = jnp.array([lambda_H])

    N, T = Y.shape

    if Omega is None:
        Omega_inv = jnp.eye(T)  # default TxT identity matrix
    else:
        Omega_inv = jnp.linalg.inv(Omega)  # invert Omega if specified

    if not is_positive_definite(Omega_inv):  # pragma: no cover
        raise ValueError("Omega_inv must be a positive definite matrix.")

    # Initialize matrices to zero of correct dimensions
    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    # Initialize fixed effects and H matrix
    (
        gamma_init,
        delta_init,
        beta_init,
        H_tilde_init,
        T_mat_init,
        in_prod_T_init,
        in_prod_init,
        lambda_L_max,
        lambda_H_max,
    ) = initialize_fixed_effects_and_H(
        Y, L, X_tilde, Z_tilde, V, W, use_unit_fe, use_time_fe, verbose=False
    )

    # Select lambda values via validation
    if lambda_L is None or lambda_H is None:
        if validation_method == "cv":
            opt_lambda_L, opt_lambda_H, lambda_L_opt_range, lambda_H_opt_range = cross_validate(
                Y=Y,
                X=X,  # type: ignore[arg-type]
                Z=Z,  # type: ignore[arg-type]
                V=V,  # type: ignore[arg-type]
                W=W,
                Omega_inv=Omega_inv,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                num_lam=n_lambda,
                max_iter=max_iter,
                tol=tol,  # type: ignore[arg-type]
                K=K,
            )
        elif validation_method == "holdout":
            if initial_window is None or step_size is None or horizon is None:  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    "Holdout validation requires initial_window, step_size, and horizon."
                )

            initial_window, step_size, horizon, K, max_window_size = validate_holdout_config(
                initial_window,  # type: ignore
                step_size,  # type: ignore
                horizon,  # type: ignore
                K,
                max_window_size,
                T,
            )
            opt_lambda_L, opt_lambda_H, lambda_L_opt_range, lambda_H_opt_range = holdout_validate(
                Y=Y,
                X=X,  # type: ignore[arg-type]
                Z=Z,  # type: ignore[arg-type]
                V=V,  # type: ignore[arg-type]
                W=W,
                Omega_inv=Omega_inv,
                use_unit_fe=use_unit_fe,
                use_time_fe=use_time_fe,
                num_lam=n_lambda,
                initial_window=initial_window,
                step_size=step_size,
                horizon=horizon,
                K=K,
                max_window_size=max_window_size,
                max_iter=max_iter,
                tol=tol,  # type: ignore[arg-type]
            )
        else:
            raise ValueError(
                "Invalid validation method. Must be 'cv' or 'holdout'."
            )  # pragma: no cover
    else:
        opt_lambda_L = jnp.array(lambda_L)
        opt_lambda_H = jnp.array(lambda_H)
        lambda_L_opt_range, lambda_H_opt_range = opt_lambda_L, opt_lambda_H

    # Fit the final model
    L_final, H_final, in_prod_final, gamma_final, delta_final, beta_final, loss_final = final_fit(
        Y=Y,
        W=W,
        X=X,  # type: ignore[arg-type]
        Z=Z,  # type: ignore[arg-type]
        V=V,  # type: ignore[arg-type]
        Omega_inv=Omega_inv,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        best_lambda_L=opt_lambda_L,
        best_lambda_H=opt_lambda_H,
        lambda_L_opt_range=lambda_L_opt_range,
        lambda_H_opt_range=lambda_H_opt_range,
    )

    # Compute the completed outcome matrix
    Y_completed = compute_Y_hat(
        L_final,
        X_tilde,
        Z_tilde,
        V,
        H_final,
        gamma_final,
        delta_final,
        beta_final,
        use_unit_fe,
        use_time_fe,
    )

    # Compute the average treatment effect
    tau = compute_treatment_effect(
        Y=Y,
        W=W,
        L=L_final,
        X_tilde=X_tilde,
        Z_tilde=Z_tilde,
        V=V,  # type: ignore[arg-type]
        H_tilde=H_final,
        gamma=gamma_final,
        delta=delta_final,
        beta=beta_final,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
    )

    return MCNNMResults(
        tau=tau,
        lambda_L=opt_lambda_L,
        lambda_H=opt_lambda_H,
        L=L_final,
        Y_completed=Y_completed,
        gamma=gamma_final,
        delta=delta_final,
        beta=beta_final,
        H=H_final,
    )


def complete_matrix(
    Y: Array,
    Mask: Array,
    X: Optional[Array] = None,
    Z: Optional[Array] = None,
    V: Optional[Array] = None,
    Omega: Optional[Array] = None,
    use_unit_fe: bool = True,
    use_time_fe: bool = True,
    lambda_L: Optional[Scalar] = None,
    lambda_H: Optional[Scalar] = None,
    n_lambda: int = 10,
    max_iter: Optional[int] = 10_000,
    tol: Optional[Scalar] = 1e-4,
    validation_method: Literal["cv", "holdout"] = "cv",
    K: int = 5,
    initial_window: Optional[int] = None,
    step_size: Optional[int] = None,
    horizon: Optional[int] = None,
    max_window_size: Optional[int] = None,
) -> Tuple[Array, Scalar, Scalar]:
    """
    Complete a matrix using the Matrix Completion with Nuclear Norm Minimization (MC-NNM) model.

    This function is a thin wrapper around the `estimate` function, focusing on matrix completion.
    It performs the estimation process for the MC-NNM model and returns the completed matrix
    along with the regularization parameters used.

    For a detailed description of the estimation process, input parameters, and their meanings,
    please refer to the documentation of the `estimate` function.

    Returns:
        tuple: A tuple containing:
            - Y_completed (Array): The completed outcome matrix.
            - lambda_L (Scalar): The final lambda_L value used in the model.
            - lambda_H (Scalar): The final lambda_H value used in the model.

    Note:
        This function uses the same parameters as the `estimate` function. For a comprehensive
        explanation of each parameter, including optional covariates, fixed effects, validation
        methods, and other configuration options, please consult the `estimate` function's
        documentation.

    Example:
        >>> Y_completed, lambda_L, lambda_H = complete_matrix(Y, W, X=X, Z=Z, V=V)
        >>> print(Y_completed.shape)  # Print the shape of the completed matrix
        >>> print(f"Lambda L: {lambda_L}, Lambda H: {lambda_H}")  # Print the regularization parameters

    See Also:
        estimate: The main function performing the complete MC-NNM estimation process.
    """

    results = estimate(  # pragma: no cover
        Y=Y,
        Mask=Mask,
        X=X,
        Z=Z,
        V=V,
        Omega=Omega,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        lambda_L=lambda_L,
        lambda_H=lambda_H,
        n_lambda=n_lambda,
        max_iter=max_iter,
        tol=tol,
        validation_method=validation_method,
        K=K,
        initial_window=initial_window,
        step_size=step_size,
        horizon=horizon,
        max_window_size=max_window_size,
    )

    return (results.Y_completed, results.lambda_L, results.lambda_H)  # type: ignore
