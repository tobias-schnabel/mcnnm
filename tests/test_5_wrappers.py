# mypy: ignore-errors
# type: ignore
import jax.numpy as jnp
import pytest
import jax

from mcnnm import complete_matrix
from mcnnm.utils import generate_data
from mcnnm.core import initialize_fixed_effects_and_H, initialize_matrices
from mcnnm.wrappers import compute_treatment_effect, estimate
from mcnnm.types import Array

key = jax.random.PRNGKey(2024)


@pytest.mark.parametrize("N, T", [(15, 30)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("noise_scale", [0.5, 1.0, 2.0])
def test_compute_treatment_effect(N, T, fe_params, X_cov, Z_cov, V_cov, noise_scale):
    use_unit_fe, use_time_fe = fe_params
    Y, W, X, Z, V, true_params = generate_data(
        nobs=N,
        nperiods=T,
        unit_fe=use_unit_fe,
        time_fe=use_time_fe,
        X_cov=X_cov,
        Z_cov=Z_cov,
        V_cov=V_cov,
        seed=2024,
        noise_scale=noise_scale,
        treatment_effect=5,
    )

    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, False, False, verbose=False)
    )

    treatment_effect = compute_treatment_effect(
        Y, W, L, X_tilde, Z_tilde, V, H_tilde, gamma, delta, beta, use_unit_fe, use_time_fe
    )

    assert isinstance(treatment_effect, float)
    assert not jnp.isnan(treatment_effect)
    assert jnp.isfinite(treatment_effect)
    assert treatment_effect != 0
    # No point in checking the exact value of the treatment effect, as it is computed off the initialization.


@pytest.mark.parametrize("N, T", [(12, 30)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
# @pytest.mark.parametrize("noise_scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("validation_method", ["cv", "holdout"])
# @pytest.mark.parametrize("K", [5, 10])
@pytest.mark.parametrize("autocorrelation", [0.0, 0.5])
@pytest.mark.parametrize("use_max_window_size", [False, True])
@pytest.mark.parametrize("use_custom_lambda", [False, True])
def test_estimate(
    N,
    T,
    fe_params,
    X_cov,
    Z_cov,
    V_cov,
    validation_method,
    autocorrelation,
    use_max_window_size,
    use_custom_lambda,
):
    use_unit_fe, use_time_fe = fe_params
    K = 3  # Use 3 folds for faster testing

    # Generate data
    Y, W, X, Z, V, true_params = generate_data(
        nobs=N,
        nperiods=T,
        unit_fe=use_unit_fe,
        time_fe=use_time_fe,
        X_cov=X_cov,
        Z_cov=Z_cov,
        V_cov=V_cov,
        seed=2024,
        noise_scale=1.0,
        autocorrelation=autocorrelation,
        assignment_mechanism="block",
        treated_fraction=0.5,
    )

    if autocorrelation > 0:
        Omega = jnp.eye(T) * autocorrelation + jnp.tri(T, k=-1) * (
            1 - autocorrelation
        )  # AR(1) covariance matrix
    else:
        Omega = None

    if validation_method == "holdout":
        initial_window = 16
        step_size = 5
        horizon = 1

    else:
        initial_window, step_size, horizon = None, None, None

    if use_max_window_size:
        max_window = 28
    else:
        max_window = None

    if use_custom_lambda:
        lambda_L = jnp.array([0.01])
        lambda_H = jnp.array([0.01])
    else:
        lambda_L = None
        lambda_H = None

    # Run estimation
    results = estimate(
        Y=Y,
        W=W,
        X=X if X_cov else None,
        Z=Z if Z_cov else None,
        V=V if V_cov else None,
        Omega=Omega,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        lambda_L=lambda_L,
        lambda_H=lambda_H,
        validation_method=validation_method,  # type: ignore
        K=K,  # Use 2 folds for faster testing
        n_lambda=3,  # Use 3 lambda values for faster testing
        max_iter=100,  # Reduce max iterations for faster testing
        tol=1e-3,  # Increase tolerance for faster convergence
        initial_window=initial_window,
        step_size=step_size,
        horizon=horizon,
        max_window_size=max_window,
    )

    # Basic checks
    assert isinstance(results.tau, float)
    assert isinstance(results.Y_completed, Array)
    assert results.Y_completed.shape == (N, T)

    # Check if estimated ATE is reasonably close to true ATE
    # Use a large tolerance due to small sample size and potential noise
    # print(
    #     f"Absolute difference between estimated and true tau: "
    #     f"{jnp.abs(results.tau - true_params['treatment_effect']):.2f}"
    # )

    # # Check if Y_completed is close to true Y for observed entries
    # mse = jnp.mean((results.Y_completed[W == 0] - Y[W == 0]) ** 2)
    # assert mse < noise_scale**2  # MSE should be less than noise variance

    # Check fixed effects
    if use_unit_fe:
        assert results.gamma is not None
        assert results.gamma.shape == (N,)
    else:
        assert jnp.allclose(results.gamma, jnp.zeros_like(results.gamma))

    if use_time_fe:
        assert results.delta is not None
        assert results.delta.shape == (T,)
    else:
        assert jnp.allclose(results.delta, jnp.zeros_like(results.delta))

    # Check other attributes
    assert results.L is not None
    assert results.H is not None
    assert results.lambda_L is not None
    assert results.lambda_H is not None

    # Check for NaN and infinite values in results
    assert not jnp.any(jnp.isnan(results.L)), "L contains NaN values"
    assert not jnp.any(jnp.isinf(results.L)), "L contains infinite values"

    assert not jnp.any(jnp.isnan(results.H)), "H contains NaN values"
    assert not jnp.any(jnp.isinf(results.H)), "H contains infinite values"

    assert not jnp.any(jnp.isnan(results.Y_completed)), "Y_completed contains NaN values"
    assert not jnp.any(jnp.isinf(results.Y_completed)), "Y_completed contains infinite values"
    assert not jnp.allclose(
        results.Y_completed, jnp.zeros_like(results.Y_completed)
    ), "Y_completed is all zeros"

    assert not jnp.isnan(results.tau), "tau contains NaN values"
    assert not jnp.isinf(results.tau), "tau contains infinite values"

    assert not jnp.any(jnp.isnan(results.gamma)), "gamma contains NaN values"
    assert not jnp.any(jnp.isinf(results.gamma)), "gamma contains infinite values"

    assert not jnp.any(jnp.isnan(results.delta)), "delta contains NaN values"
    assert not jnp.any(jnp.isinf(results.delta)), "delta contains infinite values"

    assert not jnp.any(jnp.isnan(results.beta)), "beta contains NaN values"
    assert not jnp.any(jnp.isinf(results.beta)), "beta contains infinite values"

    assert not jnp.isnan(results.lambda_L), "lambda_L contains NaN values"
    assert not jnp.isinf(results.lambda_L), "lambda_L contains infinite values"
    assert results.lambda_L >= 0, "lambda_L is negative"

    assert not jnp.isnan(results.lambda_H), "lambda_H contains NaN values"
    assert not jnp.isinf(results.lambda_H), "lambda_H contains infinite values"
    assert results.lambda_H >= 0, "lambda_H is negative"


def test_complete_matrix():
    N, T = 10, 100

    # Generate data
    Y, W, X, Z, V, true_params = generate_data(
        nobs=N,
        nperiods=T,
        unit_fe=True,
        time_fe=True,
        X_cov=True,
        Z_cov=True,
        V_cov=True,
        seed=2024,
        noise_scale=1.0,
        assignment_mechanism="block",
        treated_fraction=0.2,
    )

    # Run estimation
    Y_completed, opt_lambda_L, opt_lambda_H = complete_matrix(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega=None,
        use_unit_fe=True,
        use_time_fe=True,
        lambda_L=None,
        lambda_H=None,
        validation_method="cv",
        K=2,  # Use 2 folds for faster testing
        n_lambda=3,  # Use 3 lambda values for faster testing
        max_iter=1_000,  # Reduce max iterations for faster testing
        tol=1e-1,  # Increase tolerance for faster convergence
    )

    # Basic checks
    assert isinstance(Y_completed, Array)
    assert Y_completed.shape == (N, T)

    # Check if estimated ATE is reasonably close to true ATE
    # Use a large tolerance due to small sample size and potential noise
    # print(
    #     f"Absolute difference between estimated and true tau: "
    #     f"{jnp.abs(results.tau - true_params['treatment_effect']):.2f}"
    # )

    # # Check if Y_completed is close to true Y for observed entries
    # mse = jnp.mean((Y_completed[W == 0] - Y[W == 0]) ** 2)
    # assert mse < noise_scale**2  # MSE should be less than noise variance

    # Check other attributes
    assert Y_completed is not None
    assert opt_lambda_L is not None
    assert opt_lambda_H is not None

    # Check for NaN and infinite values in results
    assert not jnp.any(jnp.isnan(Y_completed)), "Y_completed contains NaN values"
    assert not jnp.any(jnp.isinf(Y_completed)), "Y_completed contains infinite values"
    assert not jnp.allclose(Y_completed, jnp.zeros_like(Y_completed)), "Y_completed is all zeros"

    assert not jnp.isnan(opt_lambda_L), "lambda_L contains NaN values"
    assert not jnp.isinf(opt_lambda_L), "lambda_L contains infinite values"
    assert opt_lambda_L >= 0, "lambda_L is negative"

    assert not jnp.isnan(opt_lambda_H), "lambda_H contains NaN values"
    assert not jnp.isinf(opt_lambda_H), "lambda_H contains infinite values"
    assert opt_lambda_H >= 0, "lambda_H is negative"
