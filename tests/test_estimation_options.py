import pytest
from mcnnm.estimate import estimate
from mcnnm.util import generate_data
import jax.numpy as jnp

# jax.config.update("jax_disable_jit", True)


@pytest.mark.parametrize("use_unit_fe", [False, True])
@pytest.mark.parametrize("use_time_fe", [False, True])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("Omega", [None, "autocorrelated"])
@pytest.mark.parametrize("validation_method", ["cv", "holdout"])
@pytest.mark.parametrize(
    "return_options",
    [
        "tau_only",
        "lambdas_only",
        "completed_matrices",
        "fixed_effects",
        "covariate_coefficients",
        "all",
    ],
)
@pytest.mark.parametrize("holdout_options", ["default", "custom_window", "all_custom"])
def test_mcnnm_estimation(
    use_unit_fe,
    use_time_fe,
    X_cov,
    Z_cov,
    V_cov,
    Omega,
    validation_method,
    return_options,
    holdout_options,
):
    nobs, nperiods = 10, 10
    autocorrelation = 0.5 if Omega == "autocorrelated" else 0.0
    data, true_params = generate_data(
        nobs=nobs,
        nperiods=nperiods,
        seed=42,
        unit_fe=use_unit_fe,
        time_fe=use_time_fe,
        X_cov=X_cov,
        Z_cov=Z_cov,
        V_cov=V_cov,
        autocorrelation=autocorrelation,
    )

    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)
    X = jnp.array(true_params["X"]) if X_cov else None
    Z = jnp.array(true_params["Z"]) if Z_cov else None
    V = jnp.array(true_params["V"]) if V_cov else None

    if Omega == "autocorrelated":
        Omega = jnp.eye(nperiods) * autocorrelation + jnp.tri(nperiods, k=-1) * (
            1 - autocorrelation
        )
    else:
        Omega = None

    return_tau = "tau" in return_options or return_options == "all"
    return_lambda = "lambda" in return_options or return_options == "all"
    return_completed_L = "completed_matrices" in return_options or return_options == "all"
    return_completed_Y = "completed_matrices" in return_options or return_options == "all"
    return_fixed_effects = "fixed_effects" in return_options or return_options == "all"
    return_covariate_coefficients = (
        "covariate_coefficients" in return_options or return_options == "all"
    )

    if holdout_options == "default":
        initial_window, step_size, horizon, max_window_size = None, None, None, None
    elif holdout_options == "custom_window":
        initial_window, step_size, horizon, max_window_size = 8, 1, 1, None
    else:  # all_custom
        initial_window, step_size, horizon, max_window_size = 5, 2, 2, 8

    results = estimate(
        Y,
        W,
        X=X,
        Z=Z,
        V=V,
        Omega=Omega,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        validation_method=validation_method,
        return_tau=return_tau,
        return_lambda=return_lambda,
        return_completed_L=return_completed_L,
        return_completed_Y=return_completed_Y,
        return_fixed_effects=return_fixed_effects,
        return_covariate_coefficients=return_covariate_coefficients,
        max_iter=100,
        tol=1e-4,
        n_lambda_L=3,
        n_lambda_H=3,
        K=2,
        initial_window=initial_window,
        step_size=step_size,
        horizon=horizon,
        max_window_size=max_window_size,
    )

    if return_tau:
        assert jnp.isfinite(results.tau), "Estimated treatment effect is not finite"
    if return_lambda:
        assert results.lambda_L is not None and jnp.isfinite(
            results.lambda_L
        ), "Chosen lambda_L is not valid"
        assert results.lambda_H is not None and jnp.isfinite(
            results.lambda_H
        ), "Chosen lambda_H is not valid"
    if return_completed_L:
        assert results.L.shape == Y.shape, "Completed L matrix has incorrect shape"
        assert jnp.all(jnp.isfinite(results.L)), "Completed L matrix contains non-finite values"
    if return_completed_Y:
        assert results.Y_completed.shape == Y.shape, "Completed Y matrix has incorrect shape"
        assert jnp.all(
            jnp.isfinite(results.Y_completed)
        ), "Completed Y matrix contains non-finite values"
    if return_fixed_effects:
        if use_unit_fe:
            assert results.gamma is not None and jnp.all(
                jnp.isfinite(results.gamma)
            ), "Estimated unit fixed effects are not valid"
        else:
            # change this from asserting that they're none to asserting that they're all zero
            jnp.allclose(
                results.gamma, jnp.zeros_like(results.gamma)
            ), "Unit fixed effects should be zero when not used"

        if use_time_fe:
            assert results.delta is not None and jnp.all(
                jnp.isfinite(results.delta)
            ), "Estimated time fixed effects are not valid"
        else:
            # change this from asserting that they're none to asserting that they're all zero
            jnp.allclose(
                results.delta, jnp.zeros_like(results.delta)
            ), "Time fixed effects should be zero when not used"
    if return_covariate_coefficients:
        if V_cov:
            assert results.beta is not None and jnp.all(
                jnp.isfinite(results.beta)
            ), "Estimated V coefficients are not valid"
        else:
            assert results.beta is None or jnp.all(
                results.beta == 0
            ), "V coefficients should be None or zero when not used"
        assert results.H is not None and jnp.all(
            jnp.isfinite(results.H)
        ), "Estimated H matrix is not valid"

    if validation_method == "holdout":
        if initial_window is not None:
            assert initial_window <= nperiods, "initial_window is larger than the number of periods"
        if step_size is not None:
            assert step_size > 0, "step_size must be positive"
        if horizon is not None:
            assert horizon > 0, "horizon must be positive"
        if max_window_size is not None:
            assert (
                max_window_size <= nperiods
            ), "max_window_size is larger than the number of periods"
