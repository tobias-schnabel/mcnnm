import pytest
import jax
from mcnnm.estimate import estimate
from mcnnm.util import generate_data
import jax.numpy as jnp

jax.config.update(
    "jax_disable_jit", True
)  # Tests also pass with JIT enabled, but large number of tests means
# that compilation time is significant, so disabled for convenience


@pytest.mark.parametrize(
    "fixed_effects", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "covariates",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("Omega", [None, "autocorrelated"])
@pytest.mark.parametrize("validation_method", ["cv", "holdout"])
@pytest.mark.parametrize(
    "return_options",
    [
        (False, False, False, False, False, False),  # return nothing
        (True, False, False, False, False, False),  # return only the estimated treatment effect
        (True, True, False, False, False, False),  # return tau and optimal lambdas
        (False, False, False, True, False, False),  # return only completed Y
        (False, True, False, True, False, False),  # return completed Y and optimal lambdas
        (True, True, True, True, True, True),  # return everything
    ],
)
@pytest.mark.parametrize(
    "holdout_options",
    [
        (None, None, None, None),  # use defaults
        (8, 1, 1, None),  # specify initial_window, step_size, horizon
        (5, 2, 2, 8),  # specify all options including max_window_size
    ],
)
def test_mcnnm_estimation(
    fixed_effects, covariates, Omega, validation_method, return_options, holdout_options
):
    unit_fe, time_fe = fixed_effects
    X_cov, Z_cov, V_cov = covariates
    (
        return_tau,
        return_lambda,
        return_completed_L,
        return_completed_Y,
        return_fixed_effects,
        return_covariate_coefficients,
    ) = return_options
    initial_window, step_size, horizon, max_window_size = holdout_options

    nobs, nperiods = 10, 10
    autocorrelation = 0.5 if Omega == "autocorrelated" else 0.0
    data, true_params = generate_data(
        nobs=nobs,
        nperiods=nperiods,
        seed=42,
        unit_fe=unit_fe,
        time_fe=time_fe,
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

    results = estimate(
        Y,
        W,
        X=X,
        Z=Z,
        V=V,
        Omega=Omega,
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
        assert results.lambda_L is not None, "Chosen lambda_L is None"
        assert results.lambda_H is not None, "Chosen lambda_H is None"
        assert jnp.isfinite(results.lambda_L), "Chosen lambda_L is not finite"
        assert jnp.isfinite(results.lambda_H), "Chosen lambda_H is not finite"
    if return_completed_L:
        assert results.L.shape == Y.shape, "Completed L matrix has incorrect shape"
        assert jnp.all(jnp.isfinite(results.L)), "Completed L matrix contains non-finite values"
    if return_completed_Y:
        assert results.Y_completed.shape == Y.shape, "Completed Y matrix has incorrect shape"
        assert jnp.all(
            jnp.isfinite(results.Y_completed)
        ), "Completed Y matrix contains non-finite values"
    if return_fixed_effects:
        assert jnp.all(
            jnp.isfinite(results.gamma)
        ), "Estimated unit fixed effects contain non-finite values"
        assert jnp.all(
            jnp.isfinite(results.delta)
        ), "Estimated time fixed effects contain non-finite values"
    if return_covariate_coefficients:
        assert jnp.all(
            jnp.isfinite(results.beta)
        ), "Estimated V coefficients contain non-finite values"
        assert jnp.all(jnp.isfinite(results.H)), "Estimated H matrix contains non-finite values"

    # Additional checks for holdout validation
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
