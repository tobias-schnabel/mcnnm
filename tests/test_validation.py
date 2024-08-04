import jax.numpy as jnp
import pytest
from mcnnm.estimate import estimate
from mcnnm.util import generate_data
import jax


def assert_close(true_value, estimated_value, tolerance, message):
    assert jnp.abs(true_value - estimated_value) < tolerance, (
        f"{message}: true={true_value:.4f}, estimated={estimated_value:.4f}, "
        f"difference={jnp.abs(true_value - estimated_value):.4f}"
    )


jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_disable_jit", True)


@pytest.mark.timeout(30)
def test_mcnnm_accuracy_cv_no_covariates(tolerance=0.1):
    nobs, nperiods = 100, 100
    data, true_params = generate_data(
        nobs=nobs,
        nperiods=nperiods,
        seed=42,
        unit_fe=True,
        time_fe=True,
        X_cov=False,
        Z_cov=False,
        V_cov=False,
    )

    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)

    results = estimate(Y, W, return_fixed_effects=True)

    assert_close(
        true_params["treatment_effect"], results.tau, tolerance, "Estimated treatment effect"
    )

    assert_close(
        jnp.mean(true_params["unit_fe"]),
        jnp.mean(results.gamma),
        tolerance,
        "Estimated unit fixed effects mean",
    )

    assert_close(
        jnp.mean(true_params["time_fe"]),
        jnp.mean(results.delta),
        tolerance,
        "Estimated time fixed effects mean",
    )

    assert_close(jnp.mean(true_params["L"]), jnp.mean(results.L), tolerance, "Estimated L mean")


@pytest.mark.timeout(30)
def test_mcnnm_accuracy_cv(tolerance=0.2):
    nobs, nperiods = 100, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)

    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)

    X, Z, V = jnp.array(true_params["X"]), jnp.array(true_params["Z"]), jnp.array(true_params["V"])

    results = estimate(
        Y, W, X=X, Z=Z, V=V, return_fixed_effects=True, return_covariate_coefficients=True
    )

    assert_close(
        true_params["treatment_effect"], results.tau, tolerance, "Estimated treatment effect"
    )

    assert_close(
        jnp.mean(true_params["unit_fe"]),
        jnp.mean(results.gamma),
        tolerance,
        "Estimated unit fixed effects mean",
    )

    assert_close(
        jnp.mean(true_params["time_fe"]),
        jnp.mean(results.delta),
        tolerance,
        "Estimated time fixed effects mean",
    )

    assert_close(
        jnp.mean(true_params["X_coef"]),
        jnp.mean(results.H[: X.shape[1], : Z.shape[1]]),
        tolerance,
        "Estimated X coefficients mean",
    )

    assert_close(
        jnp.mean(true_params["Z_coef"]),
        jnp.mean(results.H[: X.shape[1], : Z.shape[1]].T),
        tolerance,
        "Estimated Z coefficients mean",
    )

    assert_close(
        jnp.mean(true_params["V_coef"]),
        jnp.mean(results.beta),
        tolerance,
        "Estimated V coefficients mean",
    )


@pytest.mark.timeout(30)
def test_mcnnm_accuracy_no_covariates(tolerance=0.1):
    nobs, nperiods = 100, 100
    data, true_params = generate_data(
        nobs=nobs,
        nperiods=nperiods,
        seed=42,
        unit_fe=True,
        time_fe=True,
        X_cov=False,
        Z_cov=False,
        V_cov=False,
    )

    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)

    results = estimate(Y, W, return_fixed_effects=True, validation_method="holdout")

    assert_close(
        true_params["treatment_effect"], results.tau, tolerance, "Estimated treatment effect"
    )

    assert_close(
        jnp.mean(true_params["unit_fe"]),
        jnp.mean(results.gamma),
        tolerance,
        "Estimated unit fixed effects mean",
    )

    assert_close(
        jnp.mean(true_params["time_fe"]),
        jnp.mean(results.delta),
        tolerance,
        "Estimated time fixed effects mean",
    )

    assert_close(jnp.mean(true_params["L"]), jnp.mean(results.L), tolerance, "Estimated L mean")


@pytest.mark.timeout(30)
def test_mcnnm_accuracy(tolerance=0.2):
    nobs, nperiods = 100, 100
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)

    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)

    X, Z, V = jnp.array(true_params["X"]), jnp.array(true_params["Z"]), jnp.array(true_params["V"])

    results = estimate(
        Y,
        W,
        X=X,
        Z=Z,
        V=V,
        return_fixed_effects=True,
        return_covariate_coefficients=True,
        validation_method="holdout",
    )

    assert_close(
        true_params["treatment_effect"], results.tau, tolerance, "Estimated treatment effect"
    )

    assert_close(
        jnp.mean(true_params["unit_fe"]),
        jnp.mean(results.gamma),
        tolerance,
        "Estimated unit fixed effects mean",
    )

    assert_close(
        jnp.mean(true_params["time_fe"]),
        jnp.mean(results.delta),
        tolerance,
        "Estimated time fixed effects mean",
    )

    assert_close(
        jnp.mean(true_params["X_coef"]),
        jnp.mean(results.H[: X.shape[1], : Z.shape[1]]),
        tolerance,
        "Estimated X coefficients mean",
    )

    assert_close(
        jnp.mean(true_params["Z_coef"]),
        jnp.mean(results.H[: X.shape[1], : Z.shape[1]].T),
        tolerance,
        "Estimated Z coefficients mean",
    )

    assert_close(
        jnp.mean(true_params["V_coef"]),
        jnp.mean(results.beta),
        tolerance,
        "Estimated V coefficients mean",
    )
