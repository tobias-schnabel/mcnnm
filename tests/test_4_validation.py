import pytest
import jax.numpy as jnp

from mcnnm.utils import (
    generate_data,
    generate_time_based_validate_defaults,
)
from mcnnm.validation import cross_validate, holdout_validate


@pytest.mark.parametrize("N, T", [(10, 10)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("noise_scale", [0.5, 1.0, 2.0])
def test_cross_validate(N, T, fe_params, X_cov, Z_cov, V_cov, noise_scale):
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
    )

    opt_lambda_L, opt_lambda_H = cross_validate(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega_inv=None,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        num_lam=6,
        max_iter=1_000,
        tol=1e-1,
        K=5,
    )

    assert not jnp.isnan(opt_lambda_L)
    assert jnp.isfinite(opt_lambda_L)
    assert opt_lambda_L >= 0
    assert not jnp.isnan(opt_lambda_H)
    assert jnp.isfinite(opt_lambda_H)
    assert opt_lambda_H >= 0


@pytest.mark.parametrize("N, T", [(10, 10)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("noise_scale", [0.5, 1.0, 2.0])
def test_holdout_validate(N, T, fe_params, X_cov, Z_cov, V_cov, noise_scale):
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
    )

    initial_window, step_size, horizon, K = generate_time_based_validate_defaults(Y)

    opt_lambda_L, opt_lambda_H = holdout_validate(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega_inv=None,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        num_lam=6,
        initial_window=initial_window,
        step_size=step_size,
        horizon=horizon,
        max_iter=1_000,
        tol=1e-1,
        K=K,
    )

    assert not jnp.isnan(opt_lambda_L)
    assert jnp.isfinite(opt_lambda_L)
    assert opt_lambda_L >= 0
    assert not jnp.isnan(opt_lambda_H)
    assert jnp.isfinite(opt_lambda_H)
    assert opt_lambda_H >= 0
