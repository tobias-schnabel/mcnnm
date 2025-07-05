import jax.numpy as jnp
import pytest

from mcnnm.utils import (
    generate_data,
    generate_holdout_val_defaults,
)
from mcnnm.validation import cross_validate, final_fit, holdout_validate


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

    opt_lambda_L, opt_lambda_H, lambda_L_opt_range, lambda_H_opt_range = cross_validate(
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

    assert opt_lambda_L in lambda_L_opt_range
    assert not jnp.any(jnp.isnan(lambda_L_opt_range))
    assert jnp.all(jnp.isfinite(lambda_L_opt_range))
    assert jnp.all(lambda_L_opt_range >= 0)

    assert opt_lambda_H in lambda_H_opt_range
    assert not jnp.any(jnp.isnan(lambda_H_opt_range))
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))
    assert jnp.all(lambda_H_opt_range >= 0)
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))

    L_final, H_final, in_prod_final, gamma_final, delta_final, beta_final, loss_final = final_fit(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega_inv=None,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        best_lambda_L=opt_lambda_L,
        best_lambda_H=opt_lambda_H,
        lambda_L_opt_range=lambda_L_opt_range,
        lambda_H_opt_range=lambda_H_opt_range,
    )
    assert not jnp.any(jnp.isnan(L_final)), "L_final contains NaN values"
    assert not jnp.any(jnp.isinf(L_final)), "L_final contains infinite values"

    assert not jnp.any(jnp.isnan(H_final)), "H_final contains NaN values"
    assert not jnp.any(jnp.isinf(H_final)), "H_final contains infinite values"

    assert not jnp.any(jnp.isnan(in_prod_final)), "in_prod_final contains NaN values"
    assert not jnp.any(jnp.isinf(in_prod_final)), "in_prod_final contains infinite values"

    assert not jnp.any(jnp.isnan(gamma_final)), "gamma_final contains NaN values"
    assert not jnp.any(jnp.isinf(gamma_final)), "gamma_final contains infinite values"

    assert not jnp.any(jnp.isnan(delta_final)), "delta_final contains NaN values"
    assert not jnp.any(jnp.isinf(delta_final)), "delta_final contains infinite values"

    assert not jnp.any(jnp.isnan(beta_final)), "beta_final contains NaN values"
    assert not jnp.any(jnp.isinf(beta_final)), "beta_final contains infinite values"

    assert not jnp.isnan(loss_final), "loss_final contains NaN values"
    assert not jnp.isinf(loss_final), "loss_final contains infinite values"
    assert loss_final >= 0


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

    initial_window, step_size, horizon, K = generate_holdout_val_defaults(Y)

    opt_lambda_L, opt_lambda_H, lambda_L_opt_range, lambda_H_opt_range = holdout_validate(
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

    assert opt_lambda_L in lambda_L_opt_range
    assert not jnp.any(jnp.isnan(lambda_L_opt_range))
    assert jnp.all(jnp.isfinite(lambda_L_opt_range))
    assert jnp.all(lambda_L_opt_range >= 0)

    assert opt_lambda_H in lambda_H_opt_range
    assert not jnp.any(jnp.isnan(lambda_H_opt_range))
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))
    assert jnp.all(lambda_H_opt_range >= 0)
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))

    L_final, H_final, in_prod_final, gamma_final, delta_final, beta_final, loss_final = final_fit(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega_inv=None,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        best_lambda_L=opt_lambda_L,
        best_lambda_H=opt_lambda_H,
        lambda_L_opt_range=lambda_L_opt_range,
        lambda_H_opt_range=lambda_H_opt_range,
    )
    assert not jnp.any(jnp.isnan(L_final)), "L_final contains NaN values"
    assert not jnp.any(jnp.isinf(L_final)), "L_final contains infinite values"

    assert not jnp.any(jnp.isnan(H_final)), "H_final contains NaN values"
    assert not jnp.any(jnp.isinf(H_final)), "H_final contains infinite values"

    assert not jnp.any(jnp.isnan(in_prod_final)), "in_prod_final contains NaN values"
    assert not jnp.any(jnp.isinf(in_prod_final)), "in_prod_final contains infinite values"

    assert not jnp.any(jnp.isnan(gamma_final)), "gamma_final contains NaN values"
    assert not jnp.any(jnp.isinf(gamma_final)), "gamma_final contains infinite values"

    assert not jnp.any(jnp.isnan(delta_final)), "delta_final contains NaN values"
    assert not jnp.any(jnp.isinf(delta_final)), "delta_final contains infinite values"

    assert not jnp.any(jnp.isnan(beta_final)), "beta_final contains NaN values"
    assert not jnp.any(jnp.isinf(beta_final)), "beta_final contains infinite values"

    assert not jnp.isnan(loss_final), "loss_final contains NaN values"
    assert not jnp.isinf(loss_final), "loss_final contains infinite values"
    assert loss_final >= 0


@pytest.mark.parametrize("use_max_window", [False, True])
def test_holdout_validate_max_window(use_max_window):
    Y, W, X, Z, V, true_params = generate_data(
        nobs=10,
        nperiods=30,
        unit_fe=True,
        time_fe=True,
        X_cov=True,
        Z_cov=True,
        V_cov=True,
        seed=2024,
        noise_scale=1.0,
        assignment_mechanism="last_periods",
        last_treated_periods=20,
    )

    if use_max_window:
        max_window = Y.shape[1] // 2
        initial_window = 10
        K = 5
        step_size = 1
        horizon = 6
    else:
        max_window = None
        initial_window = 10
        K = 5
        step_size = 2
        horizon = 6

    opt_lambda_L, opt_lambda_H, lambda_L_opt_range, lambda_H_opt_range = holdout_validate(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega_inv=None,
        use_unit_fe=True,
        use_time_fe=True,
        num_lam=6,
        initial_window=initial_window,
        step_size=step_size,
        horizon=horizon,
        max_iter=1_000,
        tol=1e-1,
        K=K,
        max_window_size=max_window,
    )

    assert not jnp.isnan(opt_lambda_L)
    assert jnp.isfinite(opt_lambda_L)
    assert opt_lambda_L >= 0
    assert not jnp.isnan(opt_lambda_H)
    assert jnp.isfinite(opt_lambda_H)
    assert opt_lambda_H >= 0

    assert opt_lambda_L in lambda_L_opt_range
    assert not jnp.any(jnp.isnan(lambda_L_opt_range))
    assert jnp.all(jnp.isfinite(lambda_L_opt_range))
    assert jnp.all(lambda_L_opt_range >= 0)

    assert opt_lambda_H in lambda_H_opt_range
    assert not jnp.any(jnp.isnan(lambda_H_opt_range))
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))
    assert jnp.all(lambda_H_opt_range >= 0)
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))

    L_final, H_final, in_prod_final, gamma_final, delta_final, beta_final, loss_final = final_fit(
        Y=Y,
        W=W,
        X=X,
        Z=Z,
        V=V,
        Omega_inv=None,
        use_unit_fe=True,
        use_time_fe=True,
        best_lambda_L=opt_lambda_L,
        best_lambda_H=opt_lambda_H,
        lambda_L_opt_range=lambda_L_opt_range,
        lambda_H_opt_range=lambda_H_opt_range,
    )
    assert not jnp.any(jnp.isnan(L_final)), "L_final contains NaN values"
    assert not jnp.any(jnp.isinf(L_final)), "L_final contains infinite values"

    assert not jnp.any(jnp.isnan(H_final)), "H_final contains NaN values"
    assert not jnp.any(jnp.isinf(H_final)), "H_final contains infinite values"

    assert not jnp.any(jnp.isnan(in_prod_final)), "in_prod_final contains NaN values"
    assert not jnp.any(jnp.isinf(in_prod_final)), "in_prod_final contains infinite values"

    assert not jnp.any(jnp.isnan(gamma_final)), "gamma_final contains NaN values"
    assert not jnp.any(jnp.isinf(gamma_final)), "gamma_final contains infinite values"

    assert not jnp.any(jnp.isnan(delta_final)), "delta_final contains NaN values"
    assert not jnp.any(jnp.isinf(delta_final)), "delta_final contains infinite values"

    assert not jnp.any(jnp.isnan(beta_final)), "beta_final contains NaN values"
    assert not jnp.any(jnp.isinf(beta_final)), "beta_final contains infinite values"

    assert not jnp.isnan(loss_final), "loss_final contains NaN values"
    assert not jnp.isinf(loss_final), "loss_final contains infinite values"
    assert loss_final >= 0
