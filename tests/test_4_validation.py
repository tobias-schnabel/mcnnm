import pytest
import jax.numpy as jnp

from mcnnm.utils import (
    generate_data,
    generate_holdout_val_defaults,
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
    assert not jnp.any(jnp.isnan(lambda_L_opt_range))
    assert jnp.all(jnp.isfinite(lambda_L_opt_range))
    assert jnp.all(lambda_L_opt_range >= 0)
    assert not jnp.any(jnp.isnan(lambda_H_opt_range))
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))
    assert jnp.all(lambda_H_opt_range >= 0)
    assert jnp.all(jnp.isfinite(lambda_H_opt_range))


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

    opt_lambda_L, opt_lambda_H, max_lam_L, max_lam_H = holdout_validate(
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
    assert not jnp.isnan(max_lam_L)
    assert jnp.isfinite(max_lam_L)
    assert max_lam_L >= 0
    assert not jnp.isnan(max_lam_H)
    assert jnp.isfinite(max_lam_H)
    assert max_lam_H >= 0


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

    opt_lambda_L, opt_lambda_H, max_lam_L, max_lam_H = holdout_validate(
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
    assert not jnp.isnan(max_lam_L)
    assert jnp.isfinite(max_lam_L)
    assert max_lam_L >= 0
    assert not jnp.isnan(max_lam_H)
    assert jnp.isfinite(max_lam_H)
    assert max_lam_H >= 0


def test_holdout_validate_invalid():
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
        last_treated_periods=5,
    )
    # The generated data has 10 units and 30 time periods.
    # The last 5 periods are treated for all units, meaning they have missing entries in Y.

    max_window = 15
    initial_window = 10
    K = 5
    step_size = 2
    horizon = 6
    # The time-based validation parameters are set such that all holdout folds will fall within the last 5 periods.
    # initial_window = 10 means the first fold starts at period 10.
    # K = 5 indicates that we want to create 5 folds.
    # step_size = 2 means the start index will be incremented by 2 in each fold.
    # horizon = 6 means each fold will include 6 periods for evaluation.
    # max_window = 15 limits the window size for initializing the model configurations in each fold.

    opt_lambda_L, opt_lambda_H, max_lam_L, max_lam_H = holdout_validate(
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

    assert jnp.isnan(opt_lambda_L)
    assert jnp.isnan(opt_lambda_H)
    assert jnp.isnan(max_lam_L)
    assert jnp.isnan(max_lam_H)
    # Assert that both opt_lambda_L and opt_lambda_H are NaN (not-a-number).
    # This is expected because the time-based validation parameters are set in a way that no holdout folds fall within
    # the last 5 periods, which are the only periods in which data is unobserved
    # (i.e., there are no treated entries in any holdout folds).
    # When there are no valid folds for holdout validation, the holdout_validate function returns NaN for both
    # opt_lambda_L and opt_lambda_H.
