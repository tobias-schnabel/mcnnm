import pytest
import jax.numpy as jnp

from mcnnm.utils import (
    generate_data,
)
from mcnnm.validation import cross_validate


@pytest.mark.parametrize("N, T", [(10, 10)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("noise_scale", [0.1, 0.5, 1.0])
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
        max_iter=10_000,
        tol=1e-1,
        K=5,
    )

    assert not jnp.isnan(opt_lambda_L)
    assert jnp.isfinite(opt_lambda_L)
    assert opt_lambda_L >= 0
    assert not jnp.isnan(opt_lambda_H)
    assert jnp.isfinite(opt_lambda_H)
    assert opt_lambda_H >= 0


# @pytest.mark.parametrize("N, T", [(10,10)])
# @pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
# @pytest.mark.parametrize("X_cov", [False, True])
# @pytest.mark.parametrize("Z_cov", [False, True])
# @pytest.mark.parametrize("V_cov", [False, True])
# @pytest.mark.parametrize("noise_scale", [0.1, 0.5, 1.0])
# @pytest.mark.parametrize("use_shortest_path", [False, True])
# @pytest.mark.parametrize("use_max_window", [False, True])
# def test_time_based_validate(
#     N,
#     T,
#     fe_params,
#     X_cov,
#     Z_cov,
#     V_cov,
#     noise_scale,
#     use_shortest_path,
#     use_max_window,
# ):
#     use_unit_fe, use_time_fe = fe_params
#     Y, W, X, Z, V, true_params = generate_data(
#         nobs=N, nperiods=T, unit_fe=use_unit_fe, time_fe=use_time_fe,
#         X_cov=X_cov, Z_cov=Z_cov, V_cov=V_cov,
#         seed=2024, noise_scale=noise_scale
#     )
#
#     L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)
#
#     gamma, delta, beta, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
#         initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, False, False, verbose=False)
#     )
#
#     lambda_grid = generate_lambda_grid(lambda_L_max, lambda_H_max, 10)
#
#     if use_shortest_path:
#         lambda_grid = extract_shortest_path(lambda_grid)
#
#     initial_window, step_size, horizon, K, T = generate_time_based_validate_defaults(Y)
#
#     if use_max_window:
#         max_window = 0.9 * initial_window
#     else:
#         max_window = None
#
#     opt_lambda_L, opt_lambda_H = time_based_validate(
#         Y=Y,
#         W=W,
#         X_tilde=X_tilde,
#         Z_tilde=Z_tilde,
#         V=V,
#         Omega_inv=None,
#         L=L,
#         gamma=gamma,
#         delta=delta,
#         beta=beta,
#         H_tilde=H_tilde,
#         T_mat=T_mat,
#         in_prod=in_prod,
#         in_prod_T=in_prod_T,
#         use_unit_fe=use_unit_fe,
#         use_time_fe=use_time_fe,
#         lambda_grid=lambda_grid,
#         max_iter=1000,
#         tol=1e-3,
#         initial_window=initial_window,
#         step_size=step_size,
#         horizon=horizon,
#         K=K,
#         T=T,
#         max_window_size=max_window,
#     )
#
#     assert not jnp.isnan(opt_lambda_L)
#     assert jnp.isfinite(opt_lambda_L)
#     assert opt_lambda_L <= lambda_L_max
#     assert opt_lambda_L > 0
#     assert not jnp.isnan(opt_lambda_H)
#     assert jnp.isfinite(opt_lambda_H)
#     assert opt_lambda_H <= lambda_H_max
#     assert opt_lambda_H > 0
