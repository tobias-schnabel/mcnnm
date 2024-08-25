from mcnnm.core import initialize_matrices, initialize_fixed_effects_and_H
import pytest
import jax.numpy as jnp

from mcnnm.utils import (
    generate_data,
    generate_lambda_grid,
    extract_shortest_path,
)
from mcnnm.validation import cross_validate


@pytest.mark.parametrize("N, T", [(10, 10)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("noise_scale", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("use_shortest_path", [False, True])
def test_cross_validate(N, T, fe_params, X_cov, Z_cov, V_cov, noise_scale, use_shortest_path):
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

    L, X_tilde, Z_tilde, V = initialize_matrices(Y, X, Z, V)

    gamma, delta, beta, H_tilde, T_mat, in_prod_T, in_prod, lambda_L_max, lambda_H_max = (
        initialize_fixed_effects_and_H(Y, L, X_tilde, Z_tilde, V, W, False, False, verbose=False)
    )

    lambda_grid = generate_lambda_grid(lambda_L_max, lambda_H_max, 10)

    if use_shortest_path:
        lambda_grid = extract_shortest_path(lambda_grid)

    opt_lambda_L, opt_lambda_H = cross_validate(
        Y=Y,
        W=W,
        X_tilde=X_tilde,
        Z_tilde=Z_tilde,
        V=V,
        Omega_inv=None,
        L=L,
        gamma=gamma,
        delta=delta,
        beta=beta,
        H_tilde=H_tilde,
        T_mat=T_mat,
        in_prod=in_prod,
        in_prod_T=in_prod_T,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        lambda_grid=lambda_grid,
        max_iter=100,
        tol=1e-3,
    )

    assert not jnp.isnan(opt_lambda_L)
    assert jnp.isfinite(opt_lambda_L)
    assert opt_lambda_L < lambda_L_max
    assert opt_lambda_L > 0
    assert not jnp.isnan(opt_lambda_H)
    assert jnp.isfinite(opt_lambda_H)
    assert opt_lambda_H < lambda_H_max
    assert opt_lambda_H > 0
