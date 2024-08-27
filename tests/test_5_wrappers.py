# mypy: ignore-errors
# type: ignore
import jax.numpy as jnp
import pytest
import jax
from mcnnm.utils import generate_data
from mcnnm.core import initialize_fixed_effects_and_H, initialize_matrices
from mcnnm.wrappers import compute_treatment_effect

key = jax.random.PRNGKey(2024)


@pytest.mark.parametrize("N, T", [(10, 10)])
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
    assert treatment_effect >= 0
    # No point in checking the exact value of the treatment effect, as it is computed off the initialization.
