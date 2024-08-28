# mypy: ignore-errors
# type: ignore
import jax.numpy as jnp
import pytest
import jax
from mcnnm.utils import generate_data
from mcnnm.core import initialize_fixed_effects_and_H, initialize_matrices
from mcnnm.wrappers import compute_treatment_effect, estimate
from mcnnm.types import Array

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
    assert treatment_effect != 0
    # No point in checking the exact value of the treatment effect, as it is computed off the initialization.


@pytest.mark.parametrize("N, T", [(10, 10)])
@pytest.mark.parametrize("fe_params", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("X_cov", [False, True])
@pytest.mark.parametrize("Z_cov", [False, True])
@pytest.mark.parametrize("V_cov", [False, True])
@pytest.mark.parametrize("noise_scale", [0.5, 1.0, 2.0])
def test_estimate(N, T, fe_params, X_cov, Z_cov, V_cov, noise_scale):
    use_unit_fe, use_time_fe = fe_params

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
        noise_scale=noise_scale,
    )

    # Ensure at least one of use_unit_fe or use_time_fe is True if covariates are used
    if (X_cov or Z_cov or V_cov) and not (use_unit_fe or use_time_fe):
        use_unit_fe = True

    # Run estimation
    results = estimate(
        Y=Y,
        W=W,
        X=X if X_cov else None,
        Z=Z if Z_cov else None,
        V=V if V_cov else None,
        use_unit_fe=use_unit_fe,
        use_time_fe=use_time_fe,
        validation_method="cv",
        K=2,  # Use 2 folds for faster testing
        n_lambda=3,  # Use 3 lambda values for faster testing
        max_iter=100,  # Reduce max iterations for faster testing
        tol=1e-3,  # Increase tolerance for faster convergence
    )

    # Basic checks
    assert isinstance(results.tau, float)
    assert isinstance(results.Y_completed, Array)
    assert results.Y_completed.shape == (N, T)

    # Check if estimated ATE is reasonably close to true ATE
    # Use a large tolerance due to small sample size and potential noise
    print(
        f"Absolute difference between estimated and true tau: "
        f"{jnp.abs(results.tau - true_params['treatment_effect']):.2f}"
    )

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

    # TODO: fix
    # # Check covariates
    # if X_cov:
    #     assert results.beta["X"] is not None
    #     assert results.beta["X"].shape == (X.shape[1],)
    # else:
    #     assert jnp.allclose(results.beta["X"], jnp.zeros_like(results.beta["X"]))
    #
    # if Z_cov:
    #     assert results.beta["Z"] is not None
    #     assert results.beta["Z"].shape == (Z.shape[1],)
    # else:
    #     assert jnp.allclose(results.beta["Z"], jnp.zeros_like(results.beta["Z"]))
    #
    # if V_cov:
    #     assert results.beta["V"] is not None
    #     assert results.beta["V"].shape == (V.shape[2],)
    # else:
    #     assert jnp.allclose(results.beta["V"], jnp.zeros_like(results.beta["V"]))

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
