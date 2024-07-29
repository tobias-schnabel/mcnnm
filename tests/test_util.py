import pytest
import jax.numpy as jnp
from mcnnm.util import p_o, p_perp_o, shrink_lambda, frobenius_norm, nuclear_norm
from mcnnm.util import propose_lambda, print_with_timestamp, initialize_params, check_inputs
from mcnnm.util import generate_data, element_wise_l1_norm
import jax
from jax import random
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_disable_jit', True)

key = jax.random.PRNGKey(2024)


@pytest.fixture
def sample_data():
    N, T, P, Q, J = 10, 5, 3, 2, 4
    Y = random.normal(key, (N, T))
    W = random.bernoulli(key, 0.2, (N, T))
    X = random.normal(key, (N, P))
    Z = random.normal(key, (T, Q))
    V = random.normal(key, (N, T, J))
    return Y, W, X, Z, V


def test_p_o(sample_data):
    Y, W, _, _, _ = sample_data
    mask = (W == 0)
    assert jnp.allclose(p_o(Y, mask), Y * mask)


def test_p_perp_o(sample_data):
    Y, W, _, _, _ = sample_data
    mask = (W == 0)
    assert jnp.allclose(p_perp_o(Y, mask), Y * (1 - mask))


def test_shrink_lambda(sample_data):
    Y, _, _, _, _ = sample_data
    lambda_ = 0.1
    Y_shrunk = shrink_lambda(Y, lambda_)
    assert Y_shrunk.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y_shrunk))


def test_frobenius_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(frobenius_norm(A), jnp.sqrt(30))


def test_nuclear_norm():
    A = jnp.array([[1, 2], [3, 4]])
    assert jnp.allclose(nuclear_norm(A), jnp.sum(jnp.linalg.svd(A, compute_uv=False)))


def test_element_wise_l1_norm():
    A = jnp.array([[1, -2], [-3, 4]])
    assert jnp.allclose(element_wise_l1_norm(A), 10)


def test_propose_lambda():
    lambdas = propose_lambda(1.0, 5)
    assert len(lambdas) == 5
    assert jnp.allclose(lambdas[0], 0.01)
    assert jnp.allclose(lambdas[-1], 100.0)


def test_initialize_params(sample_data):
    Y, W, X, Z, V = sample_data
    L, H, gamma, delta, beta = initialize_params(Y, W, X, Z, V)
    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + Y.shape[0], Z.shape[1] + Y.shape[1])
    assert gamma.shape == (Y.shape[0],)
    assert delta.shape == (Y.shape[1],)
    assert beta.shape == (V.shape[2],)


def test_check_inputs():
    Y = jnp.array([[1, 2], [3, 4]])
    W = jnp.array([[0, 1], [0, 0]])
    X, Z, V, Omega = check_inputs(Y, W)
    assert X.shape == (2, 0)
    assert Z.shape == (2, 0)
    assert V.shape == (2, 2, 0)
    assert Omega.shape == (2, 2)


@pytest.mark.parametrize("unit_fe", [True, False])
@pytest.mark.parametrize("time_fe", [True, False])
@pytest.mark.parametrize("X_cov", [True, False])
@pytest.mark.parametrize("Z_cov", [True, False])
@pytest.mark.parametrize("V_cov", [True, False])
@pytest.mark.parametrize("assignment_mechanism",
                         ['staggered', 'block', 'single_treated_period', 'single_treated_unit', 'last_periods'])
@pytest.mark.parametrize("autocorrelation", [0.0, 0.5])
def test_generate_data(unit_fe, time_fe, X_cov, Z_cov, V_cov, assignment_mechanism, autocorrelation):
    nobs, nperiods = 100, 50
    treatment_probability = 0.8
    rank = 3
    treatment_effect = 2.0
    fixed_effects_scale = 0.2
    covariates_scale = 0.2
    noise_scale = 0.2
    treated_fraction = 0.4
    last_treated_periods = 5
    seed = 42

    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, treatment_probability=treatment_probability,
                                      rank=rank, treatment_effect=treatment_effect, unit_fe=unit_fe, time_fe=time_fe,
                                      X_cov=X_cov, Z_cov=Z_cov, V_cov=V_cov, fixed_effects_scale=fixed_effects_scale,
                                      covariates_scale=covariates_scale, noise_scale=noise_scale,
                                      assignment_mechanism=assignment_mechanism, treated_fraction=treated_fraction,
                                      last_treated_periods=last_treated_periods, autocorrelation=autocorrelation,
                                      seed=seed)

    assert data.shape == (nobs * nperiods, 4)
    assert set(data.columns) == {'unit', 'period', 'y', 'treat'}
    assert true_params['L'].shape == (nobs, nperiods)
    if unit_fe:
        assert true_params['unit_fe'].shape == (nobs,)
    if time_fe:
        assert true_params['time_fe'].shape == (nperiods,)
    if X_cov:
        assert true_params['X'].shape == (nobs, 2)
        assert true_params['X_coef'].shape == (2,)
    else:
        assert true_params['X'].shape == (nobs, 0)
        assert true_params['X_coef'].size == 0
    if Z_cov:
        assert true_params['Z'].shape == (nperiods, 2)
        assert true_params['Z_coef'].shape == (2,)
    else:
        assert true_params['Z'].shape == (nperiods, 0)
        assert true_params['Z_coef'].size == 0
    if V_cov:
        assert true_params['V'].shape == (nobs, nperiods, 2)
        assert true_params['V_coef'].shape == (2,)
    else:
        assert true_params['V'].shape == (nobs, nperiods, 0)
        assert true_params['V_coef'].size == 0
    assert set(data['treat'].unique()) == {0, 1}


def test_generate_data_autocorrelation():
    with pytest.raises(ValueError):
        generate_data(autocorrelation=-0.1)
    with pytest.raises(ValueError):
        generate_data(autocorrelation=1.0)


def test_print_with_timestamp(capsys):
    print_with_timestamp("Test message")
    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_p_o_shape_mismatch():
    A = jnp.array([[1, 2], [3, 4]])
    mask = jnp.array([[True, False]])
    with pytest.raises(ValueError, match="Shapes of A and mask must match."):
        p_o(A, mask)


def test_p_perp_o_shape_mismatch():
    A = jnp.array([[1, 2], [3, 4]])
    mask = jnp.array([[True, False]])
    with pytest.raises(ValueError, match="Shapes of A and mask must match."):
        p_perp_o(A, mask)


def test_frobenius_norm_non_2d():
    A = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be a 2D array."):
        frobenius_norm(A)


def test_nuclear_norm_non_2d():
    A = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be a 2D array."):
        nuclear_norm(A)


def test_element_wise_l1_norm_non_2d():
    A = jnp.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input must be a 2D array."):
        element_wise_l1_norm(A)


def test_check_inputs_shape_mismatch():
    Y = jnp.array([[1, 2], [3, 4]])
    W = jnp.array([[0, 1]])
    with pytest.raises(ValueError, match="The shape of W must match the shape of Y."):
        check_inputs(Y, W)


def test_check_inputs_with_na():
    Y = jnp.array([[1.0, 2.0], [3.0, jnp.nan]])
    W = jnp.array([[0, 1], [1, 0]])
    X, Z, V, Omega = check_inputs(Y, W)
    assert X.shape == (2, 0)
    assert Z.shape == (2, 0)
    assert V.shape == (2, 2, 0)
    assert Omega.shape == (2, 2)


def test_generate_data_invalid_assignment():
    with pytest.raises(ValueError, match="Invalid assignment mechanism specified."):
        generate_data(assignment_mechanism='invalid_mechanism')
