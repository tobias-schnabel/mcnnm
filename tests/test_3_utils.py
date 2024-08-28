import pytest
import jax.numpy as jnp
import pandas as pd
import numpy as np
from mcnnm.utils import (
    check_inputs,
    generate_data,
    convert_inputs,
    propose_lambda_values,
    generate_lambda_grid,
    extract_shortest_path,
)
from typing import Literal
import jax
from jax import random

key = jax.random.PRNGKey(2024)


def test_check_inputs():
    # Y = jnp.array([[1, 2], [3, 4]])
    # W = jnp.array([[0, 1], [0, 0]])
    Y, W, X, Z, V, true_params = generate_data(
        nobs=2, nperiods=2, X_cov=False, Z_cov=False, V_cov=False
    )
    assert X is None
    assert Z is None
    assert V is None
    assert true_params["Y(0)"].shape == (2, 2)


@pytest.mark.parametrize("unit_fe", [True, False])
@pytest.mark.parametrize("time_fe", [True, False])
@pytest.mark.parametrize("X_cov", [True, False])
@pytest.mark.parametrize("Z_cov", [True, False])
@pytest.mark.parametrize("V_cov", [True, False])
@pytest.mark.parametrize(
    "assignment_mechanism",
    [
        "staggered",
        "block",
        "single_treated_period",
        "single_treated_unit",
        "last_periods",
    ],
)
@pytest.mark.parametrize("autocorrelation", [0.0, 0.5])
def test_generate_data(
    unit_fe: bool,
    time_fe: bool,
    X_cov: bool,
    Z_cov: bool,
    V_cov: bool,
    assignment_mechanism: Literal[
        "staggered", "block", "single_treated_period", "single_treated_unit", "last_periods"
    ],
    autocorrelation: float,
):
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

    Y, W, X, Z, V, true_params = generate_data(
        nobs=nobs,
        nperiods=nperiods,
        treatment_probability=treatment_probability,
        rank=rank,
        treatment_effect=treatment_effect,
        unit_fe=unit_fe,
        time_fe=time_fe,
        X_cov=X_cov,
        Z_cov=Z_cov,
        V_cov=V_cov,
        fixed_effects_scale=fixed_effects_scale,
        covariates_scale=covariates_scale,
        noise_scale=noise_scale,
        assignment_mechanism=assignment_mechanism,
        treated_fraction=treated_fraction,
        last_treated_periods=last_treated_periods,
        autocorrelation=autocorrelation,
        seed=seed,
    )

    assert Y.shape == (nobs, nperiods)
    assert W.shape == (nobs, nperiods)
    assert true_params["L"].shape == (nobs, nperiods)

    assert "unit_fe" in true_params
    assert true_params["unit_fe"].shape == (nobs,)
    if unit_fe:
        assert not np.allclose(true_params["unit_fe"], 0)
    else:
        assert np.allclose(true_params["unit_fe"], 0)

    assert "time_fe" in true_params
    assert true_params["time_fe"].shape == (nperiods,)
    if time_fe:
        assert not np.allclose(true_params["time_fe"], 0)
    else:
        assert np.allclose(true_params["time_fe"], 0)

    if X_cov:
        assert X is not None
        assert X.shape == (nobs, 2)
        assert "X_coef" in true_params
        assert true_params["X_coef"].shape == (2,)
    else:
        assert X is None
        assert "X_coef" in true_params
        assert true_params["X_coef"].size == 0

    if Z_cov:
        assert Z is not None
        assert Z.shape == (nperiods, 2)
        assert "Z_coef" in true_params
        assert true_params["Z_coef"].shape == (2,)
    else:
        assert Z is None
        assert "Z_coef" in true_params
        assert true_params["Z_coef"].size == 0

    if V_cov:
        assert V is not None
        assert V.shape == (nobs, nperiods, 2)
        assert "V_coef" in true_params
        assert true_params["V_coef"].shape == (2,)
    else:
        assert V is None
        assert "V_coef" in true_params
        assert true_params["V_coef"].size == 0

    assert jnp.allclose(Y, true_params["Y(0)"] + W * treatment_effect)
    assert (
        isinstance(Y, jnp.ndarray) and isinstance(W, jnp.ndarray) and isinstance(true_params, dict)
    )

    assert true_params["treatment_effect"] == treatment_effect
    assert true_params["noise_scale"] == noise_scale
    assert true_params["assignment_mechanism"] == assignment_mechanism
    assert true_params["autocorrelation"] == autocorrelation


def test_generate_data_autocorrelation():
    with pytest.raises(ValueError):
        generate_data(autocorrelation=-0.1)
    with pytest.raises(ValueError):
        generate_data(autocorrelation=1.0)


# Helper function to generate random arrays
def generate_random_array(key, shape):
    return random.uniform(key, shape)


# Fixture for common test data
@pytest.fixture
def test_data():
    key = random.PRNGKey(0)
    N, T = 10, 5
    Y = generate_random_array(random.split(key)[0], (N, T))
    W = random.bernoulli(random.split(key)[1], shape=(N, T))
    return N, T, Y, W


def test_valid_inputs(test_data):
    N, T, Y, W = test_data
    X = generate_random_array(random.PRNGKey(1), (N, 3))
    Z = generate_random_array(random.PRNGKey(2), (T, 2))
    V = generate_random_array(random.PRNGKey(3), (N, T, 4))
    Omega = generate_random_array(random.PRNGKey(4), (T, T))

    result = check_inputs(Y, W, X, Z, V, Omega)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert all(isinstance(arr, jnp.ndarray) for arr in result)
    assert result[0].shape == (N, 3)
    assert result[1].shape == (T, 2)
    assert result[2].shape == (N, T, 4)
    assert result[3].shape == (T, T)


def test_minimal_valid_inputs(test_data):
    N, T, Y, W = test_data
    result = check_inputs(Y, W)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert result[0].shape == (N, 0)
    assert result[1].shape == (T, 0)
    assert result[2].shape == (N, T, 1)
    assert result[3].shape == (T, T)


def test_mismatched_W_shape(test_data):
    N, T, Y, _ = test_data
    W = jnp.ones((N + 1, T))
    with pytest.raises(ValueError, match="The shape of W .* must match the shape of Y"):
        check_inputs(Y, W)


def test_invalid_X_shape(test_data):
    N, T, Y, W = test_data
    X = jnp.ones((N + 1, 3))
    with pytest.raises(
        ValueError, match="The first dimension of X .* must match the first dimension of Y"
    ):
        check_inputs(Y, W, X)


def test_invalid_Z_shape(test_data):
    N, T, Y, W = test_data
    Z = jnp.ones((T + 1, 2))
    with pytest.raises(
        ValueError, match="The first dimension of Z .* must match the second dimension of Y"
    ):
        check_inputs(Y, W, Z=Z)


def test_invalid_V_shape(test_data):
    N, T, Y, W = test_data
    V = jnp.ones((N + 1, T, 4))
    with pytest.raises(
        ValueError, match="The first two dimensions of V .* must match the shape of Y"
    ):
        check_inputs(Y, W, V=V)


def test_invalid_Omega_shape(test_data):
    N, T, Y, W = test_data
    Omega = jnp.ones((T + 1, T + 1))
    with pytest.raises(ValueError, match="The shape of Omega .* must be"):
        check_inputs(Y, W, Omega=Omega)


def test_X_None(test_data):
    N, T, Y, W = test_data
    result = check_inputs(Y, W, X=None)
    assert result[0].shape == (N, 0)


def test_Z_None(test_data):
    N, T, Y, W = test_data
    result = check_inputs(Y, W, Z=None)
    assert result[1].shape == (T, 0)


def test_V_None(test_data):
    N, T, Y, W = test_data
    result = check_inputs(Y, W, V=None)
    assert result[2].shape == (N, T, 1)


def test_Omega_None(test_data):
    N, T, Y, W = test_data
    result = check_inputs(Y, W, Omega=None)
    assert jnp.array_equal(result[3], jnp.eye(T))


def test_edge_case_single_unit(test_data):
    _, T, Y, W = test_data
    Y = Y[:1, :]
    W = W[:1, :]
    result = check_inputs(Y, W)
    assert result[0].shape == (1, 0)
    assert result[1].shape == (T, 0)
    assert result[2].shape == (1, T, 1)
    assert result[3].shape == (T, T)


def test_edge_case_single_time_point(test_data):
    N, _, Y, W = test_data
    Y = Y[:, :1]
    W = W[:, :1]
    result = check_inputs(Y, W)
    assert result[0].shape == (N, 0)
    assert result[1].shape == (1, 0)
    assert result[2].shape == (N, 1, 1)
    assert result[3].shape == (1, 1)


def test_edge_case_empty_X(test_data):
    N, T, Y, W = test_data
    X = jnp.zeros((N, 0))
    result = check_inputs(Y, W, X=X)
    assert result[0].shape == (N, 0)


def test_edge_case_empty_Z(test_data):
    N, T, Y, W = test_data
    Z = jnp.zeros((T, 0))
    result = check_inputs(Y, W, Z=Z)
    assert result[1].shape == (T, 0)


def test_edge_case_empty_V(test_data):
    N, T, Y, W = test_data
    V = jnp.zeros((N, T, 0))
    result = check_inputs(Y, W, V=V)
    assert result[2].shape == (N, T, 0)


def test_non_jax_numpy_arrays(test_data):
    import numpy as np

    N, T, Y, W = test_data
    Y = np.array(Y)
    W = np.array(W)
    result = check_inputs(Y, W)
    assert all(isinstance(arr, jnp.ndarray) for arr in result)


def test_input_mutability(test_data):
    N, T, Y, W = test_data
    X = generate_random_array(random.PRNGKey(1), (N, 3))
    Z = generate_random_array(random.PRNGKey(2), (T, 2))
    V = generate_random_array(random.PRNGKey(3), (N, T, 4))
    Omega = generate_random_array(random.PRNGKey(4), (T, T))

    original_inputs = [Y, W, X, Z, V, Omega]
    _ = check_inputs(Y, W, X, Z, V, Omega)

    for original, current in zip(original_inputs, [Y, W, X, Z, V, Omega]):
        assert jnp.array_equal(original, current), "Input arrays should not be modified"


def test_generate_data_invalid_assignment():
    with pytest.raises(ValueError, match="Invalid assignment mechanism specified."):
        # noinspection PyTypeChecker
        generate_data(assignment_mechanism="invalid_mechanism")


def test_convert_inputs_happy_path():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    X = pd.DataFrame([[5, 6], [7, 8]])
    Z = pd.DataFrame([[9, 10], [11, 12]])
    V = [pd.DataFrame([[13, 14], [15, 16]])]
    Omega = pd.DataFrame([[1, 0], [0, 1]])

    Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W, X, Z, V, Omega)

    assert isinstance(Y_jax, jnp.ndarray)
    assert isinstance(W_jax, jnp.ndarray)
    assert isinstance(X_jax, jnp.ndarray)
    assert isinstance(Z_jax, jnp.ndarray)
    assert isinstance(V_jax, jnp.ndarray)
    assert isinstance(Omega_jax, jnp.ndarray)

    # Check shapes
    assert Y_jax.shape == (2, 2)
    assert W_jax.shape == (2, 2)
    assert X_jax.shape == (2, 2)
    assert Z_jax.shape == (2, 2)
    assert V_jax.shape == (2, 2, 1)
    assert Omega_jax.shape == (2, 2)


def convert_inputs_no_optional():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])

    Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W)

    assert isinstance(Y_jax, jnp.ndarray)
    assert isinstance(W_jax, jnp.ndarray)
    assert X_jax is None
    assert Z_jax is None
    assert V_jax is None
    assert Omega_jax is None


def convert_inputs_empty_dataframes():
    Y = pd.DataFrame()
    W = pd.DataFrame()

    with pytest.raises(ValueError):
        convert_inputs(Y, W)


def convert_inputs_mismatched_shapes():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1]])

    with pytest.raises(ValueError):
        convert_inputs(Y, W)


def test_convert_inputs_with_X_only():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    X = pd.DataFrame([[5, 6], [7, 8]])

    Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W, X=X)

    assert isinstance(X_jax, jnp.ndarray)
    assert X_jax.shape == (2, 2)
    assert Z_jax is None
    assert V_jax is None
    assert Omega_jax is None


def test_convert_inputs_with_Z_only():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    Z = pd.DataFrame([[9, 10], [11, 12]])

    Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W, Z=Z)

    assert X_jax is None
    assert isinstance(Z_jax, jnp.ndarray)
    assert Z_jax.shape == (2, 2)
    assert V_jax is None
    assert Omega_jax is None


def test_convert_inputs_with_V_only():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    V = [pd.DataFrame([[13, 14], [15, 16]])]

    Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W, V=V)

    assert X_jax is None
    assert Z_jax is None
    assert isinstance(V_jax, jnp.ndarray)
    assert V_jax.shape == (2, 2, 1)
    assert Omega_jax is None


def test_convert_inputs_with_Omega_only():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    Omega = pd.DataFrame([[1, 0], [0, 1]])

    Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W, Omega=Omega)

    assert X_jax is None
    assert Z_jax is None
    assert V_jax is None
    assert isinstance(Omega_jax, jnp.ndarray)
    assert Omega_jax.shape == (2, 2)


def test_convert_inputs_with_empty_V():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    V = []

    with pytest.raises(ValueError):
        Y_jax, W_jax, X_jax, Z_jax, V_jax, Omega_jax = convert_inputs(Y, W, V=V)


def test_convert_inputs_mismatched_X_shape():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    X = pd.DataFrame([[5, 6]])  # Only one row, should be two to match Y

    with pytest.raises(
        ValueError, match="The first dimension of X .* must match the first dimension of Y"
    ):
        convert_inputs(Y, W, X=X)


def test_convert_inputs_mismatched_Z_shape():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    Z = pd.DataFrame([[9, 10, 11]])  # Three columns, should be two to match Y

    with pytest.raises(
        ValueError, match="The first dimension of Z .* must match the second dimension of Y"
    ):
        convert_inputs(Y, W, Z=Z)


def test_convert_inputs_mismatched_V_shape():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    V = [pd.DataFrame([[13, 14], [15, 16], [17, 18]])]  # Three rows, should be two to match Y

    with pytest.raises(ValueError):
        convert_inputs(Y, W, V=V)


def test_convert_inputs_mismatched_Omega_shape():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    Omega = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3, should be 2x2 to match Y

    with pytest.raises(ValueError, match="The shape of Omega .* must be"):
        convert_inputs(Y, W, Omega=Omega)


def test_convert_inputs_empty_V_list():
    Y = pd.DataFrame([[1, 2], [3, 4]])
    W = pd.DataFrame([[0, 1], [1, 0]])
    V = []

    with pytest.raises(ValueError, match="V cannot be an empty list"):
        convert_inputs(Y, W, V=V)


def test_staggered_assignment_mechanism():
    # Set a seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # Set parameters to ensure staggered assignment and early adoption
    nobs = 100
    nperiods = 50
    treatment_probability = 0.1  # Higher probability for earlier adoption

    Y, W, X, Z, V, true_params = generate_data(
        nobs=nobs,
        nperiods=nperiods,
        treatment_probability=treatment_probability,
        assignment_mechanism="staggered",
        seed=seed,
    )

    # Check that the assignment mechanism is staggered
    assert true_params["assignment_mechanism"] == "staggered"

    # Check that at least one unit has been treated
    assert W.sum() > 0

    # Check that there's at least one unit with adoption time <= nperiods
    adoption_times = np.where(W.sum(axis=1) > 0, W.argmax(axis=1), nperiods + 1)
    assert (adoption_times <= nperiods).sum() > 0

    # Check that the treatment pattern is consistent with staggered adoption
    for i in range(nobs):
        if adoption_times[i] <= nperiods:
            assert np.all(W[i, adoption_times[i] :] == 1)
            if adoption_times[i] > 0:
                assert np.all(W[i, : adoption_times[i]] == 0)

    # Check that Y and W have the expected shapes
    assert Y.shape == (nobs, nperiods)
    assert W.shape == (nobs, nperiods)


def test_propose_lambda_default():
    lambdas = propose_lambda_values(1.0)
    assert len(lambdas) == 6
    assert jnp.allclose(lambdas[0], 0.0)
    assert jnp.allclose(lambdas[-1], 1.0)


def test_propose_lambda_custom():
    lambdas = propose_lambda_values(10.0, 0.1, 5)
    assert len(lambdas) == 5
    assert jnp.allclose(lambdas[0], 0.0)
    assert jnp.allclose(lambdas[-1], 10.0)


def test_propose_lambda_small_max_lambda():
    with pytest.raises(ValueError, match="max_lambda .* is too small"):
        propose_lambda_values(1e-11)


def test_propose_lambda_equal_max_min_lambda():
    lambdas = propose_lambda_values(1.0, 1.0, 3)
    assert len(lambdas) == 3
    assert jnp.allclose(lambdas[1:2], jnp.ones(2))
    assert jnp.allclose(lambdas[0], 0.0)


def test_propose_lambda_max_smaller_than_min():
    with pytest.raises(ValueError):
        propose_lambda_values(1.0, 10.0, 4)


def test_propose_lambda_single_value():
    with pytest.raises(ValueError):
        propose_lambda_values(5.0, 5.0, 1)


def test_generate_lambda_grid():
    max_lambda_L = 10.0
    max_lambda_H = 10.0
    n_lambda = 3

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, n_lambda)

    assert lambda_grid.shape == (9, 2)
    assert jnp.allclose(lambda_grid[-1], jnp.array([10.0, 10.0]))


def test_generate_lambda_grid_different_values():
    max_lambda_L = 5.0
    max_lambda_H = 20.0
    n_lambda = 4

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, n_lambda)

    assert lambda_grid.shape == (16, 2)
    assert jnp.allclose(lambda_grid[0], jnp.array([0.0, 0.0]))
    assert jnp.allclose(lambda_grid[-1], jnp.array([5.0, 20.0]))


def test_extract_shortest_path():
    max_lambda_L = 10.0
    max_lambda_H = 10.0
    n_lambda = 3

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, n_lambda)
    shortest_path = extract_shortest_path(lambda_grid)

    assert shortest_path.shape == (5, 2)
    assert jnp.allclose(shortest_path[0], jnp.array([10.0, 10.0]))
    assert jnp.allclose(shortest_path[-2], jnp.array([0.01, 0.0]))
    assert jnp.allclose(shortest_path[-1], jnp.array([0.0, 0.0]))


def test_extract_shortest_path_order():
    max_lambda_L = 10.0
    max_lambda_H = 10.0
    n_lambda = 3

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, n_lambda)
    shortest_path = extract_shortest_path(lambda_grid)

    # Check if lambda_L is non-increasing
    assert jnp.all(jnp.diff(shortest_path[:, 0]) <= 0)

    # Check if lambda_H starts high, then stays at the lowest value
    assert jnp.all(jnp.diff(shortest_path[:4, 1]) <= 0)
    assert jnp.allclose(shortest_path[3:-1, 1], shortest_path[-2, 1])


def test_extract_shortest_path_different_grid():
    max_lambda_L = 5.0
    max_lambda_H = 20.0
    n_lambda = 4

    lambda_grid = generate_lambda_grid(max_lambda_L, max_lambda_H, n_lambda)
    shortest_path = extract_shortest_path(lambda_grid)
    assert shortest_path.shape == (7, 2)
    assert jnp.allclose(shortest_path[0], jnp.array([5.0, 20.0]))
    assert jnp.allclose(shortest_path[-1], jnp.array([0.0, 0.0]))


def test_propose_lambda():
    max_lambda_L = 5.0
    n_lambda = 4

    lambda_values = propose_lambda_values(max_lambda_L, n_lambdas=n_lambda)

    assert len(lambda_values) == n_lambda
    assert jnp.allclose(lambda_values[0], jnp.array(0.0))
    assert jnp.allclose(lambda_values[-1], jnp.array(5.0))
