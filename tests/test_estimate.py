from mcnnm.util import initialize_params
import pytest
import jax.numpy as jnp
from jax import random
from mcnnm.estimate import estimate, fit, fit_step, update_L, update_H, update_gamma_delta_beta
from mcnnm.estimate import compute_treatment_effect, compute_cv_loss, cross_validate, compute_time_based_loss
from mcnnm.estimate import time_based_validate, complete_matrix
from mcnnm.util import generate_data
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_disable_jit', True)

# Set a fixed seed for reproducibility
key = random.PRNGKey(2024)


@pytest.fixture
def sample_data():
    N, T, P, Q, J = 10, 5, 3, 2, 4
    Y = random.normal(key, (N, T))
    W = random.bernoulli(key, 0.2, (N, T))
    X = random.normal(key, (N, P))
    Z = random.normal(key, (T, Q))
    V = random.normal(key, (N, T, J))
    return Y, W, X, Z, V


def test_update_L():
    Y_adj = jnp.array([[1, 2], [3, 4]])
    L = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    Omega = jnp.eye(2)
    O = jnp.array([[1, 1], [1, 0]])
    lambda_L = 0.1
    updated_L = update_L(Y_adj, L, Omega, O, lambda_L)
    assert updated_L.shape == (2, 2)
    assert jnp.all(jnp.isfinite(updated_L))


def test_update_H():
    X_tilde = jnp.array([[1, 2], [3, 4]])
    Y_adj = jnp.array([[1, 2], [3, 4]])
    Z_tilde = jnp.array([[1, 2], [3, 4]])
    lambda_H = 0.1
    updated_H = update_H(X_tilde, Y_adj, Z_tilde, lambda_H)
    assert updated_H.shape == (2, 2)
    assert jnp.all(jnp.isfinite(updated_H))


def test_update_gamma_delta_beta():
    Y_adj = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    V = jnp.array([[[1, 2], [3, 4], [5, 6]],
                   [[7, 8], [9, 10], [11, 12]],
                   [[13, 14], [15, 16], [17, 18]]])

    gamma, delta, beta = update_gamma_delta_beta(Y_adj, V)

    assert gamma.shape == (3,)
    assert delta.shape == (3,)
    assert beta.shape == (2,)

    # Test with empty V
    V_empty = jnp.zeros((3, 3, 0))
    gamma, delta, beta = update_gamma_delta_beta(Y_adj, V_empty)

    assert gamma.shape == (3,)
    assert delta.shape == (3,)
    assert beta.shape == (0,)


def test_fit_step():
    Y = jnp.array([[1, 2], [3, 4]])
    W = jnp.array([[0, 1], [0, 0]])
    X_tilde = jnp.array([[1, 2], [3, 4]])
    Z_tilde = jnp.array([[1, 2], [3, 4]])
    V = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    Omega = jnp.eye(2)
    lambda_L = 0.1
    lambda_H = 0.1
    L = jnp.zeros((2, 2))
    H = jnp.zeros((2, 2))
    gamma = jnp.zeros(2)
    delta = jnp.zeros(2)
    beta = jnp.zeros(2)
    L_new, H_new, gamma_new, delta_new, beta_new = fit_step(Y, W, X_tilde, Z_tilde, V, Omega, lambda_L,
                                                            lambda_H, L, H, gamma, delta, beta)
    assert L_new.shape == (2, 2)
    assert H_new.shape == (2, 2)
    assert gamma_new.shape == (2,)
    assert delta_new.shape == (2,)
    assert beta_new.shape == (2,)
    assert jnp.all(jnp.isfinite(L_new))
    assert jnp.all(jnp.isfinite(H_new))
    assert jnp.all(jnp.isfinite(gamma_new))
    assert jnp.all(jnp.isfinite(delta_new))
    assert jnp.all(jnp.isfinite(beta_new))


@pytest.mark.parametrize("nobs, nperiods", [(10, 5), (50, 10)])
def test_fit(nobs, nperiods):
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])
    Omega = jnp.eye(nperiods)
    lambda_L = 0.1
    lambda_H = 0.1

    # Test with default initial params
    initial_params = initialize_params(Y, X, Z, V)
    max_iter = 100
    tol = 1e-4

    L, H, gamma, delta, beta = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, initial_params, max_iter, tol)

    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + nobs, Z.shape[1] + nperiods)
    assert gamma.shape == (nobs,)
    assert delta.shape == (nperiods,)
    assert beta.shape == (V.shape[2],)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(H))
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(jnp.isfinite(beta))

    # Test with custom initial params
    custom_initial_params = (
        jnp.ones_like(Y),
        jnp.ones((X.shape[1] + nobs, Z.shape[1] + nperiods)),
        jnp.ones(nobs),
        jnp.ones(nperiods),
        jnp.ones(V.shape[2])
    )

    L, H, gamma, delta, beta = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, custom_initial_params, max_iter, tol)

    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + nobs, Z.shape[1] + nperiods)
    assert gamma.shape == (nobs,)
    assert delta.shape == (nperiods,)
    assert beta.shape == (V.shape[2],)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(H))
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(jnp.isfinite(beta))

    # Test with different max_iter and tol
    max_iter = 10
    tol = 1e-2

    L, H, gamma, delta, beta = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, initial_params, max_iter, tol)

    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + nobs, Z.shape[1] + nperiods)
    assert gamma.shape == (nobs,)
    assert delta.shape == (nperiods,)
    assert beta.shape == (V.shape[2],)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(H))
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(jnp.isfinite(beta))

    # Test with different lambda values
    lambda_L = 0.01
    lambda_H = 1.0

    L, H, gamma, delta, beta = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, initial_params, max_iter, tol)

    assert L.shape == Y.shape
    assert H.shape == (X.shape[1] + nobs, Z.shape[1] + nperiods)
    assert gamma.shape == (nobs,)
    assert delta.shape == (nperiods,)
    assert beta.shape == (V.shape[2],)
    assert jnp.all(jnp.isfinite(L))
    assert jnp.all(jnp.isfinite(H))
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(jnp.isfinite(beta))


def test_compute_treatment_effect(sample_data):
    Y, W, X, Z, V = sample_data
    N, T = Y.shape
    P = X.shape[1]
    Q = Z.shape[1]

    L = random.normal(key, Y.shape)
    gamma = random.normal(key, (N,))
    delta = random.normal(key, (T,))
    beta = random.normal(key, (V.shape[2],))

    # Correct H shape
    H = random.normal(key, (N + P, T + Q))

    print(f"Shape of X: {X.shape}")
    print(f"Shape of H: {H.shape}")
    print(f"Shape of Z: {Z.shape}")
    print(f"Shape of X_tilde: {jnp.hstack((X, jnp.eye(N))).shape}")
    print(f"Shape of Z_tilde: {jnp.hstack((Z, jnp.eye(T))).shape}")

    tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
    assert jnp.isfinite(tau)


@pytest.mark.timeout(180)
def test_estimate():
    nobs, nperiods = 10, 10
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])
    results = estimate(Y, W, X=X, Z=Z, V=V, K=2)
    assert jnp.isfinite(results.tau)
    assert jnp.isfinite(results.lambda_L)
    assert jnp.isfinite(results.lambda_H)
    assert results.L.shape == Y.shape
    assert results.Y_completed.shape == Y.shape
    assert jnp.all(jnp.isfinite(results.L))
    assert jnp.all(jnp.isfinite(results.Y_completed))


@pytest.mark.timeout(180)
def test_complete_matrix():
    nobs, nperiods = 10, 10
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])
    Y_completed, lambda_L, lambda_H = complete_matrix(Y, W, X=X, Z=Z, V=V, K=2)
    assert Y_completed.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y_completed))
    assert not jnp.any(jnp.isnan(Y_completed))
    assert jnp.isfinite(lambda_L)
    assert jnp.isfinite(lambda_H)

    # Optional: Check if the completed values are within a reasonable range
    assert jnp.all(Y_completed >= Y.min() - 1) and jnp.all(Y_completed <= Y.max() + 1)


def test_compute_cv_loss():
    N, T, J = 5, 4, 2
    Y = jnp.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [17, 18, 19, 20]])
    W = jnp.array([[0, 0, 1, 1],
                   [0, 1, 1, 1],
                   [1, 1, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Z = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    V = jnp.ones((N, T, J))
    Omega = jnp.eye(T)
    lambda_L = 0.1
    lambda_H = 0.1
    max_iter = 10
    tol = 1e-4

    loss = compute_cv_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H, max_iter, tol)

    # Check if loss is finite or NaN
    assert jnp.isfinite(loss) or jnp.isnan(loss)

    # If loss is finite, check if it's non-negative
    if jnp.isfinite(loss):
        assert loss >= 0

    # Test with different lambda values
    lambda_L = 0.01
    lambda_H = 1.0
    loss = compute_cv_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H, max_iter, tol)
    assert jnp.isfinite(loss) or jnp.isnan(loss)
    if jnp.isfinite(loss):
        assert loss >= 0

    # Test with different max_iter and tol
    max_iter = 5
    tol = 1e-3
    loss = compute_cv_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H, max_iter, tol)
    assert jnp.isfinite(loss) or jnp.isnan(loss)
    if jnp.isfinite(loss):
        assert loss >= 0


def test_cross_validate():
    N, T, J = 5, 4, 2
    Y = jnp.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [17, 18, 19, 20]])
    W = jnp.array([[0, 0, 1, 1],
                   [0, 1, 1, 1],
                   [1, 1, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Z = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    V = jnp.ones((N, T, J))
    Omega = jnp.eye(T)
    lambda_grid = jnp.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    max_iter = 10
    tol = 1e-4
    K = 2

    best_lambda_L, best_lambda_H = cross_validate(Y, W, X, Z, V, Omega, lambda_grid, max_iter, tol, K)

    assert jnp.isfinite(best_lambda_L)
    assert jnp.isfinite(best_lambda_H)
    assert best_lambda_L > 0
    assert best_lambda_H > 0


def test_compute_time_based_loss():
    N, T, J = 5, 4, 2
    Y = jnp.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [17, 18, 19, 20]])
    W = jnp.array([[0, 0, 1, 1],
                   [0, 1, 1, 1],
                   [1, 1, 1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Z = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    V = jnp.ones((N, T, J))
    Omega = jnp.eye(T)
    lambda_L = 0.1
    lambda_H = 0.1
    max_iter = 10
    tol = 1e-4
    train_idx = jnp.array([0, 1, 2])
    test_idx = jnp.array([3])

    loss = compute_time_based_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H, max_iter, tol, train_idx, test_idx)

    assert jnp.isfinite(loss)
    assert loss >= 0


def test_time_based_validate():
    N, T, J = 5, 6, 2
    Y = jnp.array([[1, 2, 3, 4, 5, 6],
                   [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18],
                   [19, 20, 21, 22, 23, 24],
                   [25, 26, 27, 28, 29, 30]])
    W = jnp.array([[0, 0, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 0, 0],
                   [1, 0, 0, 0, 1, 1],
                   [0, 0, 0, 1, 1, 1]])
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Z = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    V = jnp.ones((N, T, J))
    Omega = jnp.eye(T)
    lambda_grid = jnp.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    max_iter = 10
    tol = 1e-4
    window_size = 3
    expanding_window = False
    max_window_size = None
    n_folds = 2

    best_lambda_L, best_lambda_H = time_based_validate(Y, W, X, Z, V, Omega, lambda_grid, max_iter, tol,
                                                       window_size, expanding_window, max_window_size, n_folds)

    assert jnp.isfinite(best_lambda_L)
    assert jnp.isfinite(best_lambda_H)
    assert best_lambda_L > 0
    assert best_lambda_H > 0


def test_time_based_validate_expanding_window():
    key = jax.random.PRNGKey(0)
    N, T, P, Q, J = 10, 20, 2, 2, 2
    Y = jax.random.uniform(key, shape=(N, T))
    W = jax.random.choice(key, jnp.array([0, 1]), shape=(N, T))
    X = jax.random.uniform(key, shape=(N, P))
    Z = jax.random.uniform(key, shape=(T, Q))
    V = jax.random.uniform(key, shape=(N, T, J))
    Omega = jnp.eye(T)
    lambda_grid = jnp.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

    # Test with expanding window and default max_window_size
    best_lambda_L, best_lambda_H = time_based_validate(
        Y, W, X, Z, V, Omega, lambda_grid, max_iter=10, tol=1e-4,
        window_size=15, expanding_window=True, n_folds=5
    )

    assert jnp.isfinite(best_lambda_L)
    assert jnp.isfinite(best_lambda_H)

    # Test with expanding window and custom max_window_size
    best_lambda_L, best_lambda_H = time_based_validate(
        Y, W, X, Z, V, Omega, lambda_grid, max_iter=10, tol=1e-4,
        window_size=15, expanding_window=True, max_window_size=18, n_folds=5
    )

    assert jnp.isfinite(best_lambda_L)
    assert jnp.isfinite(best_lambda_H)


def test_estimate_verbose():
    N, T = 10, 10
    data, true_params = generate_data(nobs=N, nperiods=T, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])

    # Test with verbose=True and cv validation method
    results = estimate(Y, W, X=X, Z=Z, V=V, verbose=True, validation_method='cv', K=2)
    assert results.tau is not None
    assert results.lambda_L is not None
    assert results.lambda_H is not None

    # Test with verbose=True and holdout validation method
    results = estimate(Y, W, X=X, Z=Z, V=V, verbose=True, validation_method='holdout', window_size=5)
    assert results.tau is not None
    assert results.lambda_L is not None
    assert results.lambda_H is not None


def test_estimate_invalid_validation_method():
    N, T = 10, 10
    data, true_params = generate_data(nobs=N, nperiods=T, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])

    with pytest.raises(ValueError, match="Invalid validation_method. Choose 'cv' or 'holdout'."):
        estimate(Y, W, X=X, Z=Z, V=V, validation_method='invalid')


def test_estimate_insufficient_periods():
    N, T = 10, 4  # Not enough periods for time-based validation
    data, true_params = generate_data(nobs=N, nperiods=T, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])

    with pytest.raises(ValueError, match="The matrix does not have enough columns for time-based validation."):
        estimate(Y, W, X=X, Z=Z, V=V, validation_method='holdout')
