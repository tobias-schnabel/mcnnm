from mcnnm.util import initialize_params
import pytest
import jax.numpy as jnp
from jax import random
from mcnnm.estimate import *
from mcnnm.util import generate_data
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)

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
    Y_adj = jnp.array([[1, 2], [3, 4]])
    V = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    gamma, delta, beta = update_gamma_delta_beta(Y_adj, V)
    assert gamma.shape == (2,)
    assert delta.shape == (2,)
    assert beta.shape == (2,)
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(jnp.isfinite(beta))

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
    L_new, H_new, gamma_new, delta_new, beta_new = fit_step(Y, W, X_tilde, Z_tilde, V, Omega, lambda_L, lambda_H, L, H, gamma, delta, beta)
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

@pytest.mark.timeout(180)
def test_fit():
    nobs, nperiods = 50, 10
    data, true_params = generate_data(nobs=nobs, nperiods=nperiods, seed=42)
    Y = jnp.array(data.pivot(index='unit', columns='period', values='y').values)
    W = jnp.array(data.pivot(index='unit', columns='period', values='treat').values)
    X, Z, V = jnp.array(true_params['X']), jnp.array(true_params['Z']), jnp.array(true_params['V'])
    Omega = jnp.eye(nperiods)
    lambda_L = 0.1
    lambda_H = 0.1
    initial_params = (jnp.zeros_like(Y), jnp.zeros((X.shape[1] + nobs, Z.shape[1] + nperiods)), jnp.zeros(nobs), jnp.zeros(nperiods), jnp.zeros(V.shape[2]))
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
    Y_completed, lambda_L, lambda_H= complete_matrix(Y, W, X=X, Z=Z, V=V, K=2)
    assert Y_completed.shape == Y.shape
    assert jnp.all(jnp.isfinite(Y_completed))
    assert not jnp.any(jnp.isnan(Y_completed))
    assert jnp.isfinite(lambda_L)
    assert jnp.isfinite(lambda_H)

    # Optional: Check if the completed values are within a reasonable range
    assert jnp.all(Y_completed >= Y.min() - 1) and jnp.all(Y_completed <= Y.max() + 1)
