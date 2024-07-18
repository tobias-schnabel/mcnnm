import jax
import jax.numpy as jnp
from typing import Optional, Tuple, NamedTuple
from . import Array
from mcnnm.util import *
from jax import jit, vmap, lax

@jit
def update_L(Y_adj: Array, L: Array, Omega: Array, O: Array, lambda_L: float) -> Array:
    return shrink_lambda(p_o(jnp.dot(Y_adj, Omega), O) + p_perp_o(L, O), lambda_L * jnp.sum(O) / 2)

@jit
def update_H(X_tilde: Array, Y_adj: Array, Z_tilde: Array, lambda_H: float) -> Array:
    return shrink_lambda(jnp.linalg.lstsq(X_tilde, jnp.dot(Y_adj, Z_tilde))[0], lambda_H)


@jit
def update_gamma_delta_beta(Y_adj: Array, V: Array) -> Tuple[Array, Array, Array]:
    N, T = Y_adj.shape
    gamma = jnp.mean(Y_adj, axis=1)
    delta = jnp.mean(Y_adj - gamma[:, jnp.newaxis], axis=0)

    def true_fun(_):
        return jnp.linalg.lstsq(V.reshape(N * T, -1), Y_adj.reshape(N * T))[0]

    def false_fun(_):
        return jnp.zeros((V.shape[-1],))  # Ensure this matches the shape of true_fun output

    beta = jax.lax.cond(
        V.size > 0,
        true_fun,
        false_fun,
        operand=None
    )
    return gamma, delta, beta

@jit
def fit_step(Y: Array, W: Array, X_tilde: Array, Z_tilde: Array, V: Array, Omega: Array,
             lambda_L: float, lambda_H: float, L: Array, H: Array, gamma: Array, delta: Array, beta: Array) -> Tuple:
    N, T = Y.shape
    O = (W == 0)

    Y_adj = Y - jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T)) - gamma[:, jnp.newaxis] - delta[jnp.newaxis, :]
    Y_adj = Y_adj - jnp.sum(V * beta, axis=-1) if V.size > 0 else Y_adj
    L_new = update_L(Y_adj, L, Omega, O, lambda_L)

    Y_adj = Y - L_new - gamma[:, jnp.newaxis] - delta[jnp.newaxis, :]
    Y_adj = Y_adj - jnp.sum(V * beta, axis=-1) if V.size > 0 else Y_adj
    H_new = update_H(X_tilde, Y_adj, Z_tilde, lambda_H)

    Y_adj = Y - L_new - jnp.dot(X_tilde, jnp.dot(H_new, Z_tilde.T))
    Y_adj = Y_adj - jnp.sum(V * beta, axis=-1) if V.size > 0 else Y_adj
    gamma_new, delta_new, beta_new = update_gamma_delta_beta(Y_adj, V)

    return L_new, H_new, gamma_new, delta_new, beta_new


def fit(Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array,
        lambda_L: float, lambda_H: float, initial_params: Tuple,
        max_iter: int, tol: float) -> Tuple:
    # Unpack initial parameters
    L, H, gamma, delta, beta = initial_params

    # Compute dimensions and augmented covariate matrices
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))

    # Ensure beta has the correct shape
    beta = jnp.zeros((V.shape[-1],)) if V.size > 0 else jnp.zeros((0,))

    # Define the condition function for the while loop
    def cond_fn(state):
        i, L, _, _, _, _, prev_L = state
        return (i < max_iter) & (jnp.linalg.norm(L - prev_L, ord='fro') >= tol)

    # Define the body function for the while loop
    def body_fn(state):
        i, L, H, gamma, delta, beta, prev_L = state
        L_new, H_new, gamma_new, delta_new, beta_new = fit_step(Y, W, X_tilde, Z_tilde, V, Omega, lambda_L, lambda_H, L,
                                                                H, gamma, delta, beta)
        return i + 1, L_new, H_new, gamma_new, delta_new, beta_new, L

    # Set the initial state of the while loop
    initial_state = (0, L, H, gamma, delta, beta, jnp.zeros_like(L))

    # Run the while loop until convergence or max iterations
    _, L, H, gamma, delta, beta, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return L, H, gamma, delta, beta



def compute_cv_loss(Y: Array, W: Array, X: Array, Z: Array, V: Array, Omega: Array,
                    lambda_L: float, lambda_H: float, max_iter: int, tol: float) -> float:
    N = Y.shape[0]
    loss = 0.0

    key = jax.random.PRNGKey(0)
    mask = jax.random.bernoulli(key, 0.8, (N,))
    train_idx = jnp.where(mask)[0]
    test_idx = jnp.where(~mask)[0]

    Y_train, Y_test = Y[train_idx], Y[test_idx]
    W_train, W_test = W[train_idx], W[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    V_train, V_test = V[train_idx], V[test_idx]

    initial_params = initialize_params(Y_train, W_train, X_train, Z, V_train)

    L, H, gamma, delta, beta = fit(Y_train, W_train, X_train, Z, V_train, Omega,
                                   lambda_L, lambda_H, initial_params, max_iter, tol)

    Y_pred = (L[test_idx] + jnp.outer(gamma[test_idx], jnp.ones(Z.shape[0])) +
              jnp.outer(jnp.ones(test_idx.shape[0]), delta))

    if V_test.shape[2] > 0:
        Y_pred += jnp.sum(V_test * beta, axis=2)

    O_test = (W_test == 0)
    loss += jnp.sum((Y_test - Y_pred) ** 2 * O_test) / jnp.sum(O_test)
    return loss

def cross_validate(Y: Array, W: Array, X: Array, Z: Array, V: Array,
                   Omega: Array, lambda_grid: Array, max_iter: int, tol: float, K: int = 5) -> Tuple[float, float]:
    best_lambda_L = None
    best_lambda_H = None
    best_loss = jnp.inf

    for lambda_L, lambda_H in lambda_grid:
        loss = 0.0
        for k in range(K):
            loss += compute_cv_loss(Y, W, X, Z, V, Omega, lambda_L, lambda_H, max_iter, tol)
        loss /= K

        if loss < best_loss:
            best_lambda_L = lambda_L
            best_lambda_H = lambda_H
            best_loss = loss

    return best_lambda_L, best_lambda_H

def compute_treatment_effect(Y: Array, L: Array, gamma: Array, delta: Array, beta: Array, H: Array,
                             X: Array, W: Array, Z: Array, V: Array) -> float:
    N, T = Y.shape
    X_tilde = jnp.hstack((X, jnp.eye(N)))
    Z_tilde = jnp.hstack((Z, jnp.eye(T)))
    Y_completed = L + jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)

    if X.shape[1] > 0 and Z.shape[1] > 0:
        Y_completed += jnp.dot(X_tilde, jnp.dot(H, Z_tilde.T))

    if V.shape[2] > 0:
        Y_completed += jnp.sum(V * beta[None, None, :], axis=2)

    treated_units = jnp.sum(W)
    tau = jnp.sum((Y - Y_completed) * W) / treated_units
    return tau

class MCNNMResults(NamedTuple):
    tau: Optional[float] = None
    lambda_L: Optional[float] = None
    lambda_H: Optional[float] = None
    L: Optional[Array] = None
    Y_completed: Optional[Array] = None
    gamma: Optional[Array] = None
    delta: Optional[Array] = None
    beta: Optional[Array] = None
    H: Optional[Array] = None

def check_inputs(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None,
                 V: Optional[Array] = None, Omega: Optional[Array] = None) -> Tuple:
    N, T = Y.shape
    if W.shape != (N, T):
        raise ValueError("The shape of W must match the shape of Y.")
    X = jnp.zeros((N, 0)) if X is None else X
    Z = jnp.zeros((T, 0)) if Z is None else Z
    V = jnp.zeros((N, T, 0)) if V is None else V
    Omega = jnp.eye(T) if Omega is None else Omega
    return X, Z, V, Omega

def estimate(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None,
             V: Optional[Array] = None, Omega: Optional[Array] = None, lambda_L: Optional[float] = None,
             lambda_H: Optional[float] = None, return_tau: bool = True, return_lambda: bool = True,
             return_completed_L: bool = True, return_completed_Y: bool = True, return_fixed_effects: bool = False,
             return_covariate_coefficients: bool = False, max_iter: int = 1000, tol: float = 1e-4,
             verbose: bool = False, K: int = 5, n_lambda_L = 6, n_lambda_H = 6) -> MCNNMResults:
    X, Z, V, Omega = check_inputs(Y, W, X, Z, V, Omega)
    N, T = Y.shape

    if lambda_L is None or lambda_H is None:
        if verbose:
            print_with_timestamp("Cross-validating lambda_L, lambda_H")
        lambda_grid = jnp.array(jnp.meshgrid(propose_lambda(None, n_lambda_L), propose_lambda(None, n_lambda_H))).T.reshape(-1, 2)
        lambda_L, lambda_H = cross_validate(Y, W, X, Z, V, Omega, lambda_grid, max_iter // 10, tol * 10, K)
        if verbose:
            print_with_timestamp(f"Selected lambda_L: {lambda_L:.4f}, lambda_H: {lambda_H:.4f}")

    initial_params = initialize_params(Y, W, X, Z, V)
    L, H, gamma, delta, beta = fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, initial_params, max_iter, tol)

    results = {}
    if return_tau:
        tau = compute_treatment_effect(Y, L, gamma, delta, beta, H, X, W, Z, V)
        results['tau'] = tau
    if return_lambda:
        results['lambda_L'] = lambda_L
        results['lambda_H'] = lambda_H
    if return_completed_L:
        results['L'] = L
    if return_completed_Y:
        Y_completed = L + jnp.dot(jnp.hstack((X, jnp.eye(N))), jnp.dot(H, jnp.hstack((Z, jnp.eye(T))).T))
        Y_completed += jnp.outer(gamma, jnp.ones(T)) + jnp.outer(jnp.ones(N), delta)
        if V.shape[2] > 0:
            Y_completed += jnp.sum(V * beta, axis=2)
        results['Y_completed'] = Y_completed
    if return_fixed_effects:
        results['gamma'] = gamma
        results['delta'] = delta
    if return_covariate_coefficients:
        results['beta'] = beta
        results['H'] = H

    return MCNNMResults(**results)