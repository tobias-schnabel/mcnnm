import jax
import jax.numpy as jnp
from jax import random
from jax.numpy.linalg import norm
from . import Array
from typing import Optional
import time
from .timer import timer

@jax.jit
def p_o(A: Array, mask: Array) -> Array:
    """
    Projects the matrix A onto the observed entries specified by the binary mask mask.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.
    """
    if A.shape != mask.shape:
        raise ValueError("Shapes of A and mask must match.")
    return jnp.where(mask, A, jnp.zeros_like(A))


@jax.jit
def p_perp_o(A: Array, mask: Array) -> Array:
    """
    Projects the matrix A onto the unobserved entries specified by the binary mask.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.
    """
    if A.shape != mask.shape:
        raise ValueError("Shapes of A and mask must match.")
    return jnp.where(mask, jnp.zeros_like(A), A)


@jax.jit
def shrink_lambda(A: Array, lambda_: float) -> Array:
    """
    Applies the soft-thresholding operator to the singular values of a matrix A.

    Args:
        A: The input matrix.
        lambda_: The shrinkage parameter.

    Returns:
        The matrix with soft-thresholded singular values.
    """
    u, s, v_transpose = jnp.linalg.svd(A, full_matrices=False)
    s_shrunk = jnp.maximum(s - lambda_, 0)
    return jnp.dot(u * s_shrunk, v_transpose)


@jax.jit
def frobenius_norm(A: Array) -> float:
    """
    Computes the Frobenius norm of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        The Frobenius norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return norm(A, ord='fro')


@jax.jit
def nuclear_norm(A: Array) -> float:
    """
    Computes the nuclear norm (sum of singular values) of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        The nuclear norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    _, s, _ = jnp.linalg.svd(A, full_matrices=False)
    return jnp.sum(s)


@jax.jit
def element_wise_l1_norm(A: Array) -> float:
    """
    Computes the element-wise L1 norm of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        The element-wise L1 norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return jnp.sum(jnp.abs(A))


def propose_lambda(proposed_lambda: Optional[float] = None, n_lambdas: int = 6) -> Array:
    """
    Creates a log-spaced list of proposed lambda values around a given value.

    Args:
        proposed_lambda: The proposed lambda value. If None, the default sequence is used.
        n_lambdas: The number of lambda values to generate.

    Returns:
        The sequence of proposed lambda values.
    """
    if proposed_lambda is None:
        return jnp.logspace(-3, 1, n_lambdas)
    else:
        log_proposed_lambda = jnp.log10(proposed_lambda)
        log_min = log_proposed_lambda - 2
        log_max = log_proposed_lambda + 2
        return jnp.logspace(log_min, log_max, n_lambdas)


def timer(func):
    """
    A decorator that times the execution of a function.

    Args:
        func: The function to be timed.

    Returns:
        The decorated function that prints the execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.5f} seconds")
        return result
    return wrapper


def time_fit(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None,
             V: Optional[Array] = None, Omega: Optional[Array] = None,
             lambda_L: Optional[float] = None, lambda_H: Optional[float] = None,
             return_tau: bool = True, return_lambda: bool = True,
             return_completed_L: bool = True, return_completed_Y: bool = True,
             max_iter: int = 1000, tol: float = 1e-4):
    """
    Times the execution of the fit function.

    Args:
        Y: The observed outcome matrix of shape (N, T).
        W: The binary treatment matrix of shape (N, T).
        X: The unit-specific covariates matrix of shape (N, P). If None, not included.
        Z: The time-specific covariates matrix of shape (T, Q). If None, not included.
        V: The unit-time specific covariates tensor of shape (N, T, J). If None, not included.
        Omega: The autocorrelation matrix of shape (T, T). If None, no autocorrelation is assumed.
        lambda_L: The regularization parameter for the nuclear norm of L. If None, it is selected via cross-validation.
        lambda_H: The regularization parameter for the element-wise L1 norm of H. If None, it is selected via cross-validation.
        return_tau: Whether to return the average treatment effect (tau) for the treated units.
        return_lambda: Whether to return the optimal regularization parameter lambda_L.
        return_completed_L: Whether to return the completed low-rank matrix L.
        return_completed_Y: Whether to return the completed outcome matrix Y.
        max_iter: The maximum number of iterations for the algorithm.
        tol: The tolerance for the convergence of the algorithm.

    Returns:
        The result of the fit function.
    """
    from .main import fit  # Import fit locally

    @timer
    def timed_fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, return_tau, return_lambda,
                  return_completed_L, return_completed_Y, max_iter, tol):
        N, T = Y.shape
        mask = random.bernoulli(random.PRNGKey(0), 0.8, (N,))
        Y_train, Y_test = Y[mask], Y[~mask]
        W_train, W_test = W[mask], W[~mask]
        X_train, X_test = X[mask], X[~mask] if X is not None and X.shape[1] > 0 else (None, None)
        Z_train, Z_test = Z, Z  # Z is time-specific, so it doesn't change
        V_train, V_test = V[mask, :, :], V[~mask, :, :] if V is not None and V.shape[2] > 0 else (None, None)
        return fit(Y_train, W_train, X=X_train, Z=Z_train, V=V_train, Omega=Omega, lambda_L=lambda_L, lambda_H=lambda_H,
                   return_tau=return_tau, return_lambda=return_lambda,
                   return_completed_L=return_completed_L, return_completed_Y=return_completed_Y,
                   max_iter=max_iter, tol=tol)

    return timed_fit(Y, W, X, Z, V, Omega, lambda_L, lambda_H, return_tau, return_lambda,
                     return_completed_L, return_completed_Y, max_iter, tol)