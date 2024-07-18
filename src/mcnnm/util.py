import jax
import jax.numpy as jnp
from jax import random
from jax.numpy.linalg import norm
from . import Array
from typing import Optional
import time
from datetime import datetime
from typing import Optional, Tuple, NamedTuple

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
        return jnp.logspace(-3, 0, n_lambdas)
    else:
        log_proposed_lambda = jnp.log10(proposed_lambda)
        log_min = log_proposed_lambda - 2
        log_max = log_proposed_lambda + 2
        return jnp.logspace(log_min, log_max, n_lambdas)


def initialize_params(Y: Array, W: Array, X: Array, Z: Array, V: Array) -> Tuple:
    """
    Initialize parameters for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).

    Returns:
        Tuple: A tuple containing initial values for L, H, gamma, delta, and beta.
    """
    N, T = Y.shape
    L = jnp.zeros_like(Y)
    H = jnp.zeros((X.shape[1] + N, Z.shape[1] + T))
    gamma = jnp.zeros(N)
    delta = jnp.zeros(T)
    beta = jnp.zeros(V.shape[2]) if V.shape[2] > 0 else jnp.zeros(0)
    return L, H, gamma, delta, beta


def check_inputs(Y: Array, W: Array, X: Optional[Array] = None, Z: Optional[Array] = None,
                 V: Optional[Array] = None, Omega: Optional[Array] = None) -> Tuple:
    """
    Check and preprocess input arrays for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        W (Array): The binary treatment matrix of shape (N, T).
        X (Optional[Array]): The unit-specific covariates matrix of shape (N, P). Default is None.
        Z (Optional[Array]): The time-specific covariates matrix of shape (T, Q). Default is None.
        V (Optional[Array]): The unit-time specific covariates tensor of shape (N, T, J). Default is None.
        Omega (Optional[Array]): The autocorrelation matrix of shape (T, T). Default is None.

    Returns:
        Tuple: A tuple containing preprocessed X, Z, V, and Omega arrays.

    Raises:
        ValueError: If the shape of W does not match the shape of Y.
    """
    N, T = Y.shape
    if W.shape != (N, T):
        raise ValueError("The shape of W must match the shape of Y.")
    X = jnp.zeros((N, 0)) if X is None else X
    Z = jnp.zeros((T, 0)) if Z is None else Z
    V = jnp.zeros((N, T, 0)) if V is None else V
    Omega = jnp.eye(T) if Omega is None else Omega
    return X, Z, V, Omega



def print_with_timestamp(message: str) -> None:
    """
    Print a message with a human-readable timestamp (hhmmss) prefix.

    Args:
        message (str): The message to be printed.

    Returns:
        None
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")