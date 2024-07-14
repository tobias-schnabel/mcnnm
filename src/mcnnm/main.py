import jax
import jax.numpy as jnp
from typing import Optional, Union

Array = Union[jnp.ndarray, jnp.Array]


def p_O(A: Array, O: Array) -> Array:
    """
    Projects the matrix A onto the observed entries specified by the binary mask O.

    Args:
        A: The input matrix.
        O: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        The projected matrix.

    Raises:
        ValueError: If the shapes of A and O do not match.
    """
    if A.shape != O.shape:
        raise ValueError("Shapes of A and O must match.")
    return jnp.where(O, A, jnp.zeros_like(A))


def p_perp_O(A: Array, O: Array) -> Array:
    """
    Projects the matrix A onto the unobserved entries specified by the binary mask O.

    Args:
        A: The input matrix.
        O: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        The projected matrix.

    Raises:
        ValueError: If the shapes of A and O do not match.
    """
    if A.shape != O.shape:
        raise ValueError("Shapes of A and O must match.")
    return jnp.where(O, jnp.zeros_like(A), A)


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
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
    s_shrunk = jnp.maximum(s - lambda_, 0)
    return jnp.dot(U * s_shrunk, Vt)


def objective_function(
        Y: Array, L: Array, Omega: Optional[Array] = None, gamma: Optional[Array] = None,
        delta: Optional[Array] = None, beta: Optional[Array] = None, H: Optional[Array] = None,
        X: Optional[Array] = None
) -> float:
    """
    Computes the objective function value for the MC-NNM estimator (Equation 18).

    Args:
        Y: The observed outcome matrix.
        L: The low-rank matrix.
        Omega: The autocorrelation matrix. If None, no autocorrelation is assumed.
        gamma: The unit fixed effects vector. If None, unit fixed effects are not included.
        delta: The time fixed effects vector. If None, time fixed effects are not included.
        beta: The coefficient vector for the covariates. If None, unit-time specific covariates are not included.
        H: The coefficient matrix for the  covariates. If None, unit and time specific covariates are not included.
        X: The matrix of unit and time specific covariates. If None, unit and time specific covariates are not included.

    Returns:
        The objective function value.
    """
    N, T = Y.shape
    if gamma is None:
        gamma = jnp.zeros(N)
    if delta is None:
        delta = jnp.zeros(T)
    if beta is None:
        beta = jnp.zeros((N, T))
    if H is None or X is None:
        H = jnp.zeros((N + Y.shape[1], T + Y.shape[0]))
        X = jnp.zeros((N, T))
    if Omega is None:
        Omega_inv = jnp.eye(T)
    else:
        Omega_inv = jnp.linalg.inv(Omega)

    residual = Y - L - jnp.outer(gamma, jnp.ones(T)) - jnp.outer(jnp.ones(N), delta) - beta - jnp.dot(X, H)
    return jnp.sum(jnp.dot(residual, Omega_inv) * residual) / (N * T)
