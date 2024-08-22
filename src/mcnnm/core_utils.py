from typing import Tuple

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

from .types import Array, Scalar


def project_matrix(A: Array, mask: Array) -> Array:
    r"""
    Projects the matrix A onto the observed entries specified by the binary mask.
    Corresponds to :math:`P_{\mathcal{O}}` in the paper.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        Array: The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.

    .. math::

        P_{\mathcal{O}}(A) = A \odot \text{mask}

    where :math:`\odot` denotes the element-wise product.
    """
    if A.shape != mask.shape:
        raise ValueError(f"The shapes of A ({A.shape}) and mask ({mask.shape}) do not match.")
    return A * mask


def mask_unobserved(A: Array, mask: Array) -> Array:
    r"""
    Projects the matrix A onto the unobserved entries specified by the binary mask.
    Corresponds to :math:`P_{\mathcal{O}}^\perp` in the paper.

    Args:
        A: The input matrix.
        mask: The binary mask matrix, where 1 indicates an observed entry and 0 indicates an unobserved entry.

    Returns:
        Array: The projected matrix.

    Raises:
        ValueError: If the shapes of A and mask do not match.

    .. math::

        P_{\mathcal{O}}^\perp(A) = A \odot (\mathbf{1} - \text{mask})

    where :math:`\odot` denotes the element-wise product and :math:`\mathbf{1}` is a matrix of 1s.
    """
    if A.shape != mask.shape:
        raise ValueError(f"The shapes of A ({A.shape}) and mask ({mask.shape}) do not match.")
    return jnp.where(mask, jnp.zeros_like(A), A)


def frobenius_norm(A: Array) -> Scalar:
    """
    Computes the Frobenius norm of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        Scalar: The Frobenius norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return norm(A, ord="fro")


def nuclear_norm(A: Array) -> Scalar:
    """
    Computes the nuclear norm (sum of singular values) of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        Scalar: The nuclear norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    _, s, _ = jnp.linalg.svd(A, full_matrices=False)
    return jnp.sum(s)


def element_wise_l1_norm(A: Array) -> Scalar:
    """
    Computes the element-wise L1 norm of a matrix A.

    Args:
        A: The input matrix.

    Returns:
        Scalar: The element-wise L1 norm of the matrix A.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return jnp.sum(jnp.abs(A))


def shrink_lambda(A: Array, lambda_: Scalar) -> Array:
    """
    Applies the soft-thresholding operator to the singular values of a matrix A.

    Args:
        A: The input matrix.
        lambda_: The shrinkage parameter.

    Returns:
        Array: The matrix with soft-thresholded singular values.
    """
    u, s, vt = jnp.linalg.svd(A, full_matrices=False)
    s_shrunk = jnp.maximum(s - lambda_, 0)
    return u @ jnp.diag(s_shrunk) @ vt


def initialize_params(
    Y: Array, X: Array, Z: Array, V: Array, use_unit_fe: bool, use_time_fe: bool
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Initialize parameters for the MC-NNM model.

    Args:
        Y (Array): The observed outcome matrix of shape (N, T).
        X (Array): The unit-specific covariates matrix of shape (N, P).
        Z (Array): The time-specific covariates matrix of shape (T, Q).
        V (Array): The unit-time specific covariates tensor of shape (N, T, J).
        use_unit_fe (bool): Whether to use unit fixed effects.
        use_time_fe (bool): Whether to use time fixed effects.

    Returns:
        Tuple[Array, Array, Array, Array, Array]: A tuple containing initial values for L,
        H, gamma, delta, and beta.
    """
    N, T = Y.shape
    L = jnp.zeros_like(Y)
    H = jnp.zeros((X.shape[1] + N, Z.shape[1] + T))

    def init_gamma(_):
        return jnp.zeros(N)

    def init_delta(_):
        return jnp.zeros(T)

    gamma = jax.lax.cond(
        use_unit_fe,
        init_gamma,
        lambda _: jnp.zeros(N),
        operand=None,
    )

    delta = jax.lax.cond(
        use_time_fe,
        init_delta,
        lambda _: jnp.zeros(T),
        operand=None,
    )

    beta_shape = max(V.shape[2], 1)
    beta = jnp.zeros((beta_shape,))
    beta = jax.lax.cond(
        V.shape[2] > 0, lambda _: beta, lambda _: jnp.zeros_like(beta), operand=None
    )

    return L, H, gamma, delta, beta
