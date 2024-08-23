from typing import Tuple

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax import jit

from .types import Array, Scalar

jax.config.update("jax_enable_x64", True)


@jit
def mask_observed(A: Array, mask: Array) -> Array:
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


@jit
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


@jit
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


@jit
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


@jit
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


@jit
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


@jit
def normalize(mat: Array) -> Tuple[Array, Array]:
    """
    Normalize the columns of the input matrix.
    Return the normalized matrix and the column norms.
    """
    if mat.size == 0:
        col_norms = jnp.zeros(mat.shape[1])
        mat_norm = jnp.zeros_like(mat)
    else:
        epsilon = 1e-10
        col_norms = jnp.linalg.norm(mat, axis=0) + epsilon
        mat_norm = mat / col_norms

    return mat_norm, col_norms


@jit
def normalize_back(H: Array, row_scales: Array, col_scales: Array) -> Array:
    """
    Rescale the rows and columns of the matrix H using the provided scales.
    """
    H_new = H.copy()

    if row_scales.size > 0:
        H_new = H_new / row_scales[:, None]

    if col_scales.size > 0:
        H_new = H_new / col_scales[None, :]

    return H_new
