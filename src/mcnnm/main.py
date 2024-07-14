import jax
import jax.numpy as jnp
from typing import Union

Array = Union[jnp.ndarray, jnp.Array]

def P_O(A: Array, O: Array) -> Array:
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


def P_perp_O(A: Array, O: Array) -> Array:
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