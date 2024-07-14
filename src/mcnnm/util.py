import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from typing import Union

Array = Union[jnp.ndarray, jnp.DeviceArray]  # Define a type alias for array-like objects

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