import jax
import jax.numpy as jnp
from typing import Union

Array = Union[jnp.ndarray, jnp.DeviceArray]

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