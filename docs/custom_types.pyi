import jax
from typing import Union, TypeAlias

Array: TypeAlias = jax.Array
Scalar: TypeAlias = Union[float, int, Array]
