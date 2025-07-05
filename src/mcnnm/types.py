from typing import TypeAlias

import jax

Array: TypeAlias = jax.Array
Scalar: TypeAlias = float | int | Array
