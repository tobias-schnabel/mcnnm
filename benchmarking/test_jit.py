import jax
import jax.numpy as jnp
import time
from mcnnm.estimate import estimate
from mcnnm.util import generate_data
import pytest

# Enable 64-bit floats
# jax.config.update("jax_enable_x64", True)

# Enable compilation logging
# jax.config.update("jax_log_compiles", True)


def run_estimate(N, T, jit_enabled, pre_compiled_fn=None):
    """Run estimate function and time it."""
    if not jit_enabled:
        jax.config.update("jax_disable_jit", True)
    else:
        jax.config.update("jax_disable_jit", False)

    # Generate data
    data, _ = generate_data(nobs=N, nperiods=T, X_cov=True, Z_cov=True, V_cov=True)
    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)

    # Timed run
    start_time = time.time()
    if pre_compiled_fn:
        result = pre_compiled_fn(Y, W)
    else:
        result = estimate(Y, W)
    result.tau.block_until_ready()  # Ensure computation is complete
    end_time = time.time()

    execution_time = end_time - start_time

    if not jit_enabled:
        jax.config.update("jax_disable_jit", False)

    return execution_time


def compile_estimate(N, T):
    """Compile the estimate function for given dimensions and time the compilation."""
    data, _ = generate_data(nobs=N, nperiods=T, X_cov=True, Z_cov=True, V_cov=True)
    Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
    W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)

    @jax.jit
    def compiled_estimate(Y, W):
        return estimate(Y, W)

    # Time the compilation
    start_time = time.time()
    _ = compiled_estimate(Y, W).tau.block_until_ready()
    compilation_time = time.time() - start_time

    return compiled_estimate, compilation_time


@pytest.mark.parametrize(
    "N,T", [(10, 10), (50, 50), (100, 100), (1000, 1000), (100, 3000), (3000, 100), (2000, 2000)]
)
def test_estimate_jit_performance(N, T):
    print(f"\nTesting with matrix size {N}x{T}")

    # Run with JIT disabled
    time_no_jit = run_estimate(N, T, jit_enabled=False)
    print(f"Execution time without JIT: {time_no_jit:.4f} seconds")

    # Compile the function and measure compilation time
    compiled_fn, compilation_time = compile_estimate(N, T)
    print(f"Compilation time: {compilation_time:.4f} seconds")

    # Run with JIT enabled and pre-compiled function
    time_with_jit = run_estimate(N, T, jit_enabled=True, pre_compiled_fn=compiled_fn)
    print(f"Execution time with JIT (pre-compiled): {time_with_jit:.4f} seconds")

    # Compare times
    ratio = time_with_jit / time_no_jit
    if ratio > 1:
        print(f"Slowdown with JIT: {ratio:.2f}x")
    else:
        print(f"Speedup with JIT: {1 / ratio:.2f}x")

    assert ratio < 1, f"JIT should provide speedup, but got slowdown of {ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__])
