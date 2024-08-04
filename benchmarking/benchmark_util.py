import jax
import jax.numpy as jnp
import time
from mcnnm.util import (
    p_o,
    p_perp_o,
    shrink_lambda,
    frobenius_norm,
    nuclear_norm,
    element_wise_l1_norm,
    propose_lambda,
    initialize_params,
    check_inputs,
    generate_time_based_validate_defaults,
    generate_data,
)

# Enable 64-bit floats
jax.config.update("jax_enable_x64", True)

# Enable compilation logging
# jax.config.update('jax_log_compiles', True)


def benchmark_function(func, args, n_runs=100):
    # Compile the function
    jitted_func = jax.jit(func)

    # Measure compilation time
    start_time = time.time()
    _ = jitted_func(*args).block_until_ready()
    compilation_time = time.time() - start_time

    # Measure execution time (non-JIT)
    start_time = time.time()
    for _ in range(n_runs):
        _ = func(*args).block_until_ready()
    non_jit_time = (time.time() - start_time) / n_runs

    # Measure execution time (JIT)
    start_time = time.time()
    for _ in range(n_runs):
        _ = jitted_func(*args).block_until_ready()
    jit_time = (time.time() - start_time) / n_runs

    return compilation_time, non_jit_time, jit_time


def run_benchmarks():
    sizes = [(10, 10), (50, 50), (100, 100), (1000, 1000), (2000, 2000)]

    for N, T in sizes:
        print(f"\nBenchmarking with size {N}x{T}")

        # Generate sample data
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, shape=(N, T))
        mask = jax.random.choice(key, jnp.array([0, 1]), shape=(N, T))

        # Benchmark p_o
        args = (A, mask)
        compilation_time, non_jit_time, jit_time = benchmark_function(p_o, args)
        print(
            f"p_o: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark p_perp_o
        args = (A, mask)
        compilation_time, non_jit_time, jit_time = benchmark_function(p_perp_o, args)
        print(
            f"p_perp_o: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark shrink_lambda
        args = (A, 0.1)
        compilation_time, non_jit_time, jit_time = benchmark_function(shrink_lambda, args)
        print(
            f"shrink_lambda: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark frobenius_norm
        args = (A,)
        compilation_time, non_jit_time, jit_time = benchmark_function(frobenius_norm, args)
        print(
            f"frobenius_norm: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark nuclear_norm
        args = (A,)
        compilation_time, non_jit_time, jit_time = benchmark_function(nuclear_norm, args)
        print(
            f"nuclear_norm: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark element_wise_l1_norm
        args = (A,)
        compilation_time, non_jit_time, jit_time = benchmark_function(element_wise_l1_norm, args)
        print(
            f"element_wise_l1_norm: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark propose_lambda
        args = (0.1, 6)
        compilation_time, non_jit_time, jit_time = benchmark_function(propose_lambda, args)
        print(
            f"propose_lambda: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark initialize_params
        Y = jnp.random.normal(size=(N, T))
        X = jnp.random.normal(size=(N, 2))
        Z = jnp.random.normal(size=(T, 2))
        V = jnp.random.normal(size=(N, T, 2))
        args = (Y, X, Z, V)
        compilation_time, non_jit_time, jit_time = benchmark_function(initialize_params, args)
        print(
            f"initialize_params: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark check_inputs
        W = jnp.random.choice([0, 1], size=(N, T))
        args = (Y, W, X, Z, V, None)
        compilation_time, non_jit_time, jit_time = benchmark_function(check_inputs, args)
        print(
            f"check_inputs: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark generate_time_based_validate_defaults
        args = (Y, 10, 10)
        compilation_time, non_jit_time, jit_time = benchmark_function(
            generate_time_based_validate_defaults, args
        )
        print(
            f"generate_time_based_validate_defaults: Compilation time: {compilation_time:.4f}s, "
            f"Non-JIT time: {non_jit_time:.4f}s, JIT time: {jit_time:.4f}s"
        )

        # Benchmark generate_data
        # Note: generate_data uses numpy, so we won't JIT it
        start_time = time.time()
        _ = generate_data(nobs=N, nperiods=T)
        generation_time = time.time() - start_time
        print(f"generate_data: Generation time: {generation_time:.4f}s")


if __name__ == "__main__":
    run_benchmarks()
