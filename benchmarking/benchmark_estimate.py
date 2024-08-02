import jax
import jax.numpy as jnp
import time
from mcnnm.estimate import (
    update_L,
    update_H,
    update_gamma_delta_beta,
    compute_beta,
    fit_step,
    fit,
    compute_cv_loss,
    cross_validate,
    # time_based_validate,
    compute_treatment_effect,
    estimate,
    complete_matrix,
)
from mcnnm.util import generate_data, initialize_params

# Enable 64-bit floats
jax.config.update("jax_enable_x64", True)

# Enable compilation logging
# jax.config.update('jax_log_compiles', True)


def benchmark_function(func, args, n_runs=100):
    # Compile the function
    jitted_func = jax.jit(func)

    # Measure compilation time
    start_time = time.time()
    result = jitted_func(*args)
    if isinstance(result, tuple):
        for item in result:
            item.block_until_ready()
    else:
        result.block_until_ready()
    compilation_time = time.time() - start_time

    # Measure execution time (non-JIT)
    start_time = time.time()
    for _ in range(n_runs):
        result = func(*args)
        if isinstance(result, tuple):
            for item in result:
                item.block_until_ready()
        else:
            result.block_until_ready()
    non_jit_time = (time.time() - start_time) / n_runs

    # Measure execution time (JIT)
    start_time = time.time()
    for _ in range(n_runs):
        result = jitted_func(*args)
        if isinstance(result, tuple):
            for item in result:
                item.block_until_ready()
        else:
            result.block_until_ready()
    jit_time = (time.time() - start_time) / n_runs

    return compilation_time, non_jit_time, jit_time


def run_benchmarks():
    sizes = [(10, 10), (50, 50), (100, 100), (1000, 1000), (2000, 2000)]

    for N, T in sizes:
        print(f"\nBenchmarking with size {N}x{T}")

        # Generate sample data
        data, _ = generate_data(nobs=N, nperiods=T, X_cov=True, Z_cov=True, V_cov=True)
        Y = jnp.array(data.pivot(index="unit", columns="period", values="y").values)
        W = jnp.array(data.pivot(index="unit", columns="period", values="treat").values)
        X, Z, V, Omega = jnp.ones((N, 2)), jnp.ones((T, 2)), jnp.ones((N, T, 2)), jnp.eye(T)

        # Initialize parameters
        L, H, gamma, delta, beta = initialize_params(Y, X, Z, V)

        # Benchmark update_L
        args = (Y, L, Omega, W == 0, 0.1)
        compilation_time, non_jit_time, jit_time = benchmark_function(update_L, args)
        print(
            f"update_L: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark update_H
        X_tilde, Z_tilde = jnp.hstack((X, jnp.eye(N))), jnp.hstack((Z, jnp.eye(T)))
        args = (X_tilde, Y, Z_tilde, 0.1)
        compilation_time, non_jit_time, jit_time = benchmark_function(update_H, args)
        print(
            f"update_H: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark update_gamma_delta_beta
        args = (Y, V)
        compilation_time, non_jit_time, jit_time = benchmark_function(update_gamma_delta_beta, args)
        print(
            f"update_gamma_delta_beta: Compilation time: {compilation_time:.4f}s, "
            f"Non-JIT time: {non_jit_time:.4f}s, JIT time: {jit_time:.4f}s"
        )

        # Benchmark compute_beta
        args = (V, Y)
        compilation_time, non_jit_time, jit_time = benchmark_function(compute_beta, args)
        print(
            f"compute_beta: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark fit_step
        args = (Y, W, X_tilde, Z_tilde, V, Omega, 0.1, 0.1, L, H, gamma, delta, beta)
        compilation_time, non_jit_time, jit_time = benchmark_function(fit_step, args)
        print(
            f"fit_step: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark fit
        initial_params = (L, H, gamma, delta, beta)
        args = (Y, W, X, Z, V, Omega, 0.1, 0.1, initial_params, 100, 1e-4)
        compilation_time, non_jit_time, jit_time = benchmark_function(fit, args, n_runs=10)
        print(
            f"fit: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark compute_cv_loss
        args = (Y, W, X, Z, V, Omega, 0.1, 0.1, 100, 1e-4)
        compilation_time, non_jit_time, jit_time = benchmark_function(
            compute_cv_loss, args, n_runs=10
        )
        print(
            f"compute_cv_loss: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark cross_validate
        lambda_grid = jnp.array([(0.1, 0.1), (0.2, 0.2)])
        args = (Y, W, X, Z, V, Omega, lambda_grid, 100, 1e-4, 5)
        compilation_time, non_jit_time, jit_time = benchmark_function(
            cross_validate, args, n_runs=1
        )
        print(
            f"cross_validate: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # # Benchmark time_based_validate
        # args = (Y, W, X, Z, V, Omega, lambda_grid, 100, 1e-4, 5, 1, 1, 5, T)
        # compilation_time, non_jit_time, jit_time = benchmark_function(time_based_validate, args, n_runs=1)
        # print(
        #     f"time_based_validate: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s,
        #     JIT time: {jit_time:.4f}s")

        # Benchmark compute_treatment_effect
        args = (Y, L, gamma, delta, beta, H, X, W, Z, V)
        compilation_time, non_jit_time, jit_time = benchmark_function(
            compute_treatment_effect, args
        )
        print(
            f"compute_treatment_effect: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark estimate
        args = (Y, W, X, Z, V, Omega)
        compilation_time, non_jit_time, jit_time = benchmark_function(estimate, args, n_runs=1)
        print(
            f"estimate: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )

        # Benchmark complete_matrix
        args = (Y, W, X, Z, V, Omega)
        compilation_time, non_jit_time, jit_time = benchmark_function(
            complete_matrix, args, n_runs=1
        )
        print(
            f"complete_matrix: Compilation time: {compilation_time:.4f}s, Non-JIT time: {non_jit_time:.4f}s, "
            f"JIT time: {jit_time:.4f}s"
        )


if __name__ == "__main__":
    run_benchmarks()
