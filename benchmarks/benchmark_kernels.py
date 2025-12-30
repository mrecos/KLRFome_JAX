"""Benchmark kernel computations."""

import time
import jax.numpy as jnp
import jax.random as random
from klrfome.kernels.rbf import RBFKernel
from klrfome.kernels.rff import RandomFourierFeatures


def benchmark_rbf_kernel(n_samples: int = 1000, n_features: int = 10):
    """Benchmark RBF kernel computation."""
    key = random.PRNGKey(42)
    X = random.normal(key, (n_samples, n_features))
    
    kernel = RBFKernel(sigma=1.0)
    
    # Warm up
    _ = kernel(X, X)
    
    # Time computation
    start = time.time()
    K = kernel(X, X)
    elapsed = time.time() - start
    
    print(f"RBF Kernel: {n_samples}x{n_samples} matrix, {n_features} features")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Throughput: {n_samples**2 / elapsed:.2f} kernel evaluations/second")
    
    return elapsed


def benchmark_rff_kernel(
    n_samples: int = 1000, 
    n_features: int = 10,
    n_rff_features: int = 256
):
    """Benchmark RFF kernel computation."""
    key = random.PRNGKey(42)
    X = random.normal(key, (n_samples, n_features))
    
    kernel = RandomFourierFeatures(sigma=1.0, n_features=n_rff_features, seed=42)
    
    # Warm up
    _ = kernel(X, X)
    
    # Time computation
    start = time.time()
    K = kernel(X, X)
    elapsed = time.time() - start
    
    print(f"RFF Kernel: {n_samples}x{n_samples} matrix, {n_features} features, {n_rff_features} RFF features")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Throughput: {n_samples**2 / elapsed:.2f} kernel evaluations/second")
    
    return elapsed


def compare_rff_vs_exact():
    """Compare RFF vs exact RBF performance."""
    sizes = [100, 500, 1000, 2000]
    
    print("=" * 60)
    print("RFF vs Exact RBF Kernel Comparison")
    print("=" * 60)
    
    for n in sizes:
        print(f"\nSize: {n}x{n}")
        print("-" * 60)
        
        exact_time = benchmark_rbf_kernel(n_samples=n)
        rff_time = benchmark_rff_kernel(n_samples=n, n_rff_features=256)
        
        speedup = exact_time / rff_time
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    compare_rff_vs_exact()

