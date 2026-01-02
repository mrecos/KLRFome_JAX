#!/usr/bin/env python3
"""
Benchmark comparison between mean embedding and Wasserstein kernels.

This script compares the two kernel types on synthetic data to help
understand when each approach is preferable.

Usage:
    python benchmarks/compare_kernels.py
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import time
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome import KLRfome, TrainingData, SampleCollection


def generate_standard_test_data(
    key: random.PRNGKey,
    n_sites: int = 20,
    n_background: int = 40,
    samples_per_location: int = 20,
    n_features: int = 3,
    separation: float = 2.0
) -> Tuple[TrainingData, Dict]:
    """
    Generate standard test data where distributions differ mainly by mean.
    
    This is a scenario where mean embeddings should work well.
    """
    keys = random.split(key, n_sites + n_background)
    collections = []
    
    # Sites: samples centered at origin
    for i in range(n_sites):
        samples = random.normal(keys[i], (samples_per_location, n_features))
        collections.append(SampleCollection(
            samples=samples, label=1, id=f"site_{i}"
        ))
    
    # Background: samples shifted
    for i in range(n_background):
        samples = random.normal(keys[n_sites + i], (samples_per_location, n_features)) + separation
        collections.append(SampleCollection(
            samples=samples, label=0, id=f"bg_{i}"
        ))
    
    training_data = TrainingData(
        collections=collections,
        feature_names=[f"var_{i}" for i in range(n_features)]
    )
    
    return training_data, {
        'type': 'standard',
        'n_sites': n_sites,
        'n_background': n_background,
        'separation': separation
    }


def generate_bimodal_test_data(
    key: random.PRNGKey,
    n_sites: int = 20,
    n_background: int = 40,
    samples_per_location: int = 20,
    n_features: int = 3
) -> Tuple[TrainingData, Dict]:
    """
    Generate bimodal vs unimodal test data.
    
    Sites have bimodal distributions (two clusters), background is unimodal.
    Both have similar means, so mean embeddings will struggle.
    This is where Wasserstein should significantly outperform.
    """
    collections = []
    key_idx = 0
    
    # Sites: bimodal (mixture of two Gaussians)
    for i in range(n_sites):
        k1, k2 = random.split(random.PRNGKey(key_idx))
        key_idx += 1
        # Half samples from mode 1 (-2), half from mode 2 (+2)
        mode1 = random.normal(k1, (samples_per_location // 2, n_features)) * 0.5 - 2.0
        mode2 = random.normal(k2, (samples_per_location - samples_per_location // 2, n_features)) * 0.5 + 2.0
        samples = jnp.concatenate([mode1, mode2], axis=0)
        collections.append(SampleCollection(
            samples=samples, label=1, id=f"site_{i}"
        ))
    
    # Background: unimodal at origin (mean ≈ 0, same as bimodal)
    for i in range(n_background):
        samples = random.normal(random.PRNGKey(1000 + i), (samples_per_location, n_features)) * 1.5
        collections.append(SampleCollection(
            samples=samples, label=0, id=f"bg_{i}"
        ))
    
    training_data = TrainingData(
        collections=collections,
        feature_names=[f"var_{i}" for i in range(n_features)]
    )
    
    return training_data, {
        'type': 'bimodal_vs_unimodal',
        'n_sites': n_sites,
        'n_background': n_background,
        'description': 'Sites bimodal, background unimodal, similar means'
    }


def evaluate_kernel(
    model: KLRfome,
    training_data: TrainingData
) -> Dict:
    """Fit model and compute training AUC."""
    start_time = time.time()
    model.fit(training_data)
    fit_time = time.time() - start_time
    
    # Compute training predictions
    K = model._similarity_matrix
    alpha = model._fit_result.alpha
    train_preds = 1 / (1 + jnp.exp(-jnp.dot(K, alpha)))
    
    labels = np.array([c.label for c in training_data.collections])
    auc = roc_auc_score(labels, np.array(train_preds))
    
    return {
        'auc': auc,
        'fit_time': fit_time,
        'converged': model._fit_result.converged,
        'n_iterations': model._fit_result.n_iterations
    }


def run_comparison(
    training_data: TrainingData,
    sigma_values: List[float] = [0.5, 1.0, 2.0]
) -> Dict:
    """Run comparison between mean embedding and Wasserstein kernels."""
    results = {
        'mean_embedding': [],
        'wasserstein': []
    }
    
    for sigma in sigma_values:
        # Mean embedding
        model_me = KLRfome(
            sigma=sigma,
            lambda_reg=0.1,
            kernel_type='mean_embedding',
            n_rff_features=256,
            seed=42
        )
        me_result = evaluate_kernel(model_me, training_data)
        me_result['sigma'] = sigma
        results['mean_embedding'].append(me_result)
        
        # Wasserstein
        model_ws = KLRfome(
            sigma=sigma,
            lambda_reg=0.1,
            kernel_type='wasserstein',
            n_projections=100,
            seed=42
        )
        ws_result = evaluate_kernel(model_ws, training_data)
        ws_result['sigma'] = sigma
        results['wasserstein'].append(ws_result)
    
    return results


def print_results(scenario: str, metadata: Dict, results: Dict):
    """Print comparison results."""
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario}")
    print(f"{'=' * 70}")
    print(f"Details: {metadata}")
    
    print(f"\n{'Kernel Type':<20} {'Sigma':<10} {'AUC':<10} {'Time (s)':<10} {'Converged':<10}")
    print("-" * 60)
    
    for kernel_type in ['mean_embedding', 'wasserstein']:
        for r in results[kernel_type]:
            print(f"{kernel_type:<20} {r['sigma']:<10.2f} {r['auc']:<10.4f} {r['fit_time']:<10.3f} {str(r['converged']):<10}")
    
    # Summary
    best_me = max(results['mean_embedding'], key=lambda x: x['auc'])
    best_ws = max(results['wasserstein'], key=lambda x: x['auc'])
    
    print(f"\nBest Mean Embedding: AUC={best_me['auc']:.4f} (sigma={best_me['sigma']})")
    print(f"Best Wasserstein:    AUC={best_ws['auc']:.4f} (sigma={best_ws['sigma']})")
    
    if best_ws['auc'] > best_me['auc']:
        print(f"Winner: Wasserstein (+{best_ws['auc'] - best_me['auc']:.4f} AUC)")
    else:
        print(f"Winner: Mean Embedding (+{best_me['auc'] - best_ws['auc']:.4f} AUC)")


def main():
    print("=" * 70)
    print("KLRfome Kernel Comparison Benchmark")
    print("=" * 70)
    print("\nComparing Mean Embedding vs Wasserstein kernels on different scenarios.")
    
    key = random.PRNGKey(42)
    
    # Scenario 1: Standard (mean-separated) data
    print("\n\n[1/2] Standard test data (distributions differ by mean)...")
    print("Expected: Both kernels should perform similarly.")
    data1, meta1 = generate_standard_test_data(key, separation=2.0)
    results1 = run_comparison(data1)
    print_results("Standard (Mean-Separated)", meta1, results1)
    
    # Scenario 2: Bimodal vs Unimodal (same mean, different shape)
    print("\n\n[2/2] Bimodal vs Unimodal test data (same mean, different shape)...")
    print("Expected: Wasserstein should significantly outperform Mean Embedding.")
    data2, meta2 = generate_bimodal_test_data(key)
    results2 = run_comparison(data2)
    print_results("Bimodal vs Unimodal (Shape-Separated)", meta2, results2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Use Mean Embedding when:
  - Distributions differ primarily by their mean/location
  - You need faster computation (RFF approximation)
  - You want R compatibility

Use Wasserstein when:
  - Distributions have similar means but different shapes
  - You have bimodal or multimodal distributions
  - Distributional structure is important for classification
""")
    print("=" * 70)


if __name__ == "__main__":
    main()

