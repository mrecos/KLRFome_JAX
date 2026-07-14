#!/usr/bin/env python3
"""
Compare Python and R when using EXACTLY the same data (R's exported collections).
This isolates kernel computation and IRLS algorithm differences.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    output_dir = Path("diagnostic_output")

    print("=" * 80)
    print("EXACT COMPARISON: Python vs R (Same Collections)")
    print("=" * 80)

    # 1. Compare kernel matrices
    print("\n[1/3] Kernel Matrix")
    print("-" * 60)

    py_K = np.load(output_dir / "python_rdata_kernel.npy")
    r_K = pd.read_csv(output_dir / "r_rdata_kernel.csv").values

    print(f"  Shape: Python={py_K.shape}, R={r_K.shape}")

    if py_K.shape == r_K.shape:
        print(f"  Python: mean={py_K.mean():.6f}, diag_mean={np.diag(py_K).mean():.6f}")
        print(f"  R:      mean={r_K.mean():.6f}, diag_mean={np.diag(r_K).mean():.6f}")

        diff = np.abs(py_K - r_K)
        print(f"  Max difference: {diff.max():.6f}")
        print(f"  Mean difference: {diff.mean():.6f}")

        # Diagonal comparison
        py_diag = np.diag(py_K)
        r_diag = np.diag(r_K)
        diag_diff = np.abs(py_diag - r_diag)

        print(f"\n  Diagonal values:")
        print(f"  {'Idx':<4} {'Python':<10} {'R':<10} {'Diff':<10}")
        for i in range(len(py_diag)):
            marker = " ***" if diag_diff[i] > 0.001 else ""
            print(f"  {i:<4} {py_diag[i]:<10.4f} {r_diag[i]:<10.4f} {diag_diff[i]:<10.6f}{marker}")

        if diff.max() < 0.001:
            print("\n  ✓ Kernels MATCH!")
        else:
            print(f"\n  ❌ Kernels differ (max diff={diff.max():.6f})")
    else:
        print("  ❌ SHAPE MISMATCH")
        return

    # 2. Compare alpha values
    print("\n[2/3] Alpha Values")
    print("-" * 60)

    py_alpha = np.load(output_dir / "python_rdata_alpha.npy")
    r_alpha = pd.read_csv(output_dir / "r_rdata_alpha.csv").values.flatten()

    print(f"  Python: mean={py_alpha.mean():.6f}, std={py_alpha.std():.6f}")
    print(f"  R:      mean={r_alpha.mean():.6f}, std={r_alpha.std():.6f}")

    alpha_diff = np.abs(py_alpha - r_alpha)
    print(f"  Max difference: {alpha_diff.max():.6f}")
    print(f"  Mean difference: {alpha_diff.mean():.6f}")

    if alpha_diff.max() < 0.01:
        print("  ✓ Alphas MATCH!")
    else:
        print(f"  ❌ Alphas differ significantly")
        print(f"\n  Per-element comparison:")
        print(f"  {'Idx':<4} {'Python':<12} {'R':<12} {'Diff':<12}")
        for i in range(len(py_alpha)):
            marker = " ***" if alpha_diff[i] > 0.01 else ""
            print(f"  {i:<4} {py_alpha[i]:<12.6f} {r_alpha[i]:<12.6f} {alpha_diff[i]:<12.6f}{marker}")

    # 3. Compare predictions
    print("\n[3/3] Predictions")
    print("-" * 60)

    py_pred = np.load(output_dir / "python_rdata_predictions.npy")
    with open(output_dir / "r_rdata_predictions.json") as f:
        r_pred = np.array(json.load(f))

    print(f"  Python: mean={py_pred.mean():.6f}, range=[{py_pred.min():.4f}, {py_pred.max():.4f}]")
    print(f"  R:      mean={r_pred.mean():.6f}, range=[{r_pred.min():.4f}, {r_pred.max():.4f}]")

    pred_diff = np.abs(py_pred - r_pred)
    print(f"  Max difference: {pred_diff.max():.6f}")
    print(f"  Mean difference: {pred_diff.mean():.6f}")

    if pred_diff.max() < 0.01:
        print("  ✓ Predictions MATCH!")
    else:
        print(f"  ❌ Predictions differ")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    kernel_match = diff.max() < 0.001
    alpha_match = alpha_diff.max() < 0.01
    pred_match = pred_diff.max() < 0.01

    print(f"  Kernel:      {'✓ MATCH' if kernel_match else '❌ DIFFER'}")
    print(f"  Alpha:       {'✓ MATCH' if alpha_match else '❌ DIFFER'}")
    print(f"  Predictions: {'✓ MATCH' if pred_match else '❌ DIFFER'}")

    if not kernel_match:
        print("\n  → Problem is in KERNEL computation")
    elif not alpha_match:
        print("\n  → Problem is in IRLS/KLR algorithm")
    elif not pred_match:
        print("\n  → Problem is in sigmoid/prediction function")
    else:
        print("\n  → All components match! ✓")


if __name__ == "__main__":
    main()
