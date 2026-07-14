#!/usr/bin/env python3
"""
Compare Python and R outputs when using SHARED data.
This isolates algorithm differences from data extraction differences.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    output_dir = Path("diagnostic_output")

    print("=" * 80)
    print("COMPARING PYTHON vs R (SHARED DATA)")
    print("=" * 80)

    # 1. Compare training info
    print("\n[1/5] Training Info")
    print("-" * 40)

    with open(output_dir / "python_shared_training_info.json") as f:
        py_info = json.load(f)
    with open(output_dir / "r_shared_training_info.json") as f:
        r_info = json.load(f)

    for key in py_info:
        py_val = py_info[key]
        r_val = r_info.get(key, "N/A")
        match = "✓" if py_val == r_val else "❌"
        print(f"  {key}: Python={py_val}, R={r_val} {match}")

    # 2. Compare sample data
    print("\n[2/5] Sample Data (First 3 collections)")
    print("-" * 40)

    with open(output_dir / "python_shared_sample_data.json") as f:
        py_samples = json.load(f)
    with open(output_dir / "r_shared_sample_data.json") as f:
        r_samples = json.load(f)

    for i in range(min(3, len(py_samples), len(r_samples))):
        py = py_samples[i]
        r = r_samples[i]
        print(f"  Collection {i}:")
        print(f"    n_samples: Python={py['n_samples']}, R={r['n_samples']}")

        py_mean = np.array(py['mean'])
        r_mean = np.array(r['mean'])
        diff = np.abs(py_mean - r_mean).max()
        match = "✓" if diff < 0.01 else f"❌ (max diff={diff:.4f})"
        print(f"    mean: Python={np.round(py_mean, 4)}, R={np.round(r_mean, 4)} {match}")

    # 3. Compare kernel matrices
    print("\n[3/5] Kernel Matrix")
    print("-" * 40)

    py_K = np.load(output_dir / "python_shared_kernel.npy")
    r_K = pd.read_csv(output_dir / "r_shared_kernel.csv").values

    print(f"  Shape: Python={py_K.shape}, R={r_K.shape}")

    if py_K.shape == r_K.shape:
        print(f"  Python: mean={py_K.mean():.6f}, diag_mean={np.diag(py_K).mean():.6f}")
        print(f"  R:      mean={r_K.mean():.6f}, diag_mean={np.diag(r_K).mean():.6f}")

        diff = np.abs(py_K - r_K)
        print(f"  Max difference: {diff.max():.6f}")
        print(f"  Mean difference: {diff.mean():.6f}")

        # Compare diagonals
        py_diag = np.diag(py_K)
        r_diag = np.diag(r_K)
        diag_diff = np.abs(py_diag - r_diag)
        print(f"  Diagonal max diff: {diag_diff.max():.6f} at index {np.argmax(diag_diff)}")

        if diff.max() < 0.01:
            print("  ✓ Kernels match!")
        else:
            print("  ❌ Kernels differ significantly")
    else:
        print("  ❌ SHAPE MISMATCH")

    # 4. Compare labels
    print("\n[4/5] Labels")
    print("-" * 40)

    py_labels = np.load(output_dir / "python_shared_labels.npy")
    with open(output_dir / "r_shared_labels.json") as f:
        r_labels = np.array(json.load(f))

    print(f"  Python: {py_labels}")
    print(f"  R:      {r_labels}")
    match = "✓" if np.array_equal(py_labels, r_labels) else "❌"
    print(f"  Match: {match}")

    # 5. Compare alpha and predictions
    print("\n[5/5] Model Outputs")
    print("-" * 40)

    py_alpha = np.load(output_dir / "python_shared_alpha.npy")
    r_alpha = pd.read_csv(output_dir / "r_shared_alpha.csv").values.flatten()

    print(f"  Alpha shapes: Python={py_alpha.shape}, R={r_alpha.shape}")
    print(f"  Python alpha: mean={py_alpha.mean():.6f}, std={py_alpha.std():.6f}")
    print(f"  R alpha:      mean={r_alpha.mean():.6f}, std={r_alpha.std():.6f}")

    if len(py_alpha) == len(r_alpha):
        alpha_diff = np.abs(py_alpha - r_alpha)
        print(f"  Alpha max diff: {alpha_diff.max():.6f}")

    py_pred = np.load(output_dir / "python_shared_predictions.npy")
    with open(output_dir / "r_shared_predictions.json") as f:
        r_pred = np.array(json.load(f))

    print(f"  Python pred: mean={py_pred.mean():.6f}, range=[{py_pred.min():.4f}, {py_pred.max():.4f}]")
    print(f"  R pred:      mean={r_pred.mean():.6f}, range=[{r_pred.min():.4f}, {r_pred.max():.4f}]")

    if len(py_pred) == len(r_pred):
        pred_diff = np.abs(py_pred - r_pred)
        print(f"  Prediction max diff: {pred_diff.max():.6f}")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
