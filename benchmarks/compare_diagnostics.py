#!/usr/bin/env python3
"""
Compare diagnostic outputs from Python and R implementations.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Try to import pandas for better CSV handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Try to import pandas for better CSV handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_rds(filepath):
    """Load R RDS file (basic attempt - may need R for full support)."""
    # For now, we'll use CSV if available
    csv_path = filepath.with_suffix('.csv')
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=',')
    return None


def compare_arrays(arr1, arr2, name, rtol=1e-5, atol=1e-8):
    """Compare two arrays and report differences."""
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    print(f"\n{'='*80}")
    print(f"Comparing: {name}")
    print(f"{'='*80}")

    if arr1.shape != arr2.shape:
        print(f"  ❌ SHAPE MISMATCH: Python {arr1.shape} vs R {arr2.shape}")
        return False

    print(f"  Shape: {arr1.shape}")
    print(f"  Python: mean={arr1.mean():.6f}, std={arr1.std():.6f}, range=[{arr1.min():.6f}, {arr1.max():.6f}]")
    print(f"  R:      mean={arr2.mean():.6f}, std={arr2.std():.6f}, range=[{arr2.min():.6f}, {arr2.max():.6f}]")

    # Check if close
    if np.allclose(arr1, arr2, rtol=rtol, atol=atol):
        print(f"  ✓ Arrays are close (rtol={rtol}, atol={atol})")
        return True
    else:
        diff = np.abs(arr1 - arr2)
        print(f"  ❌ Arrays differ:")
        print(f"    Max absolute difference: {diff.max():.6f}")
        print(f"    Mean absolute difference: {diff.mean():.6f}")
        print(f"    Relative difference: {(diff / (np.abs(arr2) + 1e-10)).max():.6f}")

        # Find locations of largest differences
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Largest diff at index {max_idx}: Python={arr1[max_idx]:.6f}, R={arr2[max_idx]:.6f}")

        return False


def compare_kernel_matrices(py_dir, r_dir):
    """Compare kernel matrices."""
    py_kernel = np.load(py_dir / "python_kernel_matrix.npy")

    # Try to load R kernel
    r_kernel_file = r_dir / "r_kernel_matrix.rds"
    r_kernel_csv = r_dir / "r_kernel_matrix.csv"

    if r_kernel_csv.exists():
        # R CSV files: R's write.csv() doesn't include row names by default
        # So the CSV should be a pure numeric matrix
        if HAS_PANDAS:
            try:
                # Try without index_col first (R's write.csv doesn't include row names)
                df = pd.read_csv(r_kernel_csv, header=None)
                # Remove any non-numeric columns
                df = df.select_dtypes(include=[np.number])
                r_kernel = df.values
                # If shape is wrong, try with index_col (in case row names were added)
                if r_kernel.shape[1] != py_kernel.shape[1]:
                    df = pd.read_csv(r_kernel_csv, index_col=0)
                    df = df.select_dtypes(include=[np.number])
                    r_kernel = df.values
            except Exception as e:
                print(f"  ⚠ Could not load R kernel matrix with pandas: {e}")
                return False
        else:
            # Fallback to numpy
            try:
                # R's write.csv() doesn't include row names, so no header to skip
                r_kernel = np.loadtxt(r_kernel_csv, delimiter=',')
            except ValueError:
                try:
                    # Try skipping first row in case there's a header
                    r_kernel = np.loadtxt(r_kernel_csv, delimiter=',', skiprows=1)
                except ValueError as e:
                    print(f"  ⚠ Could not load R kernel matrix: {e}")
                    print(f"    Install pandas for better CSV handling: pip install pandas")
                    return False
    else:
        print("  ⚠ R kernel matrix CSV not found")
        return False

    return compare_arrays(py_kernel, r_kernel, "Kernel Matrix", rtol=1e-3, atol=1e-3)


def compare_alpha_history(py_dir, r_dir):
    """Compare alpha iteration history."""
    py_alpha = np.load(py_dir / "python_alpha_history.npy")

    r_alpha_csv = r_dir / "r_alpha_history.csv"
    if r_alpha_csv.exists():
        if HAS_PANDAS:
            try:
                df = pd.read_csv(r_alpha_csv, index_col=0)  # First column might be row names
                df = df.select_dtypes(include=[np.number])
                r_alpha = df.values
            except Exception as e:
                print(f"  ⚠ Could not load R alpha history with pandas: {e}")
                return False
        else:
            try:
                r_alpha = np.loadtxt(r_alpha_csv, delimiter=',', skiprows=1)
            except ValueError:
                try:
                    r_alpha = np.loadtxt(r_alpha_csv, delimiter=',')
                except ValueError as e:
                    print(f"  ⚠ Could not load R alpha history: {e}")
                    print(f"    Install pandas for better CSV handling: pip install pandas")
                    return False
    else:
        print("  ⚠ R alpha history CSV not found")
        return False

    # Compare final alpha
    print(f"\nComparing final alpha values:")
    py_final = py_alpha[-1]
    r_final = r_alpha[-1]

    return compare_arrays(py_final, r_final, "Final Alpha", rtol=1e-2, atol=1e-3)


def compare_training_info(py_dir, r_dir):
    """Compare training data information."""
    with open(py_dir / "python_training_info.json") as f:
        py_info = json.load(f)

    with open(r_dir / "r_training_info.json") as f:
        r_info = json.load(f)

    print(f"\n{'='*80}")
    print("Training Data Info")
    print(f"{'='*80}")

    all_match = True
    for key in py_info:
        py_val = py_info[key]
        r_val = r_info.get(key, None)
        match = py_val == r_val
        status = "✓" if match else "❌"
        print(f"  {status} {key}: Python={py_val}, R={r_val}")
        if not match:
            all_match = False

    return all_match


def compare_kernel_stats(py_dir, r_dir):
    """Compare kernel statistics."""
    with open(py_dir / "python_kernel_stats.json") as f:
        py_stats = json.load(f)

    with open(r_dir / "r_kernel_stats.json") as f:
        r_stats = json.load(f)

    print(f"\n{'='*80}")
    print("Kernel Statistics")
    print(f"{'='*80}")

    for key in ['mean', 'std', 'min', 'max', 'diagonal_mean', 'off_diagonal_mean']:
        py_val = py_stats[key]
        r_val = r_stats[key]
        diff = abs(py_val - r_val)
        rel_diff = diff / (abs(r_val) + 1e-10)
        status = "✓" if rel_diff < 0.01 else "❌"
        print(f"  {status} {key}: Python={py_val:.6f}, R={r_val:.6f}, diff={diff:.6f} ({rel_diff*100:.2f}%)")

    # Check rounding
    if 'has_rounding' in r_stats:
        print(f"\n  ⚠ R kernel values are rounded to 3 decimals: {not r_stats['has_rounding']}")


def compare_iteration_details(py_dir, r_dir):
    """Compare IRLS iteration details."""
    with open(py_dir / "python_iteration_details.json") as f:
        py_iters = json.load(f)

    with open(r_dir / "r_iteration_details.json") as f:
        r_iters = json.load(f)

    print(f"\n{'='*80}")
    print("IRLS Iteration Comparison")
    print(f"{'='*80}")

    min_len = min(len(py_iters), len(r_iters))
    print(f"  Python iterations: {len(py_iters)}")
    print(f"  R iterations: {len(r_iters)}")

    # Compare first and last iterations
    if min_len > 0:
        print(f"\n  First iteration:")
        py_first = py_iters[0]
        r_first = r_iters[0]
        for key in ['alpha_mean', 'prob_mean', 'delta_max']:
            py_val = py_first[key]
            r_val = r_first[key]
            print(f"    {key}: Python={py_val:.6f}, R={r_val:.6f}")

        if min_len > 1:
            print(f"\n  Last iteration:")
            py_last = py_iters[-1]
            r_last = r_iters[-1]
            for key in ['alpha_mean', 'prob_mean', 'delta_max']:
                py_val = py_last[key]
                r_val = r_last[key]
                print(f"    {key}: Python={py_val:.6f}, R={r_val:.6f}")


def compare_predictions(py_dir, r_dir):
    """Compare prediction statistics."""
    py_pred_file = py_dir / "python_prediction_stats.json"
    r_pred_file = r_dir / "r_prediction_stats.json"

    if not py_pred_file.exists() or not r_pred_file.exists():
        print(f"\n{'='*80}")
        print("Prediction Statistics")
        print(f"{'='*80}")
        print("  ⚠ Prediction statistics not available (quick diagnostic skips raster prediction)")
        if not py_pred_file.exists():
            print(f"    Missing: {py_pred_file}")
        if not r_pred_file.exists():
            print(f"    Missing: {r_pred_file}")
        return

    with open(py_pred_file) as f:
        py_pred = json.load(f)

    with open(r_pred_file) as f:
        r_pred = json.load(f)

    print(f"\n{'='*80}")
    print("Prediction Statistics")
    print(f"{'='*80}")

    for key in ['mean', 'std', 'min', 'max']:
        py_val = py_pred.get(key, None)
        r_val = r_pred.get(key, None)
        if py_val is None or r_val is None:
            print(f"  ⚠ {key}: Missing in one or both implementations")
            continue
        diff = abs(py_val - r_val)
        rel_diff = diff / (abs(r_val) + 1e-10)
        status = "✓" if rel_diff < 0.1 else "❌"
        print(f"  {status} {key}: Python={py_val:.6f}, R={r_val:.6f}, diff={diff:.6f} ({rel_diff*100:.2f}%)")


def main():
    py_dir = Path("diagnostic_output")
    r_dir = Path("diagnostic_output")

    if not py_dir.exists():
        print(f"Error: Python diagnostic output not found at {py_dir}")
        print("Run: python benchmarks/diagnose_differences.py")
        return

    if not r_dir.exists():
        print(f"Error: R diagnostic output not found at {r_dir}")
        print("Run: Rscript benchmarks/diagnose_r_workflow.R")
        return

    print("=" * 80)
    print("DIAGNOSTIC COMPARISON: Python vs R")
    print("=" * 80)

    # Compare training info
    compare_training_info(py_dir, r_dir)

    # Compare kernel statistics
    compare_kernel_stats(py_dir, r_dir)

    # Compare kernel matrices
    compare_kernel_matrices(py_dir, r_dir)

    # Compare alpha history
    compare_alpha_history(py_dir, r_dir)

    # Compare iteration details
    compare_iteration_details(py_dir, r_dir)

    # Compare predictions (may not exist in quick diagnostic)
    try:
        compare_predictions(py_dir, r_dir)
    except Exception as e:
        print(f"\n  ⚠ Could not compare predictions: {e}")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
