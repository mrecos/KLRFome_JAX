#!/bin/bash
# Benchmark script to compare Python/JAX and R implementations
#
# This script:
# 1. Generates benchmark data (if needed)
# 2. Runs Python workflow and captures timing
# 3. Runs R workflow and captures timing
# 4. Compares results and generates report
#
# Usage:
#   bash benchmarks/run_benchmark.sh [--regenerate-data] [--data-dir DATA_DIR]

set -e  # Exit on error

# Default values
REGENERATE_DATA=false
DATA_DIR="benchmark_data"
PYTHON_RESULTS="benchmark_python_results.json"
R_RESULTS="benchmark_r_results.json"
REPORT_FILE="benchmark_report.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --regenerate-data)
            REGENERATE_DATA=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--regenerate-data] [--data-dir DATA_DIR]"
            exit 1
            ;;
    esac
done

echo "=================================================================================="
echo "KLRfome Performance Benchmark: Python/JAX vs R"
echo "=================================================================================="
echo ""

# Step 1: Generate data if needed or requested
if [ "$REGENERATE_DATA" = true ] || [ ! -d "$DATA_DIR" ]; then
    echo "[Step 1/4] Generating benchmark data..."
    python benchmarks/generate_benchmark_data.py --output-dir "$DATA_DIR" --seed 42
    echo ""
else
    echo "[Step 1/4] Using existing benchmark data in: $DATA_DIR"
    echo ""
fi

# Step 2: Run Python workflow
echo "[Step 2/4] Running Python/JAX workflow..."
if python benchmarks/test_python_workflow.py \
    --data-dir "$DATA_DIR" \
    --sigma 0.5 \
    --lambda 0.1 \
    --output "$PYTHON_RESULTS"; then
    echo "  ✓ Python workflow completed successfully"
else
    echo "  ✗ Python workflow failed"
    exit 1
fi
echo ""

# Step 3: Run R workflow
echo "[Step 3/4] Running R workflow..."
if Rscript benchmarks/test_r_workflow.R \
    --data-dir "$DATA_DIR" \
    --sigma 0.5 \
    --lambda 0.1 \
    --output "$R_RESULTS"; then
    echo "  ✓ R workflow completed successfully"
else
    echo "  ✗ R workflow failed"
    exit 1
fi
echo ""

# Step 4: Compare results and generate report
echo "[Step 4/4] Generating comparison report..."
python << EOF
import json
import sys

try:
    with open('$PYTHON_RESULTS', 'r') as f:
        py_results = json.load(f)

    with open('$R_RESULTS', 'r') as f:
        r_results = json.load(f)

    report = []
    report.append("=" * 80)
    report.append("BENCHMARK COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")

    # Timing comparison
    report.append("TIMING COMPARISON (seconds)")
    report.append("-" * 80)

    # Helper function to extract scalar from potentially nested value
    def get_scalar(value, default=0):
        if isinstance(value, list):
            return value[0] if len(value) > 0 else default
        return value if value is not None else default

    # Map R timing to Python timing structure
    r_timing_map = {
        'load_time': get_scalar(r_results.get('load_time', 0)),
        'init_time': 0,  # R doesn't have separate init
        'prep_time': (get_scalar(r_results.get('extract_time', 0)) + get_scalar(r_results.get('format_time', 0))),
        'fit_time': (get_scalar(r_results.get('build_k_time', 0)) + get_scalar(r_results.get('fit_time', 0))),
        'predict_time': get_scalar(r_results.get('predict_time', 0))
    }

    timing_labels = {
        'load_time': 'Load/Extract Data',
        'init_time': 'Initialize Model',
        'prep_time': 'Prepare/Format Data',
        'fit_time': 'Build Kernel + Fit',
        'predict_time': 'Predict'
    }

    for key in ['load_time', 'init_time', 'prep_time', 'fit_time', 'predict_time']:
        py_time = py_results.get(key, 0)
        r_time = r_timing_map.get(key, 0)
        if py_time > 0 or r_time > 0:
            speedup = r_time / py_time if py_time > 0 else float('inf')
            report.append(f"  {timing_labels[key]:25s}  Python: {py_time:8.3f}s  R: {r_time:8.3f}s  Speedup: {speedup:5.2f}x")

    report.append("")
    report.append("-" * 80)
    py_total = py_results.get('total_time', 0)
    r_total = get_scalar(r_results.get('total_time', 0))
    total_speedup = r_total / py_total if py_total > 0 else float('inf')
    report.append(f"  {'TOTAL':25s}  Python: {py_total:8.3f}s  R: {r_total:8.3f}s  Speedup: {total_speedup:5.2f}x")
    report.append("")

    # Model information
    report.append("MODEL INFORMATION")
    report.append("-" * 80)
    report.append(f"  Python converged: {py_results.get('converged', 'N/A')}")
    report.append(f"  Python iterations: {py_results.get('n_iterations', 'N/A')}")
    r_converged_raw = r_results.get('converged', None)
    r_iterations_raw = r_results.get('iterations', None)
    r_converged = get_scalar(r_converged_raw, default=None) if r_converged_raw is not None else None
    r_iterations = get_scalar(r_iterations_raw, default=None) if r_iterations_raw is not None else None
    if r_converged is not None:
        report.append(f"  R converged: {r_converged}")
    if r_iterations is not None:
        report.append(f"  R iterations: {r_iterations}")
    report.append("")

    # Prediction information
    report.append("PREDICTION INFORMATION")
    report.append("-" * 80)
    py_shape = py_results.get('predictions_shape', [])
    r_shape_raw = r_results.get('predictions_shape', [])
    # Handle shape - should be a list of 2 values
    if isinstance(r_shape_raw, list):
        if len(r_shape_raw) == 2:
            r_shape = r_shape_raw
        elif len(r_shape_raw) == 1 and isinstance(r_shape_raw[0], list):
            r_shape = r_shape_raw[0]  # Unwrap nested list
        else:
            r_shape = r_shape_raw
    else:
        r_shape = [r_shape_raw] if r_shape_raw else []
    report.append(f"  Python shape: {py_shape}")
    report.append(f"  R shape: {r_shape}")

    py_range = py_results.get('predictions_range', [])
    r_range_raw = r_results.get('predictions_range', [])
    # Handle range - should be a list of 2 values
    if isinstance(r_range_raw, list):
        if len(r_range_raw) >= 2:
            r_range = [float(r_range_raw[0]), float(r_range_raw[1])]
        elif len(r_range_raw) == 1 and isinstance(r_range_raw[0], list):
            r_range = [float(r_range_raw[0][0]), float(r_range_raw[0][1])] if len(r_range_raw[0]) >= 2 else [0.0, 0.0]
        else:
            r_range = [0.0, 0.0]
    else:
        r_range = [0.0, 0.0]
    if len(py_range) >= 2:
        report.append(f"  Python range: [{py_range[0]:.3f}, {py_range[1]:.3f}]")
    else:
        report.append(f"  Python range: N/A")
    report.append(f"  R range: [{r_range[0]:.3f}, {r_range[1]:.3f}]")

    py_mean = py_results.get('predictions_mean', 0)
    r_mean = get_scalar(r_results.get('predictions_mean', 0))
    report.append(f"  Python mean: {py_mean:.3f}")
    report.append(f"  R mean: {r_mean:.3f}")
    report.append("")

    report.append("=" * 80)

    # Write report
    with open('$REPORT_FILE', 'w') as f:
        f.write('\n'.join(report))

    # Also print to stdout
    print('\n'.join(report))

except Exception as e:
    print(f"Error generating report: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "✓ Benchmark complete!"
echo "  Results saved to:"
echo "    - Python: $PYTHON_RESULTS"
echo "    - R: $R_RESULTS"
echo "    - Report: $REPORT_FILE"
echo ""
