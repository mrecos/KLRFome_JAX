# Plan: Fix analyze_real_sites.ipynb for Real Data Kernel Comparison

## Executive Summary

Transform `02_real_data.ipynb` into a production-ready kernel comparison notebook for real archaeological data that mirrors the proven structure of `02_kernel_comparison.ipynb` but handles the unique challenges of real-world data.

## Current State Analysis

### Data Structure (r91_all_riverine_section_6_regression_data_SITENO.csv)
- **Total samples**: 565,485 rows
- **Unique sites**: 153 (152 archaeological sites + 1 background "site")
- **Features**: 23 environmental variables
- **Critical Issues**:
  - ❌ **NON-uniform sample sizes**: 7 to 500,000 samples per site
  - ❌ **Massive background blob**: One "site" with 500,000 samples
  - ❌ **JIT incompatible**: Variable sizes prevent JIT optimization
  - ✓ Already in distribution format (grouped by SITENO)

### Current Notebook Issues (`02_real_data.ipynb`)
1. ❌ Uses simulated data instead of real CSV
2. ❌ No train/test split
3. ❌ No kernel comparison (only one model)
4. ❌ No uniform sample size enforcement
5. ❌ No scalability controls (can't start small)
6. ❌ No metrics evaluation (AUC, ROC, etc.)
7. ❌ No configuration section
8. ❌ No JIT verification

## Root Causes of Problems

### Problem 1: Background Data Structure
**Current**: One massive background distribution (500,000 samples)
**Issue**: Cannot use with JIT-compiled Wasserstein (requires uniform sizes)
**Impact**: Wasserstein predictions would take days instead of seconds

### Problem 2: Non-Uniform Site Sample Sizes
**Current**: Sites have 7-5000 samples each
**Issue**: Breaks JIT optimization requirement
**Impact**: Forces slow Python loops instead of vectorized operations

### Problem 3: No Scalability
**Current**: Must process all 153 sites at once
**Issue**: Can't test with small subset first
**Impact**: Long iteration times, hard to debug

## Proposed Solution Architecture

### Phase 1: Data Loading and Restructuring (Cells 1-5)

```python
# Cell 1: Configuration (COMPREHENSIVE)
################################################################################
# CONFIGURATION PARAMETERS - EASY SCALING
################################################################################

# Data source
DATA_PATH = '../site_data/r91_all_riverine_section_6_regression_data_SITENO.csv'
COORD_COLUMNS = ['x', 'y']
SITE_ID_COLUMN = 'SITENO'
LABEL_COLUMN = 'presence'

# Site selection (START SMALL, SCALE UP)
USE_ALL_SITES = False               # Set True for full analysis
N_SITES_SUBSET = 20                 # Number of sites if USE_ALL_SITES=False
SITE_SELECTION_SEED = 42            # Reproducible site selection

# Sampling (CRITICAL FOR JIT)
SAMPLES_PER_COLLECTION = 25         # EXACT - all collections must match
BACKGROUND_SAMPLES_PER_DIST = 25    # EXACT - same as sites
N_BACKGROUND_DISTRIBUTIONS = 50     # Split 500k background into 50 dists

# Train/test split
TEST_SIZE = 0.3                     # Fraction of sites for testing
SPLIT_RANDOM_STATE = 42

# Data preparation
STANDARDIZE_FEATURES = True         # Z-score normalization
REMOVE_CORRELATED = False           # Remove highly correlated features
CORRELATION_THRESHOLD = 0.95        # If REMOVE_CORRELATED=True

# Model parameters
SIGMA = 0.5
LAMBDA_REG = 0.1
WINDOW_SIZE = 5  # Not used for CSV data (no rasters)

# Kernels to compare
KERNEL_CONFIGS = {
    'Mean Embedding (RFF)': {...},
    'Wasserstein (p=2)': {...},
    'Wasserstein (p=1)': {...}
}

# Skip slow models for initial testing
SKIP_SLOW_MODELS = ['Mean Embedding (Exact)']  # Add when testing with small data
```

```python
# Cell 2: Load and Inspect Data
def load_and_inspect_csv_data(filepath, verbose=True):
    """Load CSV and analyze structure."""
    df = pd.read_csv(filepath)

    if verbose:
        print(f"Loaded {len(df):,} samples from {len(df[SITE_ID_COLUMN].unique())} sites")
        print(f"  Sites (presence=1): {df[df[LABEL_COLUMN]==1][SITE_ID_COLUMN].nunique()}")
        print(f"  Background (presence=0): {df[df[LABEL_COLUMN]==0][SITE_ID_COLUMN].nunique()}")

        # Check sample size distribution
        samples_per_site = df.groupby(SITE_ID_COLUMN).size()
        print(f"\nSamples per site: min={samples_per_site.min()}, "
              f"max={samples_per_site.max()}, "
              f"median={samples_per_site.median():.0f}")

        # Identify massive background
        max_site = samples_per_site.idxmax()
        if samples_per_site[max_site] > 10000:
            print(f"\n⚠️  Found massive background: '{max_site}' "
                  f"with {samples_per_site[max_site]:,} samples")

    return df

df_full = load_and_inspect_csv_data(DATA_PATH)
```

```python
# Cell 3: Subset Sites (for scalability)
def select_sites_subset(df, n_sites=20, seed=42):
    """Select random subset of archaeological sites."""
    # Get site IDs (exclude massive background)
    site_ids = df[df[LABEL_COLUMN] == 1][SITE_ID_COLUMN].unique()

    if n_sites >= len(site_ids):
        print(f"Using all {len(site_ids)} archaeological sites")
        selected_sites = site_ids
    else:
        np.random.seed(seed)
        selected_sites = np.random.choice(site_ids, size=n_sites, replace=False)
        print(f"Selected {n_sites} of {len(site_ids)} sites")

    # Filter to selected sites
    df_sites = df[df[SITE_ID_COLUMN].isin(selected_sites)]

    return df_sites, selected_sites

if USE_ALL_SITES:
    df_sites, selected_sites = df_full[df_full[LABEL_COLUMN]==1].copy(), df_full[df_full[LABEL_COLUMN]==1][SITE_ID_COLUMN].unique()
else:
    df_sites, selected_sites = select_sites_subset(df_full, N_SITES_SUBSET, SITE_SELECTION_SEED)

print(f"Working with {len(selected_sites)} archaeological sites")
print(f"Total site samples: {len(df_sites):,}")
```

```python
# Cell 4: Resample to Uniform Size (CRITICAL FOR JIT)
def resample_site_to_uniform(df, site_id, n_samples=25, seed=None):
    """Resample one site to exactly n_samples."""
    site_data = df[df[SITE_ID_COLUMN] == site_id]

    if len(site_data) < n_samples:
        # Oversample (sample with replacement)
        return site_data.sample(n=n_samples, replace=True, random_state=seed)
    else:
        # Downsample (sample without replacement)
        return site_data.sample(n=n_samples, replace=False, random_state=seed)

def create_uniform_site_collections(df_sites, selected_sites, samples_per_site=25):
    """Create SampleCollection for each site with EXACT sample size."""
    collections = []
    env_vars = [col for col in df.columns if col not in
                [SITE_ID_COLUMN, LABEL_COLUMN] + COORD_COLUMNS + ['', 'Unnamed: 0']]

    for idx, site_id in enumerate(selected_sites):
        # Resample to exact size
        site_resampled = resample_site_to_uniform(
            df_sites, site_id,
            n_samples=samples_per_site,
            seed=42 + idx
        )

        # Extract features
        samples = site_resampled[env_vars].values
        assert samples.shape[0] == samples_per_site

        # Get centroid location
        centroid_x = site_resampled[COORD_COLUMNS[0]].mean()
        centroid_y = site_resampled[COORD_COLUMNS[1]].mean()

        collections.append(
            SampleCollection(
                samples=jnp.array(samples),
                label=1,
                id=str(site_id),
                metadata={'location': (centroid_x, centroid_y), 'original_n': len(df_sites[df_sites[SITE_ID_COLUMN]==site_id])}
            )
        )

    print(f"✓ Created {len(collections)} site collections")
    print(f"  All with exactly {samples_per_site} samples (JIT-ready)")

    return collections, env_vars

site_collections, feature_names = create_uniform_site_collections(
    df_sites, selected_sites, SAMPLES_PER_COLLECTION
)
```

```python
# Cell 5: Create Background Distributions (from 500k blob)
def create_background_distributions(df_full, n_distributions=50, samples_per_dist=25):
    """Split massive background into multiple uniform distributions."""
    # Get background data (presence=0)
    bg_data = df_full[df_full[LABEL_COLUMN] == 0]
    print(f"Background data: {len(bg_data):,} samples")

    # Randomly shuffle
    bg_shuffled = bg_data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    collections = []
    env_vars = [col for col in df.columns if col not in
                [SITE_ID_COLUMN, LABEL_COLUMN] + COORD_COLUMNS + ['', 'Unnamed: 0']]

    for i in range(n_distributions):
        # Take next chunk of exactly samples_per_dist
        start_idx = i * samples_per_dist
        end_idx = start_idx + samples_per_dist

        if end_idx > len(bg_shuffled):
            # Wrap around if needed
            chunk = pd.concat([
                bg_shuffled.iloc[start_idx:],
                bg_shuffled.iloc[:end_idx - len(bg_shuffled)]
            ])
        else:
            chunk = bg_shuffled.iloc[start_idx:end_idx]

        assert len(chunk) == samples_per_dist

        # Extract features
        samples = chunk[env_vars].values
        centroid_x = chunk[COORD_COLUMNS[0]].mean()
        centroid_y = chunk[COORD_COLUMNS[1]].mean()

        collections.append(
            SampleCollection(
                samples=jnp.array(samples),
                label=0,
                id=f"background_{i}",
                metadata={'location': (centroid_x, centroid_y)}
            )
        )

    print(f"✓ Created {len(collections)} background distributions")
    print(f"  All with exactly {samples_per_dist} samples (JIT-ready)")

    return collections

background_collections = create_background_distributions(
    df_full, N_BACKGROUND_DISTRIBUTIONS, BACKGROUND_SAMPLES_PER_DIST
)
```

### Phase 2: Train/Test Split (Cell 6)

```python
# Cell 6: Train/Test Split (SITE-LEVEL, NO LEAKAGE)
def split_sites_train_test(site_collections, test_size=0.3, random_state=42):
    """Split sites into train/test (site-level split)."""
    from sklearn.model_selection import train_test_split

    # Split site collections
    train_sites, test_sites = train_test_split(
        site_collections,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Site split:")
    print(f"  Training: {len(train_sites)} sites")
    print(f"  Test: {len(test_sites)} sites")

    return train_sites, test_sites

def create_train_test_data(train_sites, test_sites, background_collections,
                          feature_names, test_size=0.3, random_state=42):
    """Create separate train and test TrainingData objects."""
    from sklearn.model_selection import train_test_split

    # Split background for train/test
    train_bg, test_bg = train_test_split(
        background_collections,
        test_size=test_size,
        random_state=random_state
    )

    # Combine sites + background for each set
    train_collections = train_sites + train_bg
    test_collections = test_sites + test_bg

    # Create TrainingData objects
    train_data = TrainingData(
        collections=train_collections,
        feature_names=feature_names,
        crs="EPSG:32617"  # UTM Zone 17N (adjust based on your data)
    )

    test_data = TrainingData(
        collections=test_collections,
        feature_names=feature_names,
        crs="EPSG:32617"
    )

    # VERIFICATION
    all_sizes = [len(c.samples) for c in train_collections + test_collections]
    assert len(set(all_sizes)) == 1, f"Non-uniform sizes: {set(all_sizes)}"

    print(f"\nFinal datasets:")
    print(f"  Training: {len(train_collections)} collections "
          f"({len(train_sites)} sites + {len(train_bg)} background)")
    print(f"  Test: {len(test_collections)} collections "
          f"({len(test_sites)} sites + {len(test_bg)} background)")
    print(f"\n✓ UNIFORM: All collections have {all_sizes[0]} samples")
    print(f"  JIT-optimized Wasserstein ENABLED")

    return train_data, test_data

train_sites, test_sites = split_sites_train_test(site_collections, TEST_SIZE, SPLIT_RANDOM_STATE)
train_data_raw, test_data_raw = create_train_test_data(
    train_sites, test_sites, background_collections,
    feature_names, TEST_SIZE, SPLIT_RANDOM_STATE
)
```

### Phase 3: Feature Standardization (Cell 7)

```python
# Cell 7: Z-Score Normalization (TRAIN STATISTICS ONLY)
def standardize_features(train_data, test_data):
    """Apply z-score normalization using ONLY training statistics."""
    # Compute from training data
    train_samples = np.vstack([c.samples for c in train_data.collections])
    means = np.mean(train_samples, axis=0)
    sds = np.std(train_samples, axis=0)
    sds = np.where(sds < 1e-10, 1.0, sds)

    def scale_collection(coll):
        scaled = (coll.samples - means) / sds
        return SampleCollection(
            samples=scaled,
            label=coll.label,
            id=coll.id,
            metadata=coll.metadata
        )

    train_scaled = TrainingData(
        collections=[scale_collection(c) for c in train_data.collections],
        feature_names=train_data.feature_names,
        crs=train_data.crs
    )

    test_scaled = TrainingData(
        collections=[scale_collection(c) for c in test_data.collections],
        feature_names=test_data.feature_names,
        crs=test_data.crs
    )

    return train_scaled, test_scaled, means, sds

if STANDARDIZE_FEATURES:
    train_data_scaled, test_data_scaled, feature_means, feature_sds = standardize_features(
        train_data_raw, test_data_raw
    )
else:
    train_data_scaled, test_data_scaled = train_data_raw, test_data_raw
```

### Phase 4: Kernel Evaluation (Cells 8-11)
**Reuse EXACT code from 02_kernel_comparison.ipynb:**
- Cell 8: Evaluation functions (compute_metrics, evaluate_kernel, evaluate_logistic_regression)
- Cell 9: Run comparison
- Cell 10: Metrics table
- Cell 11: ROC curves

### Phase 5: Visualizations (Cells 12-15)
**Adapt from 02_kernel_comparison.ipynb:**
- Cell 12: Geographic scatter plot (sites in feature space, not raster)
- Cell 13: PCA feature space analysis
- Cell 14: Feature distributions comparison
- Cell 15: Test prediction diagnostics

## Critical Differences from Synthetic Notebook

| Aspect | Synthetic (02_kernel_comparison) | Real Data (New) |
|--------|----------------------------------|-----------------|
| Data source | Generated rasters | CSV with pre-extracted samples |
| Background | Multiple distributions sampled from raster | One massive blob → split into distributions |
| Raster predictions | Yes (full spatial maps) | No (CSV has no rasters) |
| Sample extraction | Extract from raster windows | Resample from existing samples |
| Spatial context | Pixel coordinates + rasters | X,Y coordinates only |
| Visualization | Prediction maps with sites | Scatter plots in geographic space |

## Implementation Checklist

### Must-Have Features
- [ ] Comprehensive configuration section (Cell 1)
- [ ] Scalability controls (start with 20 sites, scale to 152)
- [ ] Uniform sample size enforcement (SAMPLES_PER_COLLECTION=25)
- [ ] Background blob splitting (500k → N distributions)
- [ ] Proper train/test split (site-level, before processing)
- [ ] Z-score normalization (train statistics only)
- [ ] JIT verification (confirm _uniform_samples=True)
- [ ] Multiple kernel comparison (RFF, Wass p=1, Wass p=2)
- [ ] Distribution-level metrics (AUC, sensitivity, etc.)
- [ ] Feature correlation analysis (optional)

### Nice-to-Have Features
- [ ] Feature selection based on correlation
- [ ] Cross-validation for hyperparameter tuning
- [ ] Ensemble predictions
- [ ] Partial dependence plots
- [ ] SHAP values for interpretability

## Testing Strategy

### Phase 1: Small-Scale Test (N_SITES_SUBSET=10)
1. Load 10 sites
2. Create 10 background distributions
3. Run all kernels
4. Verify JIT optimization working
5. Check metrics make sense
6. **Expected time**: ~5 minutes

### Phase 2: Medium-Scale Test (N_SITES_SUBSET=50)
1. Load 50 sites
2. Create 50 background distributions
3. Run all kernels
4. Compare to Phase 1 results
5. **Expected time**: ~15 minutes

### Phase 3: Full-Scale Run (USE_ALL_SITES=True)
1. Load all 152 sites
2. Create 100 background distributions
3. Run all kernels
4. Final evaluation
5. **Expected time**: ~30 minutes

## Expected Outcomes

### Performance Benchmarks
- **Data loading**: <5 seconds
- **Uniform resampling**: <10 seconds
- **Background splitting**: <30 seconds
- **Train/test split**: <1 second
- **Feature scaling**: <1 second
- **Kernel evaluation** (per kernel):
  - Mean Embedding (RFF): ~5 seconds
  - Wasserstein (p=2): ~30 seconds (JIT-optimized)
  - Wasserstein (p=1): ~30 seconds (JIT-optimized)
- **Total runtime** (152 sites): ~30 minutes

### Validation Checks
✓ All collections have exactly 25 samples
✓ _uniform_samples = True for all models
✓ Test AUC in range [0.6, 0.95]
✓ Train-test gap < 0.20
✓ No collection ID overlap between train/test
✓ Background distributions spatially distributed

## File Naming

**Rename**: `02_real_data.ipynb` → `03_real_data_kernel_comparison.ipynb`

**Keep sequential numbering:**
- 01: Quickstart (synthetic)
- 02: Kernel comparison (synthetic)
- 03: Real data kernel comparison (this notebook)
- 04: Hyperparameter tuning (if exists)

## Success Criteria

1. ✓ Notebook runs end-to-end without errors
2. ✓ JIT optimization confirmed (Wasserstein < 60 seconds)
3. ✓ Proper train/test split (no leakage)
4. ✓ Scalable (works with 10, 50, or 152 sites)
5. ✓ Comprehensive metrics (AUC, ROC, confusion matrix)
6. ✓ Clear configuration section
7. ✓ Matches structure of 02_kernel_comparison.ipynb
8. ✓ Well-documented with markdown explanations
