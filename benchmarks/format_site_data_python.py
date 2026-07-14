"""
Python equivalent of R's format_site_data function.
This ensures Python and R use the same training data.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Dict
from klrfome.data.formats import TrainingData, SampleCollection


def format_site_data_python(
    training_data: TrainingData,
    N_sites: int = 10,
    train_test_split: float = 0.8,
    sample_fraction: float = 0.9,
    background_site_balance: int = 1,
    seed: int = 42
) -> Tuple[TrainingData, TrainingData, Dict]:
    """
    Format training data to match R's format_site_data behavior.

    This function:
    1. Reduces number of sites to N_sites
    2. Splits into train/test sets
    3. Samples background based on training sites
    4. Applies sample_fraction reduction
    5. Groups samples by site ID

    Parameters:
        training_data: Original TrainingData object
        N_sites: Number of sites to use (reduces from all available)
        train_test_split: Fraction for training (rest is test)
        sample_fraction: Fraction of samples to keep after grouping
        background_site_balance: Ratio of background to sites
        seed: Random seed

    Returns:
        Tuple of (train_data, test_data, scaling_params)
        scaling_params: dict with 'means' and 'sds' for each feature
    """
    np.random.seed(seed)

    # Separate sites and background
    site_collections = [c for c in training_data.collections if c.label == 1]
    background_collections = [c for c in training_data.collections if c.label == 0]

    # Get unique site IDs
    site_ids = list(set(c.id for c in site_collections))

    # Reduce to N_sites (randomly sample)
    if len(site_ids) > N_sites:
        selected_site_ids = np.random.choice(site_ids, size=N_sites, replace=False).tolist()
    else:
        selected_site_ids = site_ids

    # Filter to selected sites
    selected_site_collections = [c for c in site_collections if c.id in selected_site_ids]

    # Split sites into train/test
    n_train_sites = int(len(selected_site_ids) * train_test_split)
    train_site_ids = np.random.choice(selected_site_ids, size=n_train_sites, replace=False).tolist()
    test_site_ids = [sid for sid in selected_site_ids if sid not in train_site_ids]

    train_site_collections = [c for c in selected_site_collections if c.id in train_site_ids]
    test_site_collections = [c for c in selected_site_collections if c.id in test_site_ids]

    # Sample background based on training sites
    n_train_sites_count = len(train_site_collections)
    n_background_train = n_train_sites_count * background_site_balance
    n_background_test = len(test_site_collections) * background_site_balance

    # Sample background collections (with replacement if needed)
    # np.random.choice needs array-like, so convert to indices
    bg_indices = list(range(len(background_collections)))

    if len(background_collections) >= n_background_train:
        selected_indices = np.random.choice(
            bg_indices,
            size=n_background_train,
            replace=False
        )
        train_background = [background_collections[i] for i in selected_indices]
    else:
        selected_indices = np.random.choice(
            bg_indices,
            size=n_background_train,
            replace=True
        )
        train_background = [background_collections[i] for i in selected_indices]

    if len(background_collections) >= n_background_test:
        selected_indices = np.random.choice(
            bg_indices,
            size=n_background_test,
            replace=False
        )
        test_background = [background_collections[i] for i in selected_indices]
    else:
        selected_indices = np.random.choice(
            bg_indices,
            size=n_background_test,
            replace=True
        )
        test_background = [background_collections[i] for i in selected_indices]

    # Compute scaling parameters from ALL original data (before reduction)
    # Convert JAX arrays to numpy if needed
    all_samples_list = [np.array(c.samples) for c in training_data.collections]
    all_samples = np.vstack(all_samples_list)
    means = np.mean(all_samples, axis=0)
    sds = np.std(all_samples, axis=0)
    sds = np.where(sds < 1e-10, 1.0, sds)  # Avoid division by zero

    # Convert to JAX arrays for consistency
    means_jax = jnp.array(means)
    sds_jax = jnp.array(sds)

    # Scale all collections
    def scale_collection(coll: SampleCollection) -> SampleCollection:
        samples_array = jnp.array(coll.samples) if not isinstance(coll.samples, jnp.ndarray) else coll.samples
        scaled_samples = (samples_array - means_jax) / sds_jax
        return SampleCollection(
            samples=scaled_samples,
            label=coll.label,
            id=coll.id,
            metadata=coll.metadata
        )

    train_site_collections = [scale_collection(c) for c in train_site_collections]
    test_site_collections = [scale_collection(c) for c in test_site_collections]
    train_background = [scale_collection(c) for c in train_background]
    test_background = [scale_collection(c) for c in test_background]

    # Apply sample_fraction reduction
    # For each collection, randomly sample a fraction of samples
    def reduce_samples(coll: SampleCollection, fraction: float) -> SampleCollection:
        samples_array = jnp.array(coll.samples) if not isinstance(coll.samples, jnp.ndarray) else coll.samples
        n_samples = len(samples_array)
        n_keep = max(1, int(n_samples * fraction))
        if n_keep >= n_samples:
            # Keep all samples
            return coll
        indices = np.random.choice(n_samples, size=n_keep, replace=False)
        # Use JAX advanced indexing
        selected_samples = samples_array[indices]
        return SampleCollection(
            samples=selected_samples,
            label=coll.label,
            id=coll.id,
            metadata=coll.metadata
        )

    train_site_collections = [reduce_samples(c, sample_fraction) for c in train_site_collections]
    test_site_collections = [reduce_samples(c, sample_fraction) for c in test_site_collections]
    train_background = [reduce_samples(c, sample_fraction) for c in train_background]
    test_background = [reduce_samples(c, sample_fraction) for c in test_background]

    # R behavior (line 278-281): Training background is split into N_back_bags collections
    # Each background row is randomly assigned to one of N_back_bags groups
    N_back_bags = N_sites * background_site_balance

    # Collect all background samples into a single list
    all_train_bg_samples = []
    for coll in train_background:
        samples_array = jnp.array(coll.samples) if not isinstance(coll.samples, jnp.ndarray) else coll.samples
        for sample in samples_array:
            all_train_bg_samples.append(sample)

    # Randomly assign each sample to one of N_back_bags groups (matching R line 280)
    if len(all_train_bg_samples) > 0:
        # Create random assignment vector (one per sample)
        assignments = np.random.choice(N_back_bags, size=len(all_train_bg_samples), replace=True)

        # Group samples by assignment
        grouped_bg = [[] for _ in range(N_back_bags)]
        for sample, assign in zip(all_train_bg_samples, assignments):
            grouped_bg[assign].append(sample)

        # Create collections for each group (only non-empty groups)
        train_background_collections = []
        for i, group_samples in enumerate(grouped_bg):
            if len(group_samples) > 0:
                samples_matrix = jnp.array(group_samples)
                bg_coll = SampleCollection(
                    samples=samples_matrix,
                    label=0,
                    id=f"background{i+1}",  # Match R naming: paste0("background", 1:N_back_bags)
                    metadata=None
                )
                train_background_collections.append(bg_coll)
    else:
        train_background_collections = []

    # R behavior: Training sites grouped by site ID, background grouped into N_back_bags
    # Test: Split into individual samples
    train_collections = train_site_collections + train_background_collections

    # Test: Split each collection into individual sample collections (matching R line 285-288)
    # R does: split tbl_test_data (which includes both sites and background) by individual row ID
    # This creates one collection per sample row
    test_collections = []

    # Combine test sites and background (like R's rbind)
    all_test_collections = test_site_collections + test_background

    for coll in all_test_collections:
        # Get samples as array
        samples_array = jnp.array(coll.samples) if not isinstance(coll.samples, jnp.ndarray) else coll.samples
        n_samples = len(samples_array)

        # Create one collection per sample in this collection
        for i in range(n_samples):
            sample_row = samples_array[i:i+1]  # Keep as 2D array (1, n_features)
            sample_coll = SampleCollection(
                samples=sample_row,
                label=coll.label,
                id=f"{coll.id}_{i}",  # Unique ID per sample (matching R's paste0(SITENO, "_", seq_len(n())))
                metadata=coll.metadata
            )
            test_collections.append(sample_coll)

    # Shuffle
    np.random.shuffle(train_collections)
    np.random.shuffle(test_collections)

    train_data = TrainingData(
        collections=train_collections,
        feature_names=training_data.feature_names,
        crs=training_data.crs
    )

    test_data = TrainingData(
        collections=test_collections,
        feature_names=training_data.feature_names,
        crs=training_data.crs
    )

    scaling_params = {
        'means': means.tolist(),
        'sds': sds.tolist()
    }

    return train_data, test_data, scaling_params
