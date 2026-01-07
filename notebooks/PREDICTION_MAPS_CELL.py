"""
ADD THIS AS NEW CELLS IN YOUR NOTEBOOK (after Cell 18, before ROC curves)

This generates raster predictions for each kernel and visualizes them
with training and test sites overlaid.
"""

# ============================================================================
# NEW CELL: Markdown Header
# ============================================================================
"""
### 5.2 Generate Raster Predictions for Each Model

Generate full raster predictions to visualize spatial patterns learned by each model.
"""

# ============================================================================
# NEW CELL: Generate Raster Predictions Function
# ============================================================================
print("="*80)
print("GENERATING RASTER PREDICTIONS FOR ALL MODELS")
print("="*80)
print()

def generate_raster_predictions_for_kernel(kernel_name, config, training_data,
                                           raster_scaled, sigma=0.5, window_size=5,
                                           batch_size=1000):
    """Generate raster predictions using a fitted kernel model.

    Args:
        kernel_name: Name of the kernel
        config: Kernel configuration dict
        training_data: TrainingData used to fit the model (scaled)
        raster_scaled: Scaled raster to make predictions on
        sigma: RBF kernel bandwidth
        window_size: Window size for focal predictions
        batch_size: Batch size for prediction (default 1000)

    Returns:
        predictions: 2D array of prediction probabilities
        model: Fitted KLRfome model
    """
    from klrfome.kernels.wasserstein import SlicedWassersteinDistance

    # Build model config
    model_kwargs = {
        'sigma': sigma,
        'lambda_reg': LAMBDA_REG,
        'window_size': window_size,
        'seed': 42
    }
    for key, value in config.items():
        if key not in ['description', 'color']:
            model_kwargs[key] = value

    # Create and fit model
    print(f"  Fitting {kernel_name}...")
    model = KLRfome(**model_kwargs)
    model.fit(training_data)

    # Generate predictions
    print(f"  Predicting on raster ({raster_scaled.data.shape[1]}×{raster_scaled.data.shape[2]} pixels)...")
    start_time = time.time()
    predictions = model.predict(raster_scaled, batch_size=batch_size, show_progress=False)
    predict_time = time.time() - start_time
    print(f"  Completed in {predict_time:.2f}s")

    return np.array(predictions), model

# Generate predictions for each kernel
raster_predictions = {}
fitted_models = {}

for name, config in KERNEL_CONFIGS.items():
    predictions, model = generate_raster_predictions_for_kernel(
        name, config,
        train_data_scaled,
        raster_scaled,
        sigma=SIGMA,
        window_size=5,
        batch_size=1000
    )
    raster_predictions[name] = predictions
    fitted_models[name] = model

    # Store predictions in kernel_results for later use
    if name in kernel_results:
        kernel_results[name].raster_predictions = predictions

print()
print("="*80)
print(f"✓ Generated raster predictions for {len(raster_predictions)} models")
print("="*80)


# ============================================================================
# NEW CELL: Markdown Header
# ============================================================================
"""
### 5.3 Prediction Maps with Train/Test Sites

Visualize each model's spatial predictions with training sites (red circles)
and test sites (blue triangles) overlaid.
"""

# ============================================================================
# NEW CELL: Prediction Map Visualizations
# ============================================================================
# Create prediction maps for each kernel
n_models = len(KERNEL_CONFIGS)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8*n_rows))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (name, predictions) in enumerate(raster_predictions.items()):
    ax = axes[idx]

    # Plot predictions
    im = ax.imshow(predictions, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')

    # Overlay TRAINING sites (red circles)
    for _, row in train_sites_gdf.iterrows():
        row_idx, col = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
        ax.plot(col, row_idx, 'o', color='red', markersize=8,
               markeredgecolor='white', markeredgewidth=2,
               label='Training Sites' if _ == 0 else '')

    # Overlay TEST sites (blue triangles)
    for _, row in test_sites_gdf.iterrows():
        row_idx, col = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
        ax.plot(col, row_idx, '^', color='blue', markersize=8,
               markeredgecolor='white', markeredgewidth=2,
               label='Test Sites' if _ == 0 else '')

    # Get model performance
    result = kernel_results.get(name)
    if result:
        train_auc = result.train_auc
        test_auc = result.auc
        gap = train_auc - test_auc

        title = f'{name}\nTrain AUC: {train_auc:.3f} | Test AUC: {test_auc:.3f} | Gap: {gap:+.3f}'
    else:
        title = name

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Column', fontsize=10)
    ax.set_ylabel('Row', fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Probability', rotation=270, labelpad=15, fontsize=10)

    # Add legend (only for first plot)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Hide unused subplots
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Model Predictions on Full Raster - {DIFFICULTY.capitalize()} Difficulty\n'
             f'Red Circles = Training Sites | Blue Triangles = Test Sites (Held-Out)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print()
print("Prediction Map Summary:")
print("="*80)
for name, predictions in raster_predictions.items():
    pred_min = np.min(predictions)
    pred_max = np.max(predictions)
    pred_mean = np.mean(predictions)
    high_prob = np.sum(predictions > 0.7) / predictions.size * 100

    print(f"{name}:")
    print(f"  Min: {pred_min:.4f}, Max: {pred_max:.4f}, Mean: {pred_mean:.4f}")
    print(f"  High probability (>0.7): {high_prob:.1f}% of pixels")


# ============================================================================
# NEW CELL: Side-by-Side Comparison (Best vs Baseline)
# ============================================================================
# Compare best performing kernel vs logistic regression baseline
print()
print("="*80)
print("BEST MODEL vs BASELINE COMPARISON")
print("="*80)

# Find best kernel by test AUC (excluding LR)
kernel_only_results = {k: v for k, v in kernel_results.items() if k != 'Logistic Regression'}
best_kernel_name = max(kernel_only_results.keys(), key=lambda k: kernel_only_results[k].auc)
best_kernel_result = kernel_only_results[best_kernel_name]

print(f"Best Kernel: {best_kernel_name} (Test AUC: {best_kernel_result.auc:.4f})")
print()

# Create side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Best Kernel
if best_kernel_name in raster_predictions:
    im0 = axes[0].imshow(raster_predictions[best_kernel_name], cmap='RdYlGn',
                        vmin=0, vmax=1, interpolation='nearest')

    # Overlay sites
    for _, row in train_sites_gdf.iterrows():
        row_idx, col = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
        axes[0].plot(col, row_idx, 'o', color='red', markersize=6,
                    markeredgecolor='white', markeredgewidth=1.5)

    for _, row in test_sites_gdf.iterrows():
        row_idx, col = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
        axes[0].plot(col, row_idx, '^', color='blue', markersize=6,
                    markeredgecolor='white', markeredgewidth=1.5)

    axes[0].set_title(f'{best_kernel_name}\nTest AUC: {best_kernel_result.auc:.3f} '
                     f'(Train: {best_kernel_result.train_auc:.3f})',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Plot 2: Feature 1 (for context)
im1 = axes[1].imshow(raster_stack.data[0], cmap='viridis', interpolation='nearest')

# Overlay sites
for _, row in train_sites_gdf.iterrows():
    row_idx, col = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
    axes[1].plot(col, row_idx, 'o', color='red', markersize=6,
                markeredgecolor='white', markeredgewidth=1.5,
                label='Training' if _ == 0 else '')

for _, row in test_sites_gdf.iterrows():
    row_idx, col = rasterio.transform.rowcol(transform, row.geometry.x, row.geometry.y)
    axes[1].plot(col, row_idx, '^', color='blue', markersize=6,
                markeredgecolor='white', markeredgewidth=1.5,
                label='Test (Held-Out)' if _ == 0 else '')

axes[1].set_title('Environmental Feature 1\n(Training Data Context)',
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')
axes[1].legend(loc='upper right', fontsize=9)
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Plot 3: Prediction uncertainty (variance across models)
all_preds = np.array([pred for pred in raster_predictions.values()])
pred_std = np.std(all_preds, axis=0)

im2 = axes[2].imshow(pred_std, cmap='hot_r', interpolation='nearest')
axes[2].set_title('Prediction Uncertainty\n(Std Dev Across All Models)',
                 fontsize=12, fontweight='bold')
axes[2].set_xlabel('Column')
axes[2].set_ylabel('Row')
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Std Dev')

plt.suptitle(f'Model Comparison: Best Kernel vs Environmental Context\n'
             f'{DIFFICULTY.capitalize()} Difficulty | {len(train_sites_gdf)} Train Sites, '
             f'{len(test_sites_gdf)} Test Sites',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

print()
print("✓ Prediction maps generated successfully!")
print(f"✓ Train sites (red circles): {len(train_sites_gdf)}")
print(f"✓ Test sites (blue triangles): {len(test_sites_gdf)}")
print("="*80)
