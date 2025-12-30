"""Benchmark prediction performance."""

import time
import jax.numpy as jnp
import jax.random as random
import numpy as np
from rasterio.transform import from_bounds

from klrfome.data.formats import SampleCollection, TrainingData, RasterStack
from klrfome.kernels.rff import RandomFourierFeatures
from klrfome.kernels.distribution import MeanEmbeddingKernel
from klrfome.models.klr import KernelLogisticRegression
from klrfome.prediction.focal import FocalPredictor


def benchmark_focal_prediction(
    raster_size: int = 100,
    window_size: int = 3,
    n_training: int = 10
):
    """Benchmark focal window prediction."""
    key = random.PRNGKey(42)
    
    # Create training data
    collections = []
    for i in range(n_training):
        samples = random.normal(key, (20, 3))
        key, _ = random.split(key)
        coll = SampleCollection(
            samples=samples,
            label=1 if i < n_training // 2 else 0,
            id=f"loc_{i}"
        )
        collections.append(coll)
    
    training_data = TrainingData(
        collections=collections,
        feature_names=["var1", "var2", "var3"]
    )
    
    # Build kernel and fit model
    rff = RandomFourierFeatures(sigma=1.0, n_features=128, seed=42)
    kernel = MeanEmbeddingKernel(rff)
    K = kernel.build_similarity_matrix(collections)
    
    klr = KernelLogisticRegression(lambda_reg=0.1)
    result = klr.fit(K, training_data.labels)
    
    # Create raster
    data = jnp.array(np.random.rand(3, raster_size, raster_size))
    transform = from_bounds(0, 0, 1, 1, raster_size, raster_size)
    raster_stack = RasterStack(
        data=data,
        transform=transform,
        crs="EPSG:4326",
        band_names=["var1", "var2", "var3"]
    )
    
    # Create predictor
    predictor = FocalPredictor(
        distribution_kernel=kernel,
        klr_alpha=result.alpha,
        training_data=training_data,
        window_size=window_size
    )
    
    # Warm up
    _ = predictor.predict_raster(raster_stack, batch_size=100, show_progress=False)
    
    # Time prediction
    start = time.time()
    predictions = predictor.predict_raster(
        raster_stack, 
        batch_size=1000, 
        show_progress=False
    )
    elapsed = time.time() - start
    
    n_pixels = raster_size * raster_size
    print(f"Focal Prediction: {raster_size}x{raster_size} raster, {window_size}x{window_size} window")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Throughput: {n_pixels / elapsed:.2f} pixels/second")
    
    return elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Focal Prediction Benchmark")
    print("=" * 60)
    benchmark_focal_prediction(raster_size=50, window_size=3)

