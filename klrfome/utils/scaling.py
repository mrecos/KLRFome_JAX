"""Feature scaling utilities for KLRfome."""

from typing import Dict, Optional
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..data.formats import RasterStack


def compute_scaling_stats(
    data: Float[Array, "n d"]
) -> Dict[str, Float[Array, "d"]]:
    """
    Compute mean and standard deviation for scaling.
    
    Parameters:
        data: Input data array (n_samples, n_features)
    
    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    
    # Avoid division by zero
    std = jnp.where(std == 0, 1.0, std)
    
    return {'mean': mean, 'std': std}


def scale_data(
    data: Float[Array, "n d"],
    mean: Float[Array, "d"],
    std: Float[Array, "d"]
) -> Float[Array, "n d"]:
    """
    Scale data using mean and standard deviation.
    
    Parameters:
        data: Input data
        mean: Mean values for each feature
        std: Standard deviation values for each feature
    
    Returns:
        Scaled data (z-scores)
    """
    return (data - mean) / std


def scale_prediction_rasters(
    raster_stack: RasterStack,
    mean: Float[Array, "n_bands"],
    std: Float[Array, "n_bands"],
    nodata: Optional[float] = None
) -> RasterStack:
    """
    Scale prediction rasters to match training data statistics.
    
    Parameters:
        raster_stack: Raster stack to scale
        mean: Mean values for each band (from training data)
        std: Standard deviation values for each band (from training data)
        nodata: No-data value to preserve
    
    Returns:
        Scaled RasterStack
    """
    scaled_data = raster_stack.data.copy()
    
    # Scale each band
    for band_idx in range(raster_stack.n_bands):
        band_data = scaled_data[band_idx]
        
        # Mask nodata values
        if nodata is not None:
            mask = band_data != nodata
            band_data = jnp.where(mask, (band_data - mean[band_idx]) / std[band_idx], nodata)
        else:
            band_data = (band_data - mean[band_idx]) / std[band_idx]
        
        scaled_data = scaled_data.at[band_idx].set(band_data)
    
    return RasterStack(
        data=scaled_data,
        transform=raster_stack.transform,
        crs=raster_stack.crs,
        band_names=raster_stack.band_names,
        nodata=nodata or raster_stack.nodata
    )


def get_scaling_from_training_data(
    training_data: 'TrainingData'  # type: ignore
) -> Dict[str, Float[Array, "n_features"]]:
    """
    Extract scaling statistics from training data.
    
    Parameters:
        training_data: TrainingData object
    
    Returns:
        Dictionary with 'mean' and 'std' for each feature
    """
    # Collect all samples
    all_samples = []
    for coll in training_data.collections:
        all_samples.append(coll.samples)
    
    all_data = jnp.concatenate(all_samples, axis=0)
    
    return compute_scaling_stats(all_data)

