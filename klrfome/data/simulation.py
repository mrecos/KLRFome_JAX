"""Data simulation functions for testing and examples."""

import numpy as np
import jax.numpy as jnp
import jax.random as random
import pandas as pd
from typing import Dict, List, Tuple, Optional
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter

from ..data.formats import SampleCollection, TrainingData, RasterStack


def get_sim_data(
    site_samples: int = 800,
    N_site_bags: int = 75,
    sites_var1_mean: float = 50,
    sites_var1_sd: float = 10,
    sites_var2_mean: float = 3,
    sites_var2_sd: float = 2,
    backg_var1_mean: float = 100,
    backg_var1_sd: float = 20,
    backg_var2_mean: float = 6,
    backg_var2_sd: float = 3,
    background_site_balance: int = 1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate archaeological site data for testing.
    
    This function generates simulated site and background data with configurable
    distribution parameters. Similar to the R version.
    
    Parameters:
        site_samples: Number of site samples to generate
        N_site_bags: Number of site groups (bags)
        sites_var1_mean: Mean of variable 1 for sites
        sites_var1_sd: Standard deviation of variable 1 for sites
        sites_var2_mean: Mean of variable 2 for sites
        sites_var2_sd: Standard deviation of variable 2 for sites
        backg_var1_mean: Mean of variable 1 for background
        backg_var1_sd: Standard deviation of variable 1 for background
        backg_var2_mean: Mean of variable 2 for background
        backg_var2_sd: Standard deviation of variable 2 for background
        background_site_balance: Ratio of background to site groups
        seed: Random seed
    
    Returns:
        DataFrame with columns: presence, SITENO, var1, var2
    """
    np.random.seed(seed)
    
    # Generate site data
    site_nos = np.random.choice(
        [f"Site{i}" for i in range(1, N_site_bags + 1)],
        size=site_samples,
        replace=True
    )
    
    sites = pd.DataFrame({
        'presence': 1,
        'SITENO': site_nos,
        'var1': np.random.normal(sites_var1_mean, sites_var1_sd, site_samples),
        'var2': np.random.normal(sites_var2_mean, sites_var2_sd, site_samples)
    })
    
    # Generate background data
    back_samples = site_samples * background_site_balance
    backs = pd.DataFrame({
        'presence': 0,
        'SITENO': 'background',
        'var1': np.random.normal(backg_var1_mean, backg_var1_sd, back_samples),
        'var2': np.random.normal(backg_var2_mean, backg_var2_sd, back_samples)
    })
    
    # Combine and sort
    sim_data = pd.concat([sites, backs], ignore_index=True)
    sim_data = sim_data.sort_values('SITENO').reset_index(drop=True)
    
    return sim_data


def sim_trend(
    cols: int,
    rows: int,
    n: int = 3,
    size: int = 6,
    seed: int = 42,
    favorable_surface: Optional[np.ndarray] = None,
    favor_threshold: float = 0.3
) -> Dict:
    """
    Create a trend surface for simulated landscapes.
    
    Generates n site locations and creates a distance gradient from them.
    If favorable_surface is provided, sites are placed in favorable areas
    (where values are below favor_threshold percentile).
    
    Parameters:
        cols: Number of columns in raster
        rows: Number of rows in raster
        n: Number of simulated site locations
        size: Size of simulated sites in pixels
        seed: Random seed
        favorable_surface: Optional 2D array indicating favorable areas (lower = more favorable)
        favor_threshold: Percentile threshold for favorable areas (0-1, lower = more selective)
    
    Returns:
        Dictionary with 'coords' (site coordinates) and 'trend' (trend raster array)
    """
    np.random.seed(seed)
    
    # Generate site locations
    site_coords = []
    
    if favorable_surface is not None:
        # Place sites in favorable areas (where values are low)
        # Find favorable pixels (below threshold percentile)
        threshold_val = np.percentile(favorable_surface, favor_threshold * 100)
        favorable_mask = favorable_surface <= threshold_val
        
        # Get coordinates of favorable pixels
        favorable_coords = np.column_stack(np.where(favorable_mask))
        
        if len(favorable_coords) < n:
            # If not enough favorable pixels, use all pixels
            favorable_coords = np.column_stack(np.where(np.ones((rows, cols), dtype=bool)))
        
        # Sample n sites from favorable areas, ensuring minimum distance
        min_distance = size * 2  # Minimum distance between sites
        selected_indices = []
        
        for _ in range(n):
            if len(selected_indices) == 0:
                # First site: random from favorable areas
                idx = np.random.randint(0, len(favorable_coords))
                selected_indices.append(idx)
            else:
                # Subsequent sites: ensure minimum distance from existing sites
                attempts = 0
                max_attempts = 1000
                while attempts < max_attempts:
                    idx = np.random.randint(0, len(favorable_coords))
                    new_coord = favorable_coords[idx]
                    
                    # Check distance to all existing sites
                    too_close = False
                    for sel_idx in selected_indices:
                        existing_coord = favorable_coords[sel_idx]
                        dist = np.sqrt((new_coord[0] - existing_coord[0])**2 + 
                                     (new_coord[1] - existing_coord[1])**2)
                        if dist < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        selected_indices.append(idx)
                        break
                    attempts += 1
                
                # If we couldn't find a good spot, just pick randomly
                if attempts >= max_attempts:
                    idx = np.random.randint(0, len(favorable_coords))
                    selected_indices.append(idx)
        
        # Convert to (col, row) format
        for idx in selected_indices:
            row, col = favorable_coords[idx]
            # Ensure within bounds
            x = max(size, min(cols - size - 1, col))
            y = max(size, min(rows - size - 1, row))
            site_coords.append([x, y])
    else:
        # Random placement (original behavior)
        for _ in range(n):
            x = np.random.randint(size, cols - size)
            y = np.random.randint(size, rows - size)
            site_coords.append([x, y])
    
    site_coords = np.array(site_coords)
    
    # Create distance gradient with exponential decay for stronger signal
    trend = np.zeros((rows, cols))
    
    # Use a decay parameter to control how quickly trend decreases with distance
    decay_factor = 0.1  # Smaller = faster decay, stronger signal near sites
    
    for row in range(rows):
        for col in range(cols):
            # Distance to nearest site
            distances = np.sqrt(
                (site_coords[:, 0] - col)**2 + (site_coords[:, 1] - row)**2
            )
            min_dist = np.min(distances)
            
            # Exponential decay: trend = exp(-decay_factor * distance)
            # Normalize so trend is 1 at sites (distance = 0)
            # Use a scale factor to control the effective range
            scale = max(cols, rows) / 10.0  # Adjust this to control range
            trend[row, col] = np.exp(-decay_factor * min_dist / scale)
    
    # Normalize to [0, 1] range
    trend_min = np.min(trend)
    trend_max = np.max(trend)
    if trend_max > trend_min:
        trend = (trend - trend_min) / (trend_max - trend_min)
    
    return {
        'coords': site_coords,
        'trend': trend
    }


def nlm_gaussianfield(
    cols: int,
    rows: int,
    autocorr_range: float = 20.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a spatially autocorrelated Gaussian random field.
    
    Python equivalent of R's `NLMR::nlm_gaussianfield()`. Creates a 2D
    Gaussian random field with spatial autocorrelation by applying a
    Gaussian filter to random noise.
    
    Parameters:
        cols: Number of columns in the output raster
        rows: Number of rows in the output raster
        autocorr_range: Spatial autocorrelation range (controls the degree
            of spatial correlation). Larger values create smoother, more
            correlated fields. Similar to the `autocorr_range` parameter
            in the R function. This is converted to sigma for the Gaussian
            filter: sigma = autocorr_range / 3 (approximate conversion).
        seed: Random seed for reproducibility
    
    Returns:
        2D numpy array of shape (rows, cols) with spatially correlated values
        in the range [0, 1]
    
    Examples:
        >>> field = nlm_gaussianfield(100, 100, autocorr_range=20, seed=42)
        >>> field.shape
        (100, 100)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random noise
    noise = np.random.rand(rows, cols)
    
    # Convert autocorr_range to sigma for Gaussian filter
    # The autocorr_range in R corresponds roughly to 3*sigma in the filter
    # This is an approximate conversion based on the effective range of
    # spatial correlation in a Gaussian filter
    sigma = autocorr_range / 3.0
    
    # Apply Gaussian filter to create spatial autocorrelation
    # This smooths the random noise, creating spatially correlated patterns
    correlated_field = gaussian_filter(noise, sigma=sigma)
    
    # Normalize to [0, 1] range (matching R function behavior)
    min_val = np.min(correlated_field)
    max_val = np.max(correlated_field)
    if max_val > min_val:
        normalized = (correlated_field - min_val) / (max_val - min_val)
    else:
        normalized = correlated_field
    
    return normalized


def rescale_sim_raster(
    raster_data: np.ndarray,
    mean: float,
    sd: float
) -> np.ndarray:
    """
    Rescale raster to match specified mean and standard deviation.
    
    Parameters:
        raster_data: Input raster array
        mean: Target mean
        sd: Target standard deviation
    
    Returns:
        Rescaled raster array
    """
    current_mean = np.mean(raster_data)
    current_sd = np.std(raster_data)
    
    if current_sd == 0:
        return np.full_like(raster_data, mean)
    
    rescaled = mean + (raster_data - current_mean) * (sd / current_sd)
    return rescaled


def create_simulated_raster_stack(
    cols: int = 100,
    rows: int = 100,
    n_bands: int = 2,
    seed: int = 42
) -> RasterStack:
    """
    Create a simulated raster stack for testing.
    
    Parameters:
        cols: Number of columns
        rows: Number of rows
        n_bands: Number of bands
        seed: Random seed
    
    Returns:
        RasterStack with simulated data
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.rand(n_bands, rows, cols)
    
    # Convert to JAX array
    data_jax = jnp.array(data)
    
    transform = from_bounds(0, 0, 1, 1, cols, rows)
    band_names = [f"var{i+1}" for i in range(n_bands)]
    
    return RasterStack(
        data=data_jax,
        transform=transform,
        crs="EPSG:4326",
        band_names=band_names
    )

