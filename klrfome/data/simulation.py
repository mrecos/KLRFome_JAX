"""Data simulation functions for testing and examples."""

import numpy as np
import jax.numpy as jnp
import jax.random as random
import pandas as pd
from typing import Dict, List, Tuple, Optional
from rasterio.transform import from_bounds

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
    seed: int = 42
) -> Dict:
    """
    Create a trend surface for simulated landscapes.
    
    Generates n site locations and creates a distance gradient from them.
    
    Parameters:
        cols: Number of columns in raster
        rows: Number of rows in raster
        n: Number of simulated site locations
        size: Size of simulated sites in pixels
        seed: Random seed
    
    Returns:
        Dictionary with 'coords' (site coordinates) and 'trend' (trend raster array)
    """
    np.random.seed(seed)
    
    # Generate random site locations
    site_coords = []
    for _ in range(n):
        x = np.random.randint(size, cols - size)
        y = np.random.randint(size, rows - size)
        site_coords.append([x, y])
    
    site_coords = np.array(site_coords)
    
    # Create distance gradient
    trend = np.zeros((rows, cols))
    
    for row in range(rows):
        for col in range(cols):
            # Distance to nearest site
            distances = np.sqrt(
                (site_coords[:, 0] - col)**2 + (site_coords[:, 1] - row)**2
            )
            min_dist = np.min(distances)
            
            # Normalize to max possible distance
            max_dist = np.sqrt(cols**2 + rows**2)
            trend[row, col] = 1 - (min_dist / max_dist)
    
    return {
        'coords': site_coords,
        'trend': trend
    }


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

