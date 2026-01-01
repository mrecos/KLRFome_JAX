#!/usr/bin/env python3
"""
Generate benchmark data for comparing Python/JAX and R implementations.

This script creates synthetic rasters and site points that can be used
by both Python and R scripts for performance benchmarking.

Usage:
    python benchmarks/generate_benchmark_data.py [--output-dir OUTPUT_DIR] [--seed SEED]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point

# Add parent directory to path to import klrfome
sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome import RasterStack
from klrfome.data.simulation import sim_trend, rescale_sim_raster, nlm_gaussianfield


def generate_benchmark_data(
    output_dir: str = "benchmark_data",
    cols: int = 200,
    rows: int = 200,
    n_sites: int = 25,
    autocorr_range: float = 20,
    seed: int = 42
):
    """
    Generate synthetic rasters and site points for benchmarking.
    
    Parameters:
        output_dir: Directory to save output files
        cols: Number of columns in raster
        rows: Number of rows in raster
        n_sites: Number of site locations
        autocorr_range: Spatial autocorrelation range
        seed: Random seed for reproducibility
    """
    print(f"Generating benchmark data with seed={seed}...")
    print(f"  Raster size: {rows}x{cols}")
    print(f"  Number of sites: {n_sites}")
    print(f"  Autocorrelation range: {autocorr_range}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create site-likely rasters (conditions favorable for sites)
    s_var1r = nlm_gaussianfield(cols, rows, autocorr_range=autocorr_range, seed=seed)
    s_var1 = rescale_sim_raster(s_var1r, 50, 10)   # Mean=50, SD=10
    s_var2 = rescale_sim_raster(s_var1r, 3, 2)     # Mean=3, SD=2
    s_var3 = rescale_sim_raster(s_var1r, 5, 1.5)   # Mean=5, SD=1.5
    
    # Step 2: Create site-unlikely rasters (background conditions)
    b_var1r = nlm_gaussianfield(cols, rows, autocorr_range=autocorr_range, seed=seed + 1000)
    b_var1 = rescale_sim_raster(b_var1r, 100, 20)  # Mean=100, SD=20
    b_var2 = rescale_sim_raster(b_var1r, 6, 3)     # Mean=6, SD=3
    b_var3 = rescale_sim_raster(b_var1r, 10, 2)    # Mean=10, SD=2
    
    # Step 3: Place sites in favorable areas
    favorability_surface = s_var1r
    trend_result = sim_trend(
        cols, rows,
        n=n_sites,
        size=6,
        seed=seed,
        favorable_surface=favorability_surface,
        favor_threshold=0.3
    )
    trend = trend_result['trend']
    site_coords_pixels = trend_result['coords']
    
    # Step 4: Create stronger trend surface
    trend_power = 2.0
    trend = np.power(trend, trend_power)
    inv_trend = 1 - trend
    
    # Step 5: Combine rasters
    mix_strength = 0.8
    var1 = (s_var1 * (trend * mix_strength + (1 - mix_strength) * 0.5)) + \
           (b_var1 * (inv_trend * mix_strength + (1 - mix_strength) * 0.5))
    var2 = (s_var2 * (trend * mix_strength + (1 - mix_strength) * 0.5)) + \
           (b_var2 * (inv_trend * mix_strength + (1 - mix_strength) * 0.5))
    var3 = (s_var3 * (trend * mix_strength + (1 - mix_strength) * 0.5)) + \
           (b_var3 * (inv_trend * mix_strength + (1 - mix_strength) * 0.5))
    
    # Step 6: Create RasterStack and save rasters
    transform = from_bounds(0, 0, 1, 1, cols, rows)
    band_names = ['var1', 'var2', 'var3']
    
    # Save each band as a separate GeoTIFF (compatible with R)
    raster_files = []
    for i, (band_name, band_data) in enumerate(zip(band_names, [var1, var2, var3])):
        raster_file = output_path / f"{band_name}.tif"
        
        with rasterio.open(
            raster_file,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=band_data.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(band_data, 1)
        
        raster_files.append(str(raster_file))
        print(f"  Saved raster: {raster_file}")
    
    # Step 7: Create and save site points
    site_points = []
    for coord in site_coords_pixels:
        x, y = rasterio.transform.xy(transform, coord[1], coord[0])
        site_points.append(Point(x, y))
    
    sites_gdf = gpd.GeoDataFrame(
        geometry=site_points,
        crs="EPSG:4326"
    )
    
    # Save as shapefile (compatible with R)
    sites_file = output_path / "sites.shp"
    sites_gdf.to_file(sites_file)
    print(f"  Saved sites: {sites_file}")
    
    # Also save as GeoJSON (easier to read)
    sites_geojson = output_path / "sites.geojson"
    sites_gdf.to_file(sites_geojson, driver='GeoJSON')
    print(f"  Saved sites (GeoJSON): {sites_geojson}")
    
    # Save metadata
    metadata = {
        'seed': seed,
        'cols': cols,
        'rows': rows,
        'n_sites': n_sites,
        'autocorr_range': autocorr_range,
        'raster_files': raster_files,
        'sites_file': str(sites_file),
        'band_names': band_names,
        'transform': {
            'xmin': 0.0,
            'ymin': 0.0,
            'xmax': 1.0,
            'ymax': 1.0,
            'width': cols,
            'height': rows
        }
    }
    
    import json
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_file}")
    
    print(f"\n✓ Benchmark data generated successfully in: {output_path.absolute()}")
    print(f"  Rasters: {len(raster_files)} files")
    print(f"  Sites: {len(sites_gdf)} locations")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark data for KLRfome performance comparison"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_data',
        help='Output directory for benchmark data (default: benchmark_data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--cols',
        type=int,
        default=200,
        help='Number of columns in raster (default: 200)'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=200,
        help='Number of rows in raster (default: 200)'
    )
    parser.add_argument(
        '--n-sites',
        type=int,
        default=25,
        help='Number of site locations (default: 25)'
    )
    parser.add_argument(
        '--autocorr-range',
        type=float,
        default=20.0,
        help='Spatial autocorrelation range (default: 20.0)'
    )
    
    args = parser.parse_args()
    
    generate_benchmark_data(
        output_dir=args.output_dir,
        cols=args.cols,
        rows=args.rows,
        n_sites=args.n_sites,
        autocorr_range=args.autocorr_range,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

