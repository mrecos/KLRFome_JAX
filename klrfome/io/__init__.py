"""I/O utilities for raster and vector data."""

from .raster import RasterStack, save_raster, load_raster
from .raster_source import RasterSource, align_bags_to_raster, build_spatial_background_bags
from .tabular import TabularBagConfig, load_tabular_bags, resolve_tabular_config
from .vector import extract_at_points, generate_background_points

__all__ = [
    "RasterStack",
    "save_raster",
    "load_raster",
    "extract_at_points",
    "generate_background_points",
    "RasterSource",
    "align_bags_to_raster",
    "build_spatial_background_bags",
    "TabularBagConfig",
    "load_tabular_bags",
    "resolve_tabular_config",
]
