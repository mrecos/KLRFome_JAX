"""Raster I/O utilities for KLRfome."""

from ..data.formats import RasterStack

__all__ = ["RasterStack", "save_raster", "load_raster"]


def save_raster(file_path: str, data, transform, crs: str, nodata=None):
    """Save a 2D or band-first 3D array as a GeoTIFF."""
    import numpy as np
    import rasterio
    from rasterio.crs import CRS

    data = np.array(data)
    if data.ndim == 2:
        height, width = data.shape
        count = 1
        data = data.reshape(1, height, width)
    elif data.ndim == 3:
        count, height, width = data.shape
    else:
        raise ValueError("Data must be 2D or 3D")

    crs_obj = CRS.from_string(crs) if crs else None
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs_obj,
        transform=transform,
        nodata=nodata,
    ) as destination:
        destination.write(data)


def load_raster(file_path: str) -> RasterStack:
    """Load a raster file as an eager compatibility ``RasterStack``."""
    return RasterStack.from_multiband(file_path)
