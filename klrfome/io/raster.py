"""Raster I/O utilities for KLRfome."""

from ..data.formats import RasterStack

__all__ = ["RasterStack", "save_raster", "load_raster"]


def save_raster(
    file_path: str,
    data,
    transform,
    crs: str,
    nodata=None
):
    """
    Save a 2D or 3D array as a GeoTIFF.
    
    Parameters:
        file_path: Output file path
        data: Array to save (2D or 3D: height x width or bands x height x width)
        transform: Affine transformation
        crs: Coordinate reference system
        nodata: No-data value
    """
    import rasterio
    from rasterio.crs import CRS
    import numpy as np
    
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
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs_obj,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data)


def load_raster(file_path: str) -> RasterStack:
    """
    Load a raster file as a RasterStack.
    
    Parameters:
        file_path: Path to raster file
    
    Returns:
        RasterStack object
    """
    return RasterStack.from_multiband(file_path)

