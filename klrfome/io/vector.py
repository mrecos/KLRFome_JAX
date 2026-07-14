"""Vector I/O utilities for GeoPandas integration."""

from typing import List, Optional
import numpy as np
import jax.numpy as jnp

from ..data.formats import SampleCollection, RasterStack


def extract_at_points(
    raster_stack: RasterStack,
    points: "geopandas.GeoDataFrame",  # type: ignore
    buffer_radius: Optional[float] = None,
    n_samples: int = 10,
    random_seed: Optional[int] = None,
) -> List[SampleCollection]:
    """
    Extract samples from raster at point/polygon locations.

    Parameters:
        raster_stack: RasterStack to extract from
        points: GeoDataFrame with point or polygon geometries
        buffer_radius: Optional buffer radius around points (in CRS units)
        n_samples: Number of samples to extract per location
        random_seed: Random seed for sampling

    Returns:
        List of SampleCollection objects
    """
    import rasterio
    from shapely.geometry import Point

    if random_seed is not None:
        np.random.seed(random_seed)

    collections = []

    # Convert raster transform to something we can use
    transform = raster_stack.transform

    for idx, row in points.iterrows():
        geom = row.geometry

        # Apply buffer if specified
        if buffer_radius is not None:
            if geom.geom_type == "Point":
                geom = geom.buffer(buffer_radius)
            else:
                geom = geom.buffer(buffer_radius)

        # Get bounding box of geometry
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        # Convert to pixel coordinates
        # Use rasterio's rowcol function - returns (row, col)
        min_row, min_col = rasterio.transform.rowcol(transform, bounds[0], bounds[3])
        max_row, max_col = rasterio.transform.rowcol(transform, bounds[2], bounds[1])

        # Ensure we're within raster bounds
        min_row = max(0, min_row)
        max_row = min(raster_stack.height - 1, max_row)
        min_col = max(0, min_col)
        max_col = min(raster_stack.width - 1, max_col)

        # Extract samples within the geometry
        samples_list = []

        # Generate random points within the geometry
        attempts = 0
        max_attempts = n_samples * 100  # Prevent infinite loop

        while len(samples_list) < n_samples and attempts < max_attempts:
            # Random point in bounding box
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])

            point = Point(x, y)

            # Check if point is within geometry
            if geom.contains(point) or geom.intersects(point):
                # Convert to pixel coordinates - rowcol returns (row, col)
                row, col = rasterio.transform.rowcol(transform, x, y)

                # Check bounds
                if 0 <= row < raster_stack.height and 0 <= col < raster_stack.width:

                    # Extract pixel values for all bands
                    pixel_values = []
                    for band_idx in range(raster_stack.n_bands):
                        value = float(raster_stack.data[band_idx, row, col])
                        # Skip nodata values
                        if raster_stack.nodata is None or value != raster_stack.nodata:
                            pixel_values.append(value)

                    if len(pixel_values) == raster_stack.n_bands:
                        samples_list.append(pixel_values)

            attempts += 1

        # If we didn't get enough samples, use grid sampling as fallback
        if len(samples_list) < n_samples:
            # Use grid of points within geometry
            for row in range(min_row, max_row + 1, max(1, (max_row - min_row) // n_samples)):
                for col in range(min_col, max_col + 1, max(1, (max_col - min_col) // n_samples)):
                    if len(samples_list) >= n_samples:
                        break

                    # Convert pixel to geographic coordinates
                    x, y = rasterio.transform.xy(transform, row, col)
                    point = Point(x, y)

                    if geom.contains(point) or geom.intersects(point):
                        pixel_values = []
                        for band_idx in range(raster_stack.n_bands):
                            value = float(raster_stack.data[band_idx, row, col])
                            if raster_stack.nodata is None or value != raster_stack.nodata:
                                pixel_values.append(value)

                        if len(pixel_values) == raster_stack.n_bands:
                            samples_list.append(pixel_values)

                if len(samples_list) >= n_samples:
                    break

        # Create SampleCollection
        if len(samples_list) > 0:
            samples_array = jnp.array(samples_list)
            coll = SampleCollection(
                samples=samples_array,
                label=1,  # Default to site (can be overridden)
                id=f"location_{idx}",
                metadata={"geometry": str(geom), "index": idx},
            )
            collections.append(coll)

    return collections


def generate_background_points(
    raster_stack: RasterStack,
    exclusion_geoms: Optional[List] = None,
    n_points: int = 1000,
    random_seed: Optional[int] = None,
) -> "geopandas.GeoDataFrame":  # type: ignore
    """
    Generate random background sample points.

    Parameters:
        raster_stack: RasterStack to generate points within
        exclusion_geoms: List of geometries to exclude (e.g., site buffers)
        n_points: Number of background points to generate
        random_seed: Random seed

    Returns:
        GeoDataFrame with point geometries
    """
    import geopandas as gpd
    from shapely.geometry import Point
    import rasterio

    if random_seed is not None:
        np.random.seed(random_seed)

    # Get raster bounds
    transform = raster_stack.transform
    bounds = rasterio.transform.array_bounds(
        raster_stack.height, raster_stack.width, transform
    )  # (minx, miny, maxx, maxy)

    points = []
    attempts = 0
    max_attempts = n_points * 10

    while len(points) < n_points and attempts < max_attempts:
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])

        point = Point(x, y)

        # Check if point should be excluded
        if exclusion_geoms is not None:
            excluded = any(
                geom.contains(point) or geom.intersects(point) for geom in exclusion_geoms
            )
            if excluded:
                attempts += 1
                continue

        points.append(point)
        attempts += 1

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=points, crs=raster_stack.crs)

    return gdf


def extract_distribution_at_points(
    raster_stack: RasterStack,
    points: "geopandas.GeoDataFrame",  # type: ignore
    n_samples: int = 20,
    label: int = 1,
    random_seed: Optional[int] = None,
) -> List[SampleCollection]:
    """
    Extract a genuine multi-pixel distribution around each location.

    For each point we take the *distinct* neighbouring raster cells, expanding a
    square window around the centre pixel until at least ``n_samples`` valid cells
    are available (or the raster edge is reached), then draw ``n_samples`` of them.
    This avoids the degenerate "same pixel repeated N times" bags produced by
    random sub-pixel sampling of a zero-area point, so each bag has non-trivial
    within-bag variance and the mean-embedding / Wasserstein kernels actually see
    a distribution. It also makes training bags the same kind of object as the
    focal windows used at prediction time.

    Parameters:
        raster_stack: RasterStack to extract from
        points: GeoDataFrame of point (or polygon) geometries
        n_samples: Number of samples per location (bag size)
        label: Label to assign to every produced collection (1 site, 0 background)
        random_seed: Seed for the cell draw

    Returns:
        List of SampleCollection, one per location with a usable neighbourhood.
    """
    import rasterio

    if random_seed is not None:
        np.random.seed(random_seed)

    data = np.asarray(raster_stack.data)
    n_bands, height, width = data.shape
    nodata = raster_stack.nodata

    collections: List[SampleCollection] = []

    for idx, row in points.iterrows():
        geom = row.geometry
        if geom.geom_type == "Point":
            x, y = geom.x, geom.y
        else:
            x, y = geom.centroid.x, geom.centroid.y

        r, c = rasterio.transform.rowcol(raster_stack.transform, x, y)
        r = int(np.clip(r, 0, height - 1))
        c = int(np.clip(c, 0, width - 1))

        # Grow the neighbourhood until we have enough valid (non-nodata) cells.
        radius = 1
        cells = np.empty((0, n_bands))
        while True:
            r0, r1 = max(0, r - radius), min(height, r + radius + 1)
            c0, c1 = max(0, c - radius), min(width, c + radius + 1)
            block = data[:, r0:r1, c0:c1].reshape(n_bands, -1).T  # (k, n_bands)
            if nodata is not None:
                block = block[~np.any(block == nodata, axis=1)]
            cells = block
            at_edge = r0 == 0 and c0 == 0 and r1 == height and c1 == width
            if cells.shape[0] >= n_samples or at_edge:
                break
            radius += 1

        if cells.shape[0] == 0:
            continue

        replace = cells.shape[0] < n_samples
        sel = np.random.choice(cells.shape[0], size=n_samples, replace=replace)
        samples = cells[sel]

        collections.append(
            SampleCollection(
                samples=jnp.asarray(samples),
                label=label,
                id=f"location_{idx}",
                metadata={"row": r, "col": c},
            )
        )

    return collections
