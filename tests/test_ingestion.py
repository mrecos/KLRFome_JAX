"""Tests for equivalent tabular/raster bag ingestion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import Point, box

from klrfome.io.raster_source import (
    RasterSource,
    align_bags_to_raster,
    build_spatial_background_bags,
)
from klrfome.io.tabular import TabularBagConfig, load_tabular_bags


def _write(path: Path, values: np.ndarray, transform=None, nodata=-9999.0) -> str:
    transform = transform or from_origin(0, 6, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:26918",
        transform=transform,
        nodata=nodata,
    ) as destination:
        destination.write(values.astype("float32"), 1)
    return str(path)


@pytest.fixture
def raster_source(tmp_path):
    a = np.arange(36, dtype=float).reshape(6, 6)
    b = a * 10
    b[2, 2] = -9999
    paths = [_write(tmp_path / "a.tif", a), _write(tmp_path / "b.tif", b)]
    return RasterSource(paths, band_names=["a", "b"])


def test_tabular_and_raster_alignment_produce_identical_values(raster_source):
    rows = np.array([1, 1, 3, 4])
    cols = np.array([1, 1, 3, 4])
    xs, ys = rasterio.transform.xy(raster_source.transform, rows, cols, offset="center")
    frame = pd.DataFrame(
        {
            "a": [7, 7, 21, 28],
            "b": [70, 70, 210, 280],
            "x": xs,
            "y": ys,
            "SITENO": [9, 9, 9, 9],
            "presence": [1, 1, 1, 1],
        }
    )
    tabular = load_tabular_bags(
        frame,
        TabularBagConfig(feature_columns=["a", "b"], crs="EPSG:26918", min_unique_cells=3),
    )
    assert tabular.metadata["duplicates_removed"] == 1
    aligned = align_bags_to_raster(tabular, raster_source, min_cells=3)
    np.testing.assert_allclose(aligned.collections[0].samples, [[7, 70], [21, 210], [28, 280]])
    assert aligned.feature_names == tabular.feature_names
    assert aligned.crs == tabular.crs


def test_raster_source_rejects_misalignment(tmp_path):
    values = np.ones((4, 4))
    first = _write(tmp_path / "first.tif", values, from_origin(0, 4, 1, 1))
    second = _write(tmp_path / "second.tif", values, from_origin(0.5, 4, 1, 1))
    with pytest.raises(ValueError, match="not aligned"):
        RasterSource([first, second])


def test_polygon_and_point_window_are_windowed_and_use_all_band_mask(raster_source):
    polygon = box(1, 2, 4, 5)
    bag = raster_source.extract_geometry(polygon, "polygon", 1)
    # The 3x3 polygon covers nine cells, but band b masks row=2,col=2.
    assert bag.n_samples == 8
    assert raster_source.read_log[-1] is not None
    assert raster_source.read_log[-1].width < raster_source.width
    with pytest.raises(ValueError, match="requires"):
        raster_source.extract_geometry(Point(2.5, 3.5), "point", 1)
    point_bag = raster_source.extract_geometry(Point(4.5, 1.5), "point", 1, window_size=3)
    assert 1 <= point_bag.n_samples <= 9


def test_batched_window_reads_match_individual_reads(raster_source):
    windows = [Window(0, 0, 3, 3), Window(2, 2, 3, 3)]
    expected = [raster_source.read_window(window) for window in windows]
    observed = list(raster_source.read_windows(windows))
    for (expected_values, expected_valid), (observed_values, observed_valid) in zip(
        expected, observed
    ):
        np.testing.assert_allclose(observed_values, expected_values, equal_nan=True)
        np.testing.assert_array_equal(observed_valid, expected_valid)


def test_background_bags_match_sizes_and_never_overlap_sites(raster_source):
    site = raster_source.extract_geometry(box(0, 4, 2, 6), "site", 1)
    backgrounds = build_spatial_background_bags(
        raster_source, [site], n_background=2, cap_cells=4, seed=7
    )
    site_cells = {tuple(cell) for cell in site.metadata["cell_indices"]}
    assert [bag.n_samples for bag in backgrounds] == [min(site.n_samples, 4)] * 2
    for bag in backgrounds:
        assert not site_cells & {tuple(cell) for cell in bag.metadata["cell_indices"]}


def test_background_bags_can_match_site_sizes_exactly(raster_source):
    small = raster_source.extract_geometry(box(0, 4, 2, 6), "small", 1)
    large = raster_source.extract_geometry(box(2, 0, 6, 2), "large", 1)
    backgrounds = build_spatial_background_bags(
        raster_source,
        [small, large],
        n_background=2,
        cap_cells=20,
        seed=3,
        match_sizes_exactly=True,
    )
    assert sorted(bag.n_samples for bag in backgrounds) == sorted(
        [small.n_samples, large.n_samples]
    )
    assert {bag.metadata["size_matching"] for bag in backgrounds} == {"exact_permutation"}

    with pytest.raises(ValueError, match="n_background"):
        build_spatial_background_bags(
            raster_source,
            [small, large],
            n_background=3,
            match_sizes_exactly=True,
        )
