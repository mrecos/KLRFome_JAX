"""Lazy, alignment-validated raster ingestion and spatial bag construction."""

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Sequence, Set, Tuple, cast

import jax.numpy as jnp
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from shapely.geometry import mapping

from ..data.formats import Bag, BagDataset

CellIndex = Tuple[int, int]


@dataclass
class RasterSource:
    """A co-registered raster stack that reads only requested windows."""

    paths: Sequence[str]
    band_names: Optional[Sequence[str]] = None
    read_log: List[Optional[Window]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("RasterSource requires at least one raster")
        self.paths = tuple(str(Path(path)) for path in self.paths)
        names = (
            tuple(self.band_names)
            if self.band_names is not None
            else tuple(Path(path).stem for path in self.paths)
        )
        if len(names) != len(self.paths) or len(set(names)) != len(names):
            raise ValueError("band_names must be unique and match the raster count")
        self.band_names = names

        reference = None
        nodata_values = []
        dtypes = []
        for name, path in zip(names, self.paths):
            with rasterio.open(path) as source:
                if source.count != 1:
                    raise ValueError(f"Raster {name!r} must contain exactly one band")
                signature = (
                    source.width,
                    source.height,
                    source.transform,
                    source.crs,
                    tuple(source.res),
                )
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise ValueError(
                        f"Raster {name!r} is not aligned in dimensions, transform, CRS, or resolution"
                    )
                nodata_values.append(source.nodata)
                dtypes.append(source.dtypes[0])
        assert reference is not None
        self.width, self.height, self.transform, crs, self.resolution = reference
        self.crs = crs.to_string() if crs is not None else None
        self.nodata_values = tuple(nodata_values)
        self.dtypes = tuple(dtypes)

    def read_window(self, window: Optional[Window]) -> Tuple[np.ndarray, np.ndarray]:
        """Return band-first values and the all-band validity intersection."""
        values: List[np.ndarray] = []
        valid = None
        for path in self.paths:
            with rasterio.open(path) as source:
                band = source.read(1, window=window, masked=True)
            data = np.asarray(band.filled(np.nan), dtype=float)
            band_valid = ~np.ma.getmaskarray(band) & np.isfinite(data)
            values.append(data)
            valid = band_valid if valid is None else valid & band_valid
        self.read_log.append(window)
        assert valid is not None
        return np.stack(values, axis=0), valid

    def read_windows(self, windows: Iterable[Window]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Read many windows while reusing one open handle per raster band."""
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(path)) for path in self.paths]
            for window in windows:
                values: List[np.ndarray] = []
                valid = None
                for source in sources:
                    band = source.read(1, window=window, masked=True)
                    data = np.asarray(band.filled(np.nan), dtype=float)
                    band_valid = ~np.ma.getmaskarray(band) & np.isfinite(data)
                    values.append(data)
                    valid = band_valid if valid is None else valid & band_valid
                self.read_log.append(window)
                assert valid is not None
                yield np.stack(values, axis=0), valid

    def iter_windows(self, block_size: int = 512) -> Iterable[Window]:
        for row_off in range(0, self.height, block_size):
            for col_off in range(0, self.width, block_size):
                yield Window(
                    col_off,
                    row_off,
                    min(block_size, self.width - col_off),
                    min(block_size, self.height - row_off),
                )

    def valid_mask(self, window: Optional[Window] = None) -> np.ndarray:
        return self.read_window(window)[1]

    def sample_points(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample all bands at x/y pairs and return values, validity, and cell indices."""
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n, 2)")
        rows, cols = rowcol(self.transform, coordinates[:, 0], coordinates[:, 1])
        cell_indices = np.column_stack([rows, cols]).astype(int)
        within = (
            (cell_indices[:, 0] >= 0)
            & (cell_indices[:, 0] < self.height)
            & (cell_indices[:, 1] >= 0)
            & (cell_indices[:, 1] < self.width)
        )
        values: np.ndarray = np.full((len(coordinates), len(self.paths)), np.nan, dtype=float)
        valid = within.copy()
        for band_index, path in enumerate(self.paths):
            with rasterio.open(path) as source:
                sampled = list(source.sample(coordinates, indexes=1, masked=True))
            for index, value in enumerate(sampled):
                if np.ma.is_masked(value) or not within[index]:
                    valid[index] = False
                    continue
                scalar = float(np.asarray(value).reshape(-1)[0])
                if not np.isfinite(scalar):
                    valid[index] = False
                else:
                    values[index, band_index] = scalar
        return values, valid & np.isfinite(values).all(axis=1), cell_indices

    def extract_geometry(
        self,
        geometry: Any,
        bag_id: str,
        label: Literal[0, 1],
        buffer_distance: Optional[float] = None,
        window_size: Optional[int] = None,
        group_id: Optional[str] = None,
        stratum_id: Optional[str] = None,
    ) -> Bag:
        """Extract every valid polygon cell; points require an explicit footprint."""
        geom = geometry
        is_point = getattr(geom, "geom_type", None) == "Point"
        use_square_window = is_point and window_size is not None
        if is_point and buffer_distance is None and window_size is None:
            raise ValueError("Point extraction requires buffer_distance or window_size")
        if is_point and buffer_distance is not None:
            geom = geom.buffer(buffer_distance)

        if use_square_window:
            if window_size is None or window_size < 1:
                raise ValueError("window_size must be positive")
            row, col = rowcol(self.transform, geom.x, geom.y)
            half = window_size // 2
            window = Window(col - half, row - half, window_size, window_size)
            window = window.intersection(Window(0, 0, self.width, self.height))
            covered = None
        else:
            with rasterio.open(self.paths[0]) as reference:
                window = geometry_window(reference, [mapping(geom)])
                window = window.intersection(Window(0, 0, self.width, self.height))
                local_transform = reference.window_transform(window)
            covered = ~geometry_mask(
                [mapping(geom)],
                out_shape=(int(window.height), int(window.width)),
                transform=local_transform,
                invert=False,
            )

        stack, valid = self.read_window(window)
        if covered is not None:
            valid &= covered
        local_rows, local_cols = np.nonzero(valid)
        if len(local_rows) == 0:
            raise ValueError(f"Geometry {bag_id!r} contains no all-band-valid raster cells")
        samples = stack[:, local_rows, local_cols].T
        global_rows: np.ndarray = local_rows + int(window.row_off)
        global_cols: np.ndarray = local_cols + int(window.col_off)
        xs, ys = xy(self.transform, global_rows, global_cols, offset="center")
        return Bag(
            samples=jnp.asarray(samples),
            label=label,
            id=bag_id,
            coordinates=jnp.asarray(np.column_stack([xs, ys])),
            group_id=group_id,
            stratum_id=stratum_id,
            metadata={
                "adapter": "raster",
                "feature_names": list(cast(Sequence[str], self.band_names)),
                "crs": self.crs,
                "cell_indices": np.column_stack([global_rows, global_cols]).tolist(),
            },
        )

    def sample_valid_anchors(
        self, n: int, seed: int = 42, candidate_multiplier: int = 8
    ) -> List[CellIndex]:
        """Uniformly sample valid cells using blockwise random-priority selection."""
        if n < 1:
            return []
        keep = max(n * candidate_multiplier, n)
        rng = np.random.default_rng(seed)
        best_keys: np.ndarray = np.empty(0, dtype=float)
        best_rows: np.ndarray = np.empty(0, dtype=int)
        best_cols: np.ndarray = np.empty(0, dtype=int)
        for window in self.iter_windows():
            valid = self.valid_mask(window)
            rows, cols = np.nonzero(valid)
            if len(rows) == 0:
                continue
            rows = rows + int(window.row_off)
            cols = cols + int(window.col_off)
            keys = rng.random(len(rows))
            all_keys = np.concatenate([best_keys, keys])
            all_rows = np.concatenate([best_rows, rows])
            all_cols = np.concatenate([best_cols, cols])
            if len(all_keys) > keep:
                selected = np.argpartition(all_keys, keep - 1)[:keep]
                best_keys, best_rows, best_cols = (
                    all_keys[selected],
                    all_rows[selected],
                    all_cols[selected],
                )
            else:
                best_keys, best_rows, best_cols = all_keys, all_rows, all_cols
        order = np.argsort(best_keys)
        return [(int(best_rows[i]), int(best_cols[i])) for i in order]


def align_bags_to_raster(
    dataset: BagDataset,
    source: RasterSource,
    min_cells: int = 3,
    cap_cells: Optional[int] = None,
    seed: int = 42,
) -> BagDataset:
    """Apply the all-band raster mask and replace CSV values with raster cell values."""
    source_names = list(cast(Sequence[str], source.band_names))
    if list(dataset.feature_names) != source_names:
        raise ValueError("Tabular feature order must match RasterSource band order")
    if dataset.crs is not None and source.crs is not None and dataset.crs != source.crs:
        raise ValueError("Tabular and raster CRS differ")
    retained = []
    exclusions = []
    rng = np.random.default_rng(seed)
    for bag in dataset.collections:
        if bag.coordinates is None:
            raise ValueError(f"Bag {bag.id!r} has no coordinates for raster alignment")
        values, valid, cells = source.sample_points(np.asarray(bag.coordinates))
        valid_indices = np.flatnonzero(valid)
        # Several input coordinates can map to one raster cell; retain one value per cell.
        unique_indices = []
        seen: Set[CellIndex] = set()
        for index in valid_indices:
            cell = (int(cells[index, 0]), int(cells[index, 1]))
            if cell not in seen:
                seen.add(cell)
                unique_indices.append(int(index))
        if cap_cells is not None and len(unique_indices) > cap_cells:
            unique_indices = sorted(
                rng.choice(unique_indices, size=cap_cells, replace=False).tolist()
            )
        if len(unique_indices) < min_cells:
            exclusions.append(
                {
                    "id": bag.id,
                    "reason": "fewer_than_min_mask_consistent_cells",
                    "valid_unique_cells": len(unique_indices),
                }
            )
            continue
        chosen: NDArray[np.int_] = np.asarray(unique_indices, dtype=int)
        metadata = dict(bag.metadata or {})
        metadata.update(
            {
                "adapter": "tabular_raster_aligned",
                "raster_invalid_or_duplicate_removed": bag.n_samples - len(chosen),
                "cell_indices": cells[chosen].tolist(),
                "feature_names": source_names,
                "crs": source.crs,
            }
        )
        retained.append(
            Bag(
                jnp.asarray(values[chosen]),
                bag.label,
                bag.id,
                metadata=metadata,
                coordinates=jnp.asarray(np.asarray(bag.coordinates)[chosen]),
                group_id=bag.group_id,
                stratum_id=bag.stratum_id,
            )
        )
    if not retained:
        raise ValueError("No bags remain after raster alignment")
    metadata = dict(dataset.metadata or {})
    metadata["raster_alignment_exclusions"] = exclusions
    return BagDataset(
        retained,
        source_names,
        crs=source.crs,
        study_design=dataset.study_design,
        metadata=metadata,
    )


def build_spatial_background_bags(
    source: RasterSource,
    site_bags: Sequence[Bag],
    n_background: Optional[int] = None,
    cap_cells: int = 120,
    seed: int = 42,
    stratum_id: Optional[str] = None,
    match_sizes_exactly: bool = False,
) -> List[Bag]:
    """Build spatial bags around uniform valid anchors, matching site bag sizes.

    By default target sizes are a bootstrap sample from the retained site-bag
    sizes.  ``match_sizes_exactly=True`` uses a permutation when the class counts
    are equal, which is useful as a finite-sample sensitivity control without
    changing the established default background design.
    """
    if not site_bags:
        raise ValueError("site_bags must be nonempty")
    n_background = n_background or len(site_bags)
    rng = np.random.default_rng(seed)
    sizes: NDArray[np.int_] = np.asarray(
        [min(bag.n_samples, cap_cells) for bag in site_bags], dtype=int
    )
    if match_sizes_exactly:
        if n_background != len(site_bags):
            raise ValueError(
                "Exact size matching requires n_background to equal the number of site bags"
            )
        template_indices = rng.permutation(len(site_bags))
    else:
        template_indices = rng.choice(len(site_bags), n_background, replace=True)
    target_sizes = sizes[template_indices]
    occupied: Set[CellIndex] = set()
    for bag in site_bags:
        for row_value, col_value in (bag.metadata or {}).get("cell_indices", []):
            occupied.add((int(row_value), int(col_value)))

    anchors = source.sample_valid_anchors(n_background, seed=seed, candidate_multiplier=25)
    source_names = list(cast(Sequence[str], source.band_names))
    backgrounds: List[Bag] = []
    for anchor_row, anchor_col in anchors:
        if len(backgrounds) >= n_background:
            break
        if (anchor_row, anchor_col) in occupied:
            continue
        target = int(target_sizes[len(backgrounds)])
        selected = None
        for radius_multiplier in (1.5, 2.5, 4.0, 7.0):
            radius = max(1, int(np.ceil(np.sqrt(target) * radius_multiplier / 2)))
            window = Window(
                max(0, anchor_col - radius),
                max(0, anchor_row - radius),
                min(source.width, anchor_col + radius + 1) - max(0, anchor_col - radius),
                min(source.height, anchor_row + radius + 1) - max(0, anchor_row - radius),
            )
            stack, valid = source.read_window(window)
            local_rows, local_cols = np.nonzero(valid)
            global_rows: np.ndarray = local_rows + int(window.row_off)
            global_cols: np.ndarray = local_cols + int(window.col_off)
            allowed = np.asarray(
                [(int(row), int(col)) not in occupied for row, col in zip(global_rows, global_cols)]
            )
            local_rows, local_cols = local_rows[allowed], local_cols[allowed]
            global_rows, global_cols = global_rows[allowed], global_cols[allowed]
            if len(global_rows) < target:
                continue
            distance = (global_rows - anchor_row) ** 2 + (global_cols - anchor_col) ** 2
            nearest = np.argsort(distance)[:target]
            selected = (
                stack[:, local_rows[nearest], local_cols[nearest]].T,
                global_rows[nearest],
                global_cols[nearest],
            )
            break
        if selected is None:
            continue
        samples, rows, cols = selected
        xs, ys = xy(source.transform, rows, cols, offset="center")
        backgrounds.append(
            Bag(
                jnp.asarray(samples),
                0,
                f"background-{len(backgrounds):05d}",
                metadata={
                    "adapter": "raster_background",
                    "anchor_cell": [anchor_row, anchor_col],
                    "cell_indices": np.column_stack([rows, cols]).tolist(),
                    "size_template_id": site_bags[int(template_indices[len(backgrounds)])].id,
                    "size_matching": "exact_permutation" if match_sizes_exactly else "bootstrap",
                    "feature_names": source_names,
                    "crs": source.crs,
                },
                coordinates=jnp.asarray(np.column_stack([xs, ys])),
                group_id=f"background-{len(backgrounds):05d}",
                stratum_id=stratum_id,
            )
        )
    if len(backgrounds) != n_background:
        raise RuntimeError(
            f"Could construct only {len(backgrounds)} of {n_background} background bags"
        )
    return backgrounds
