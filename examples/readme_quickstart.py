#!/usr/bin/env python3
"""Executable counterpart to the README quick-start workflow.

The default predicts a small raster preview so CI verifies the complete path
quickly. Pass ``--full-surface`` to predict the full 200 by 200 example extent.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Union

import geopandas as gpd
import numpy as np

from klrfome import KLRfome, ModelSpec, RasterStack


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = REPOSITORY_ROOT / "example_data"


def run_quickstart(full_surface: bool = False) -> Dict[str, Union[int, float, list]]:
    """Fit M1 on the bundled data and return a finite prediction summary."""
    raster_paths = [str(EXAMPLE_DATA / name) for name in ("var1.tif", "var2.tif", "var3.tif")]
    rasters = RasterStack.from_files(raster_paths)
    sites = gpd.read_file(EXAMPLE_DATA / "sites.geojson")

    specification = ModelSpec.m1(rff_features=256)
    model = KLRfome(
        spec=specification,
        lambda_reg=0.1,
        window_size=5,
        seed=42,
    )
    training = model.prepare_data(
        raster_stack=rasters,
        sites=sites,
        n_background=25,
        samples_per_location=20,
    )
    model.fit(training)

    prediction_rasters = rasters
    if not full_surface:
        preview_size = 24
        prediction_rasters = RasterStack(
            data=rasters.data[:, :preview_size, :preview_size],
            transform=rasters.transform,
            crs=rasters.crs,
            band_names=list(rasters.band_names),
            nodata=rasters.nodata,
        )

    scores = np.asarray(
        model.predict(prediction_rasters, batch_size=512, show_progress=full_surface)
    )
    if not np.isfinite(scores).all():
        raise RuntimeError("README quick start produced nonfinite relative-suitability scores")

    return {
        "method": specification.method_id,
        "n_site_bags": training.n_sites,
        "n_background_bags": training.n_background,
        "prediction_shape": list(scores.shape),
        "minimum_relative_suitability": float(scores.min()),
        "maximum_relative_suitability": float(scores.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-surface",
        action="store_true",
        help="Predict the complete example raster instead of the CI-sized preview.",
    )
    args = parser.parse_args()
    print(json.dumps(run_quickstart(full_surface=args.full_surface), indent=2))


if __name__ == "__main__":
    main()
