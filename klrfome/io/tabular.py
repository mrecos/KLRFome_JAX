"""Format-neutral tabular adapter for cell-level bag data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..data.formats import Bag, BagDataset


@dataclass(frozen=True)
class TabularBagConfig:
    """Explicit columns with conservative, case-insensitive autodetection."""

    feature_columns: Optional[Sequence[str]] = None
    label_column: Optional[str] = None
    id_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    group_column: Optional[str] = None
    stratum_column: Optional[str] = None
    crs: Optional[str] = None
    study_design: str = "presence_background"
    deduplicate: bool = True
    min_unique_cells: int = 3


def _detect_one(columns: Sequence[str], candidates: Sequence[str], role: str) -> str:
    lowered: Dict[str, List[str]] = {}
    for column in columns:
        lowered.setdefault(column.lower(), []).append(column)
    matches = []
    for candidate in candidates:
        matches.extend(lowered.get(candidate.lower(), []))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(
            f"Could not safely autodetect {role}; configure it explicitly " f"(matches: {matches})"
        )
    return matches[0]


def resolve_tabular_config(frame: pd.DataFrame, config: TabularBagConfig) -> TabularBagConfig:
    columns = list(frame.columns)
    label = config.label_column or _detect_one(
        columns, ("presence", "present", "label", "pa"), "label column"
    )
    identifier = config.id_column or _detect_one(
        columns, ("siteno", "site_no", "site_id", "bag_id"), "bag id column"
    )
    x_column = config.x_column or _detect_one(
        columns, ("x", "easting", "longitude", "lon"), "x column"
    )
    y_column = config.y_column or _detect_one(
        columns, ("y", "northing", "latitude", "lat"), "y column"
    )
    if config.feature_columns is None:
        excluded = {
            label,
            identifier,
            x_column,
            y_column,
            config.group_column,
            config.stratum_column,
        }
        features = [
            column
            for column in columns
            if column not in excluded
            and not column.lower().startswith("unnamed:")
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    else:
        features = list(config.feature_columns)
    missing = [
        column
        for column in [label, identifier, x_column, y_column, *features]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Configured columns are missing: {missing}")
    if not features:
        raise ValueError("No feature columns were configured or safely detected")
    return TabularBagConfig(
        feature_columns=tuple(features),
        label_column=label,
        id_column=identifier,
        x_column=x_column,
        y_column=y_column,
        group_column=config.group_column,
        stratum_column=config.stratum_column,
        crs=config.crs,
        study_design=config.study_design,
        deduplicate=config.deduplicate,
        min_unique_cells=config.min_unique_cells,
    )


def load_tabular_bags(
    source: Union[str, Path, pd.DataFrame],
    config: Optional[TabularBagConfig] = None,
    labels: Optional[Iterable[int]] = None,
) -> BagDataset:
    """Load grouped cells, deduplicate coordinates, and retain a complete audit."""
    frame = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    resolved = resolve_tabular_config(frame, config or TabularBagConfig())
    assert resolved.feature_columns is not None
    assert resolved.label_column is not None and resolved.id_column is not None
    assert resolved.x_column is not None and resolved.y_column is not None
    if labels is not None:
        allowed = {int(value) for value in labels}
        frame = frame[frame[resolved.label_column].isin(allowed)].copy()

    bags = []
    exclusions = []
    total_duplicates = 0
    grouping = [resolved.label_column, resolved.id_column]
    for (raw_label, raw_id), group in frame.groupby(grouping, sort=True, dropna=False):
        if pd.isna(raw_id):
            exclusions.append({"id": None, "reason": "missing_id", "rows": len(group)})
            continue
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            exclusions.append({"id": str(raw_id), "reason": "invalid_label", "rows": len(group)})
            continue
        if label not in (0, 1):
            exclusions.append({"id": str(raw_id), "reason": "invalid_label", "rows": len(group)})
            continue
        binary_label = cast(Literal[0, 1], label)

        needed = [*resolved.feature_columns, resolved.x_column, resolved.y_column]
        valid = np.isfinite(group[needed].to_numpy(dtype=float)).all(axis=1)
        clean = group.loc[valid].copy()
        invalid_count = int((~valid).sum())
        before_deduplication = len(clean)
        if resolved.deduplicate:
            clean = clean.drop_duplicates(
                subset=[resolved.id_column, resolved.x_column, resolved.y_column], keep="first"
            )
        duplicate_count = before_deduplication - len(clean)
        total_duplicates += duplicate_count
        if len(clean) < resolved.min_unique_cells:
            exclusions.append(
                {
                    "id": str(raw_id),
                    "reason": "fewer_than_min_unique_cells",
                    "rows": len(group),
                    "valid_unique_cells": len(clean),
                }
            )
            continue
        group_id = (
            str(clean.iloc[0][resolved.group_column])
            if resolved.group_column is not None
            else str(raw_id)
        )
        stratum_id = (
            str(clean.iloc[0][resolved.stratum_column])
            if resolved.stratum_column is not None
            else None
        )
        bags.append(
            Bag(
                samples=jnp.asarray(clean[list(resolved.feature_columns)].to_numpy(dtype=float)),
                label=binary_label,
                id=f"{'site' if label else 'background'}-{raw_id}",
                coordinates=jnp.asarray(
                    clean[[resolved.x_column, resolved.y_column]].to_numpy(dtype=float)
                ),
                group_id=group_id,
                stratum_id=stratum_id,
                metadata={
                    "source_id": str(raw_id),
                    "input_rows": len(group),
                    "invalid_rows": invalid_count,
                    "duplicates_removed": duplicate_count,
                    "feature_names": list(resolved.feature_columns),
                    "crs": resolved.crs,
                },
            )
        )
    if not bags:
        raise ValueError("No bags satisfy the tabular validation contract")
    return BagDataset(
        collections=bags,
        feature_names=list(resolved.feature_columns),
        crs=resolved.crs,
        study_design=resolved.study_design,  # type: ignore[arg-type]
        metadata={
            "adapter": "tabular",
            "input_rows": len(frame),
            "duplicates_removed": total_duplicates,
            "exclusions": exclusions,
            "resolved_columns": {
                "label": resolved.label_column,
                "id": resolved.id_column,
                "x": resolved.x_column,
                "y": resolved.y_column,
                "features": list(resolved.feature_columns),
            },
        },
    )
