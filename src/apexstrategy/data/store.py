"""Parquet store. Every table is validated against its contract on write.

Validating on write rather than read means a broken build fails at the point the bad
data was produced, where the traceback still points at the cause.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .. import config
from .schema import TABLES_BY_NAME, TableSpec

log = logging.getLogger(__name__)


def table_path(name: str, processed_dir: Path | None = None) -> Path:
    return (processed_dir or config.PROCESSED_DIR) / f"{name}.parquet"


def write_table(
    name: str, df: pd.DataFrame, processed_dir: Path | None = None, validate: bool = True
) -> Path:
    """Validate against the table's contract, then write.

    Columns are reordered to the spec so the on-disk schema is stable across rebuilds.
    """
    spec: TableSpec = TABLES_BY_NAME[name]
    df = df.reindex(columns=list(spec.column_names))
    if validate:
        spec.validate_frame(df)

    path = table_path(name, processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s rows=%d -> %s", name, len(df), path)
    return path


def read_table(name: str, processed_dir: Path | None = None) -> pd.DataFrame:
    path = table_path(name, processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"{name} not built yet: {path}. Run `apex-build`.")
    return pd.read_parquet(path)


def read_all(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {name: read_table(name, processed_dir) for name in TABLES_BY_NAME}


def race_id_for(circuit: str, season: int | None = None) -> str:
    return f"{season or config.SEASON}_{circuit}_{config.SESSION}"


def race_laps(
    circuit: str, processed_dir: Path | None = None, season: int | None = None
) -> pd.DataFrame:
    """Every lap of one race, clean or not.

    The simulator needs the excluded laps too: a pit-in lap still happened and still
    consumed race time. ``training_laps`` is the model-fitting view; this is the
    simulation view.
    """
    laps = read_table("laps", processed_dir)
    out = laps[(laps["circuit"] == circuit) & (laps["season"] == (season or config.SEASON))]
    if out.empty:
        raise KeyError(f"no laps stored for {circuit!r} in {season or config.SEASON}")
    return out.reset_index(drop=True)


def race_total_laps(
    circuit: str, processed_dir: Path | None = None, season: int | None = None
) -> int:
    """Scheduled race distance. A calendar fact, known before the race starts."""
    results = read_table("results", processed_dir)
    match = (results["circuit"] == circuit) & (results["season"] == (season or config.SEASON))
    return int(results.loc[match, "total_laps"].max())


def training_laps(processed_dir: Path | None = None) -> pd.DataFrame:
    """Clean laps from training circuits **in the training season** only.

    The single entry point models should use. Monaco is excluded here rather than at each
    call site so that D3's exclusion cannot be forgotten by a new model, and the season
    filter is here for the same reason: once a held-out season is in the store, a model
    that reads laps directly would train on it and report a flattering out-of-sample
    number. Both exclusions are structural, not remembered.
    """
    return trainable_laps(read_table("laps", processed_dir))


def trainable_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Apply the training-pool filters to a laps frame that is already in hand."""
    allowed = {c.key for c in config.TRAINING_CIRCUITS}
    return laps[
        laps["is_clean"]
        & laps["circuit"].isin(allowed)
        & (laps["season"] == config.TRAINING_SEASON)
    ].reset_index(drop=True)


def fittable_laps(processed_dir: Path | None = None) -> pd.DataFrame:
    """Every lap of the training season, clean or not.

    What the simulator's constants are fitted from: pit losses need the pit laps that
    ``training_laps`` drops. Season-gated for the same reason.
    """
    laps = read_table("laps", processed_dir)
    return laps[laps["season"] == config.TRAINING_SEASON].reset_index(drop=True)
