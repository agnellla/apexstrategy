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


def training_laps(processed_dir: Path | None = None) -> pd.DataFrame:
    """Clean laps from training circuits only.

    The single entry point models should use. Monaco is excluded here rather than at
    each call site so that D3's exclusion cannot be forgotten by a new model.
    """
    laps = read_table("laps", processed_dir)
    allowed = {c.key for c in config.TRAINING_CIRCUITS}
    return laps[laps["is_clean"] & laps["circuit"].isin(allowed)].reset_index(drop=True)
