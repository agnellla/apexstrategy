"""Phase 1 entry point: rebuild every processed table from FastF1.

    uv run python -m apexstrategy.data.build

Races not already cached are fetched on first run (~30-120s each). Downstream modules
read only from the Parquet store -- no model should reach back into FastF1 itself.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .. import config
from . import store
from .clean import yield_summary
from .gaps import DIRTY_AIR_THRESHOLD_S, dirty_air_sensitivity, position_agreement
from .load import load_session
from .normalize import normalize_session

log = logging.getLogger(__name__)

TABLE_NAMES = ("laps", "stints", "pits", "track_status", "weather", "results")


def build(circuit_keys: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Ingest, normalise, validate, and write all tables."""
    config.ensure_dirs()
    circuits = (
        tuple(config.CIRCUITS_BY_KEY[k] for k in circuit_keys) if circuit_keys else config.CIRCUITS
    )

    collected: dict[str, list[pd.DataFrame]] = {n: [] for n in TABLE_NAMES}
    for circuit in circuits:
        log.info("processing %s", circuit.key)
        tables = normalize_session(load_session(circuit))
        for name, df in tables.items():
            collected[name].append(df)

    out: dict[str, pd.DataFrame] = {}
    for name, frames in collected.items():
        merged = pd.concat(frames, ignore_index=True)
        store.write_table(name, merged)
        out[name] = merged
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuits",
        nargs="*",
        choices=sorted(config.CIRCUITS_BY_KEY),
        help="subset to build (default: all configured circuits)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tables = build(tuple(args.circuits) if args.circuits else None)

    laps = tables["laps"]
    print("\n=== clean-lap yield ===")
    print(yield_summary(laps).to_string())
    print(f"\noverall clean: {100 * laps['is_clean'].mean():.1f}%  ({len(laps)} laps)")

    print("\n=== reconstructed vs reported position ===")
    print(position_agreement(laps).to_string())

    print(f"\n=== dirty-air threshold sensitivity (using {DIRTY_AIR_THRESHOLD_S}s) ===")
    print(dirty_air_sensitivity(laps).to_string(index=False))

    stints = tables["stints"]
    print("\n=== stint end reasons ===")
    print(stints["end_reason"].value_counts().to_string())
    print(f"censored: {stints['is_censored'].sum()}/{len(stints)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
