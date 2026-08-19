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
from .load import discover_races, load_race, load_session
from .normalize import normalize_session

log = logging.getLogger(__name__)

TABLE_NAMES = ("laps", "stints", "pits", "track_status", "weather", "results")


def build(
    circuit_keys: tuple[str, ...] | None = None,
    seasons: tuple[int, ...] = (config.SEASON,),
    all_circuits: bool = False,
) -> dict[str, pd.DataFrame]:
    """Ingest, normalise, validate, and write all tables.

    Multi-season because a held-out season is the only test that asks the model to
    generalise across cars, tyres and regulations rather than merely across circuits.
    Season is part of ``race_id`` and of every table, and everything model-facing filters
    to ``config.TRAINING_SEASON``, so a season in the store is a season the models cannot
    see unless something deliberately asks for it.

    Note that this rewrites each table wholesale, so a partial build drops what it does not
    rebuild. Pass every season you want to keep.
    """
    config.ensure_dirs()
    circuits = (
        tuple(config.CIRCUITS_BY_KEY[k] for k in circuit_keys) if circuit_keys else config.CIRCUITS
    )

    collected: dict[str, list[pd.DataFrame]] = {n: [] for n in TABLE_NAMES}
    for season in seasons:
        if all_circuits:
            refs = discover_races(season)
            log.info("%s: %d races on the calendar", season, len(refs))
            for i, ref in enumerate(refs, 1):
                log.info("processing %s %s (%d/%d)", season, ref.key, i, len(refs))
                try:
                    tables = normalize_session(load_race(ref))
                except Exception as exc:
                    # A race can be missing, incomplete, or shaped in a way the contracts
                    # reject. Skipping loudly beats failing a 70-race ingest on one bad row.
                    log.warning("skipping %s %s: %s", season, ref.key, exc)
                    continue
                for name, df in tables.items():
                    collected[name].append(df)
            continue

        for circuit in circuits:
            log.info("processing %s %s", season, circuit.key)
            try:
                tables = normalize_session(load_session(circuit, season=season))
            except Exception as exc:  # a circuit may simply not be on that year's calendar
                log.warning("skipping %s %s: %s", season, circuit.key, exc)
                continue
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
    parser.add_argument(
        "--all-circuits",
        action="store_true",
        help="ingest every race on each season's calendar, discovered from the schedule",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=[config.SEASON],
        help="seasons to ingest; models only ever fit on config.TRAINING_SEASON",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tables = build(
        tuple(args.circuits) if args.circuits else None,
        tuple(args.seasons),
        all_circuits=args.all_circuits,
    )

    laps = tables["laps"]
    print("\n=== clean-lap yield ===")
    print(yield_summary(laps).to_string())
    print(f"\noverall clean: {100 * laps['is_clean'].mean():.1f}%  ({len(laps)} laps)")
    print("\n=== laps by season (only TRAINING_SEASON is ever fitted on) ===")
    print(laps.groupby(["season", "circuit"]).size().to_string())

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
