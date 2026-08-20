"""Final validation of the uncertainty architecture before it is frozen.

    uv run python -m apexstrategy.evaluation.final_report

``calibration_report`` answers "is the race-time interval right". This answers the wider
question the freeze requires: right *where*, wrong *how*, and is the wrongness something a
variance model should be fixing at all.

## Two levels, because a compound has no race time

Race-time calibration is one number per driver per replay, so it can be cut by season,
race and circuit and no further. Tyre compound and stint age vary *within* a driver's race,
so they need the per-lap predictive distribution instead. Both are computed here from the
same simulations; they answer different questions and neither substitutes for the other.

## Four diagnoses, kept apart on purpose

Under-coverage has causes with opposite fixes, and a pooled number hides which:

- **width** -- the intervals are the wrong size. PIT spread departs from uniform while its
  centre does not. This is the only failure a variance model should fix.
- **mean bias** -- the point estimate is off. PIT centre departs from 0.5. Widening
  intervals here buys coverage by hiding an error, which is worse than reporting it.
- **regime shift** -- mean bias whose cause is that the process changed, evidenced from
  outside the residuals. Not inferable from PIT alone; this module flags severe bias and
  names the evidence where it has it.
- **irreducible** -- PIT already uniform. Nothing left to model.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .. import config
from .replay import COVERAGE_LEVELS, DEFAULT_FROM_LAPS, ReplayResult, replay_all
from .season_holdout import holdout_all

log = logging.getLogger(__name__)

UNIFORM_SD = float(np.sqrt(1 / 12))

#: How far PIT centre may sit from 0.5, and spread from 0.289, before it is called a
#: failure. With ~80 outcomes per cell the standard error on a PIT mean is about 0.03, so a
#: tighter band would be reporting noise as a finding.
BIAS_TOLERANCE = 0.08
WIDTH_TOLERANCE = 0.03
#: Beyond this the distribution has moved, not merely widened, and PIT spread stops
#: describing interval width at all.
SEVERE_BIAS = 0.25


def lap_calibration(result: ReplayResult) -> pd.DataFrame:
    """Per-lap predictive calibration, so compound and stint age can be cut.

    The simulator already carries the whole ``(n_sims, n_drivers, n_laps)`` sample, so this
    is a lookup rather than a re-run: for every lap a driver actually completed, where did
    the real lap time fall inside the draws for that lap.
    """
    sim = result.simulation
    frame = result.laps
    if frame.empty:
        return pd.DataFrame()

    driver_index = {d: i for i, d in enumerate(sim.drivers)}
    rows = frame[frame["driver"].isin(driver_index)].copy()
    di = rows["driver"].map(driver_index).to_numpy()
    li = rows["lap_number"].to_numpy() - sim.from_lap - 1
    keep = (li >= 0) & (li < sim.lap_times_s.shape[2])
    rows, di, li = rows[keep], di[keep], li[keep]

    draws = sim.lap_times_s[:, di, li]  # (n_sims, n_rows)
    actual = rows["actual_s"].to_numpy()[None, :]
    out = rows.assign(pit=(draws <= actual).mean(axis=0))
    for level in COVERAGE_LEVELS:
        lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        out[f"covered_{int(level * 100)}"] = (lo <= actual[0]) & (actual[0] <= hi)
    return out


def summarise(frame: pd.DataFrame, by: list[str] | None, min_n: int = 30) -> pd.DataFrame:
    """Coverage, PIT and a diagnosis per cell."""
    cols = [f"covered_{int(x * 100)}" for x in COVERAGE_LEVELS]
    grouped = frame.groupby(by) if by else frame.assign(_all="pooled").groupby("_all")
    out = grouped[cols].mean()
    out.columns = [f"cov_{int(x * 100)}" for x in COVERAGE_LEVELS]
    out["pit_mean"] = grouped["pit"].mean()
    out["pit_sd"] = grouped["pit"].std()
    out["n"] = grouped.size()
    out = out[out["n"] >= min_n]
    out["diagnosis"] = [classify(r) for _, r in out.iterrows()]
    return out


def classify(row: pd.Series) -> str:
    """Which of the four causes this cell is, in the order that matters.

    Bias is tested first: once the truth lands almost entirely on one side, PIT spread
    describes how tightly the errors bunch *outside* the interval and says nothing useful
    about width, so a width verdict read from it would be actively misleading.
    """
    bias = row["pit_mean"] - 0.5
    if abs(bias) >= SEVERE_BIAS:
        return "regime/severe bias " + ("(optimistic)" if bias > 0 else "(pessimistic)")
    if abs(bias) > BIAS_TOLERANCE:
        return "mean bias " + ("(optimistic)" if bias > 0 else "(pessimistic)")
    if row["pit_sd"] < UNIFORM_SD - WIDTH_TOLERANCE:
        return "width (too wide)"
    if row["pit_sd"] > UNIFORM_SD + WIDTH_TOLERANCE:
        return "width (too narrow)"
    return "calibrated (irreducible)"


def _race_frame(results: list[ReplayResult], split: str) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.concat(
        [r.races.assign(circuit=r.circuit, from_lap=r.from_lap, split=split) for r in results],
        ignore_index=True,
    )


def _lap_frame(results: list[ReplayResult], split: str) -> pd.DataFrame:
    frames = [
        lap_calibration(r).assign(circuit=r.circuit, from_lap=r.from_lap, split=split)
        for r in results
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.round(3).to_string() if not table.empty else "  (no cells above the minimum)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-laps", nargs="*", type=int, default=list(DEFAULT_FROM_LAPS))
    parser.add_argument("--n-sims", type=int, default=400)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    laps = tuple(args.from_laps)

    _, seen = replay_all(None, laps, args.n_sims)
    _, unseen = holdout_all(None, laps, args.n_sims)

    races = pd.concat([_race_frame(seen, "seen"), _race_frame(unseen, "unseen")])
    seen_label = "-".join(str(x)[-2:] for x in config.TRAINING_SEASONS)
    races["season"] = np.where(
        races["split"] == "unseen", str(config.HOLDOUT_SEASONS[0]), seen_label
    )
    per_lap = pd.concat([_lap_frame(seen, "seen"), _lap_frame(unseen, "unseen")])
    clean = per_lap[per_lap["category"] == "clean"]

    print("MODEL EVALUATION -- FINAL VALIDATION")
    print(
        f"nominal coverage {' '.join(f'{int(x * 100)}%' for x in COVERAGE_LEVELS)}"
        f" | PIT mean 0.500, sd {UNIFORM_SD:.3f}"
    )
    print(f"training seasons {config.TRAINING_SEASONS}, held out {config.HOLDOUT_SEASONS}")

    print("\n=== point prediction (not being optimised; reported for completeness) ===")
    mae = clean.groupby("split")["error_s"].agg(mae=lambda s: s.abs().mean(), bias="mean", n="size")
    print(mae.round(3).to_string())

    _show("RACE-TIME calibration, pooled by split", summarise(races, ["split"]))
    _show("by season", summarise(races, ["season"]))
    _show("by circuit, unseen season", summarise(races[races.split == "unseen"], ["circuit"], 20))
    _show("by decision lap", summarise(races, ["split", "from_lap"], 20))

    print("\n=== LAP-LEVEL calibration (clean laps only) ===")
    _show("pooled by split", summarise(clean, ["split"], 200))
    _show("by tyre compound", summarise(clean, ["split", "compound"], 200))
    last_lap = clean.groupby(["split", "circuit"])["lap_number"].transform("max")
    clean = clean.assign(
        stint_age=pd.cut(clean["tyre_age"], [0, 5, 10, 20, 30, 100]).astype(str),
        race_progress=pd.cut(
            clean["lap_number"] / last_lap,
            [0, 0.33, 0.66, 1.01],
            labels=["early", "middle", "late"],
        ).astype(str),
    )
    _show("by stint age (laps on the set)", summarise(clean, ["split", "stint_age"], 200))
    _show("by race progress", summarise(clean, ["split", "race_progress"], 200))

    per_race = summarise(races, ["split", "circuit", "from_lap"], 15).sort_values("cov_80")
    _show("WORST UNDER-COVERED (race-time)", per_race.head(6))
    _show("WORST OVER-COVERED (race-time)", per_race.tail(6).iloc[::-1])

    print("\n=== diagnosis mix, unseen season ===")
    mix = summarise(races[races.split == "unseen"], ["circuit", "from_lap"], 15)
    print(mix["diagnosis"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
