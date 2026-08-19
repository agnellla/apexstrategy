"""Coverage and PIT for the simulator, pooled and broken down, across seasons.

    uv run python -m apexstrategy.evaluation.calibration_report

``evaluation.calibration`` answers this within the training season. This answers it the way
the acceptance criterion is written: on races and seasons the model has never seen, broken
down far enough to tell an interval that is too narrow from a point estimate that is
biased, because those look identical in a pooled coverage number and have opposite fixes.

**Read the PIT before concluding anything about width.** Coverage below nominal has two
causes. If the probability integral transform piles up at both ends, the intervals are too
narrow and more variance is the answer. If it sits off-centre, the point estimate is
biased, the spread is innocent, and widening the intervals to cover the bias would be
hiding an error rather than reporting one.
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


def _frame(results: list[ReplayResult], label: str) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.concat(
        [r.races.assign(circuit=r.circuit, from_lap=r.from_lap, split=label) for r in results],
        ignore_index=True,
    )


def coverage(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    cols = [f"covered_{int(x * 100)}" for x in COVERAGE_LEVELS]
    grouped = frame.groupby(by) if by else frame.assign(_all="pooled").groupby("_all")
    out = grouped[cols].mean()
    out.columns = [f"cover_{int(x * 100)}" for x in COVERAGE_LEVELS]
    out["pit_mean"] = grouped["pit"].mean()
    out["pit_sd"] = grouped["pit"].std()
    out["tail"] = grouped["pit"].apply(lambda s: ((s < 0.05) | (s > 0.95)).mean())
    out["n"] = grouped.size()
    return out


def diagnose(row: pd.Series) -> str:
    """Name the failure mode rather than printing three numbers and leaving it to the reader.

    Bias is checked first and, when it is severe, reported alone. Once the truth is landing
    almost entirely on one side of the predictive distribution, the spread of the PIT stops
    describing interval width and starts describing how tightly the errors are bunched
    outside the interval, and quoting a width verdict from it is worse than saying nothing.

    On width: PIT bunched in the middle (sd below the uniform 0.289) means the intervals
    are wider than the errors they cover; PIT piled at both ends means they are too narrow.
    Reading that backwards inverts the prescription, so it is spelled out rather than left
    as a bare comparison.
    """
    if row["pit_mean"] > 0.75 or row["pit_mean"] < 0.25:
        side = "slower" if row["pit_mean"] > 0.5 else "faster"
        return (
            f"badly biased: the truth lands {side} than the projection almost every time "
            f"(PIT mean {row['pit_mean']:.2f}), so width is not the problem"
        )

    width = (
        "too wide"
        if row["pit_sd"] < UNIFORM_SD - 0.03
        else ("too narrow" if row["pit_sd"] > UNIFORM_SD + 0.03 else "about right")
    )
    bias = (
        "projections optimistic"
        if row["pit_mean"] > 0.58
        else "projections pessimistic"
        if row["pit_mean"] < 0.42
        else "unbiased"
    )
    return f"intervals {width}, point estimate {bias}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", nargs="*", default=None)
    parser.add_argument("--from-laps", nargs="*", type=int, default=list(DEFAULT_FROM_LAPS))
    parser.add_argument("--n-sims", type=int, default=400)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    circuits = tuple(args.circuits) if args.circuits else None
    laps = tuple(args.from_laps)

    _, seen = replay_all(circuits, laps, args.n_sims)
    _, unseen = holdout_all(circuits, laps, args.n_sims)
    frame = pd.concat([_frame(seen, "in-season LORO"), _frame(unseen, "held-out season")])

    print(
        f"nominal: {' '.join(f'{int(x * 100)}%' for x in COVERAGE_LEVELS)}"
        f"   PIT mean 0.500, sd {UNIFORM_SD:.3f}, tail 0.100\n"
    )

    print("=== pooled, by split ===")
    pooled = coverage(frame, ["split"])
    print(pooled.round(3).to_string())
    for label, row in pooled.iterrows():
        print(f"  {label}: {diagnose(row)}")

    print("\n=== by decision lap ===")
    print(coverage(frame, ["split", "from_lap"]).round(3).to_string())

    print("\n=== by circuit, held-out season ===")
    held = frame[frame["split"] == "held-out season"]
    if not held.empty:
        by_circuit = coverage(held, ["circuit"]).sort_values("cover_80")
        print(by_circuit.round(3).to_string())
        for label, row in by_circuit.iterrows():
            print(f"  {label}: {diagnose(row)}")

    print(f"\ntraining seasons {config.TRAINING_SEASONS}, held out {config.HOLDOUT_SEASONS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
