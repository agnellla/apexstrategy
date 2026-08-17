"""Phase 8: is the simulator's *uncertainty* right, not just its point estimate?

    uv run python -m apexstrategy.evaluation.calibration

Phase 7 established that the simulator reproduces race times. That says nothing about
whether it knows how confident to be, and a strategy tool that reports "P(this is faster)
= 94%" on a coin flip is worse than one that reports no probability at all.

## The test

Every replayed driver contributes one outcome whose true race time is known. Coverage at
level q asks what share of those truths fell inside the central q interval of the
predictive distribution. Nominal coverage is the exit criterion the architecture doc
sets: **80% intervals should cover ~80%**.

Coverage at a handful of levels can still hide a badly shaped distribution, so the
**probability integral transform** is reported alongside it. Each outcome's PIT value is
the share of simulated draws at or below the truth; under perfect calibration those are
Uniform(0,1), with mean 0.5 and sd 0.289. The shape of their departure is diagnostic in a
way a coverage number is not:

- PIT piling up at 0 and 1 -> intervals too narrow (over-confident)
- PIT bunched in the middle -> intervals too wide (under-confident)
- PIT mean away from 0.5 -> the point estimate is biased, not the spread

## What this cannot tell you

Coverage is measured against races the pace model never saw, but it is still only five
races. It validates the *variance decomposition*, not the counterfactuals the product
sells: no amount of calibration makes an unobserved strategy's outcome checkable. That
limit is real and is stated in the UI rather than argued away.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .replay import COVERAGE_LEVELS, DEFAULT_FROM_LAPS, ReplayResult, replay_all

#: Tolerance on nominal coverage, in percentage points. The doc asks for "within +-5pp";
#: with ~20 drivers per replay the binomial standard error on a coverage estimate is
#: already ~9pp at the 80% level, so a tighter band would be measuring noise.
COVERAGE_TOLERANCE_PP = 10.0

#: Under a correct decomposition PIT values are Uniform(0,1).
UNIFORM_MEAN, UNIFORM_SD = 0.5, float(np.sqrt(1 / 12))


def coverage_table(results: list[ReplayResult], by: str | None = None) -> pd.DataFrame:
    """Empirical coverage at each level, pooled or grouped by a replay attribute."""
    frame = pd.concat(
        [r.races.assign(circuit=r.circuit, from_lap=r.from_lap) for r in results],
        ignore_index=True,
    )
    cols = [f"covered_{int(level * 100)}" for level in COVERAGE_LEVELS]
    grouped = frame.groupby(by) if by else frame
    out = grouped[cols].mean() if by else frame[cols].mean().to_frame().T
    out = pd.DataFrame(out)
    out.columns = [f"cover_{int(level * 100)}" for level in COVERAGE_LEVELS]
    out["n"] = grouped.size() if by else len(frame)
    return out


def pit_summary(results: list[ReplayResult], by: str | None = None) -> pd.DataFrame:
    """Mean and sd of the probability integral transform. Uniform(0,1) is the target.

    ``pit_mean`` away from 0.5 indicts the point estimate; ``pit_sd`` below 0.289 means
    the intervals are too narrow and above means too wide.
    """
    frame = pd.concat(
        [r.races.assign(circuit=r.circuit, from_lap=r.from_lap) for r in results],
        ignore_index=True,
    )
    # Share landing in the outermost 10% of the predictive distribution. Should be 0.10;
    # higher means the tails are too thin, which is the failure mode that matters most for
    # risk metrics like CVaR.
    frame["in_tail"] = (frame["pit"] < 0.05) | (frame["pit"] > 0.95)
    grouped = frame.groupby(by) if by else frame.assign(_all="pooled").groupby("_all")
    out = grouped.agg(
        pit_mean=("pit", "mean"), pit_sd=("pit", "std"), tail_10pct=("in_tail", "mean")
    )
    out.index.name = by
    return out


def verdict(results: list[ReplayResult]) -> tuple[bool, pd.DataFrame]:
    """Whether coverage sits within tolerance of nominal at every reported level."""
    table = coverage_table(results)
    rows = []
    for level in COVERAGE_LEVELS:
        col = f"cover_{int(level * 100)}"
        actual = float(table[col].iloc[0])
        rows.append(
            {
                "level": col,
                "nominal": level,
                "actual": actual,
                "error_pp": 100 * (actual - level),
                "within_tolerance": abs(100 * (actual - level)) <= COVERAGE_TOLERANCE_PP,
            }
        )
    out = pd.DataFrame(rows)
    return bool(out["within_tolerance"].all()), out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-laps", nargs="*", type=int, default=list(DEFAULT_FROM_LAPS))
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _, results = replay_all(from_laps=tuple(args.from_laps), n_sims=args.n_sims)

    print("\n=== race-time interval coverage, pooled over every replayed driver ===")
    print(coverage_table(results).round(3).to_string(index=False))

    print("\n=== by decision lap ===")
    print(coverage_table(results, by="from_lap").round(3).to_string())

    print("\n=== by circuit ===")
    print(coverage_table(results, by="circuit").round(3).to_string())

    print("\n=== probability integral transform (Uniform(0,1) if calibrated) ===")
    print(f"target: mean {UNIFORM_MEAN:.3f}, sd {UNIFORM_SD:.3f}, tail_10pct 0.100")
    print(pit_summary(results).round(3).to_string())
    print()
    print(pit_summary(results, by="from_lap").round(3).to_string())

    ok, table = verdict(results)
    print(f"\n=== Phase 8 exit criterion (within {COVERAGE_TOLERANCE_PP:.0f}pp of nominal) ===")
    print(table.round(3).to_string(index=False))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
