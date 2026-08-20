"""Does a model fitted on 2024 predict 2025? The only test that asks the real question.

    uv run python -m apexstrategy.evaluation.season_holdout
    uv run python -m apexstrategy.evaluation.season_holdout --circuits hungary --from-laps 20

Leave-one-race-out inside a single season asks the model to generalise across *circuits*.
That is a real question and Phase 7 answered it. It is not the question anyone deploying
this would ask, which is whether a model fitted on last year's cars, tyres and regulations
says anything useful about this year's.

Between 2024 and 2025 the field changed, the tyre allocation changed and the cars changed.
Nothing in the pace model knows that. If the projections survive the jump, the degradation
and race-progress structure it learned is a property of racing rather than of one season.
If they do not, every number this project reports is a number about 2024, and the honest
move is to say so on the page rather than to quietly keep reporting it.

## What is held constant, and what is not

The pace model, the pit-loss sample, the noise components and the compound caps are fitted
on **2024 only**, enforced by ``store.fittable_laps`` rather than remembered here. The 2025
race supplies exactly what a live system would have at lap L: the strategies everyone
actually ran, the cautions that actually happened, and the scheduled distance. Same inputs
as the Phase 7 replay, one season later.

Circuits are matched by name, so this isolates *season* transfer at circuits the model has
already seen. A new circuit in a new season confounds the two and is a different question.

## Read it against the within-season number, not against zero

The Phase 7 gate is a median race-time error under 1% of the projected window, and 2024
posts roughly 0.4-0.6% from decision lap 15 onward. The interesting quantity here is not
whether 2025 clears 1%; it is how much worse it is than 2024 at the same circuits and laps,
because that difference is the cost of a season.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .. import config
from ..data import store
from ..models.pace import TARGET, add_reference_pace, feature_frame
from ..models.train import SELECTED
from ..sim.engine import SimComponents, simulate_field
from ..sim.scenario import observed_plans
from ..sim.state import build_state
from .replay import (
    COVERAGE_LEVELS,
    DEFAULT_FROM_LAPS,
    GATE_RACE_TIME_PCT,
    ReplayResult,
    _caution_laps,
    _caution_regimes,
    _compare_laps,
    _compare_race_times,
)

log = logging.getLogger(__name__)


def training_season_model():
    """The pace model fitted on the whole training season. No circuit is held out.

    Deliberately different from ``replay.loro_pace_model``: here the held-out thing is the
    *season*, so holding out the circuit as well would measure two transfers at once and
    would also be a fit nobody would ever deploy. A team going into 2025 uses everything it
    has from 2024.
    """
    laps = add_reference_pace(store.fittable_laps())
    train = store.trainable_laps(laps).dropna(subset=[TARGET, "tyre_age"])
    log.info("fitted on %d clean laps from %d", len(train), config.TRAINING_SEASON)
    return SELECTED().fit(feature_frame(train), train[TARGET])


def replay_holdout_race(
    circuit: str,
    season: int,
    from_lap: int = 20,
    n_sims: int = 500,
    seed: int = config.SEED,
    pace_model=None,
    components: SimComponents | None = None,
) -> ReplayResult | None:
    """Replay one held-out-season race with constants fitted on the training season."""
    try:
        laps = store.race_laps(circuit, season=season)
    except KeyError:
        return None
    total_laps = store.race_total_laps(circuit, season=season)
    if laps["lap_number"].max() <= from_lap:
        return None

    pace_model = pace_model or training_season_model()
    components = components or SimComponents.fit(pace_model)

    state = build_state(laps, lap=from_lap, total_laps=total_laps, pace_model=pace_model)
    simulation = simulate_field(
        state,
        observed_plans(laps, after_lap=from_lap),
        components,
        n_sims=n_sims,
        seed=seed,
        to_lap=total_laps,
        caution_laps=_caution_laps(laps[laps["lap_number"] > from_lap]),
        caution_regimes=_caution_regimes(laps[laps["lap_number"] > from_lap]),
        rival_policy="sim_pace",
    )
    return ReplayResult(
        circuit,
        from_lap,
        total_laps,
        n_sims,
        _compare_laps(simulation, laps, from_lap, total_laps),
        _compare_race_times(simulation, laps, from_lap, total_laps),
        simulation,
    )


def holdout_all(
    circuits: tuple[str, ...] | None = None,
    from_laps: tuple[int, ...] = DEFAULT_FROM_LAPS,
    n_sims: int = 500,
    season: int | None = None,
) -> tuple[pd.DataFrame, list[ReplayResult]]:
    """Replay every held-out-season race the store has, from every decision lap."""
    season = season or config.HOLDOUT_SEASONS[0]
    circuits = circuits or tuple(c.key for c in config.TRAINING_CIRCUITS)

    model = training_season_model()
    components = SimComponents.fit(model)

    rows, results = [], []
    for circuit in circuits:
        for from_lap in from_laps:
            r = replay_holdout_race(
                circuit, season, from_lap, n_sims, pace_model=model, components=components
            )
            if r is None:
                log.info("no %d race stored for %s", season, circuit)
                continue
            results.append(r)
            rows.append({**r.metrics(), "season": season})
    return pd.DataFrame(rows), results


def main(argv: list[str] | None = None) -> int:
    from .replay import replay_all

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", nargs="*", default=None)
    parser.add_argument("--from-laps", nargs="*", type=int, default=list(DEFAULT_FROM_LAPS))
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the 2024 leave-one-race-out run this is meant to be read against",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    season = args.season or config.HOLDOUT_SEASONS[0]
    circuits = tuple(args.circuits) if args.circuits else None

    summary, results = holdout_all(circuits, tuple(args.from_laps), args.n_sims, season)
    if summary.empty:
        print(f"no {season} races in the store. Ingest with:")
        print(f"  uv run python -m apexstrategy.data.build --seasons {config.SEASON} {season}")
        return 1

    print(f"\n=== {season} held out, model fitted on {config.TRAINING_SEASON} ===")
    print(summary.round(3).to_string(index=False))

    cols = [
        "clean_lap_mae_s",
        "clean_lap_bias_s",
        "race_time_median_pct",
        *[f"cover_{int(x * 100)}" for x in COVERAGE_LEVELS],
    ]
    print("\n=== by decision lap ===")
    print(summary.groupby("from_lap")[cols].mean().round(3).to_string())

    passed = [r for r in results if r.passes()[0]]
    print(f"\n=== gate: median race-time error < {GATE_RACE_TIME_PCT}% ===")
    print(f"passed {len(passed)}/{len(results)} replays")
    for r in results:
        ok, why = r.passes()
        if not ok:
            print(f"  FAIL {r.circuit}@L{r.from_lap}: {why}")

    if not args.no_baseline:
        # The comparison is the point. A 2025 number on its own says nothing; a 2025 number
        # beside the 2024 number at the same circuits and laps is the cost of a season.
        base_summary, base_results = replay_all(circuits, tuple(args.from_laps), args.n_sims)
        print(f"\n=== cost of the season jump ({config.TRAINING_SEASON} -> {season}) ===")
        merged = (
            base_summary.groupby("from_lap")[cols]
            .mean()
            .join(summary.groupby("from_lap")[cols].mean(), lsuffix="_2024", rsuffix="_holdout")
        )
        for col in cols:
            merged[f"{col}_delta"] = merged[f"{col}_holdout"] - merged[f"{col}_2024"]
        show = [
            c for c in merged.columns if c.startswith(("race_time_median_pct", "clean_lap_mae"))
        ]
        print(merged[show].round(3).to_string())
        print(
            f"\n{config.TRAINING_SEASON}: {sum(r.passes()[0] for r in base_results)}"
            f"/{len(base_results)} passed   {season}: {len(passed)}/{len(results)} passed"
        )

    return 0 if len(passed) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
