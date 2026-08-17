"""Phase 7: replay validation -- the credibility gate.

Hand the simulator a real race, at a real lap, with the strategies every driver actually
executed, and ask it to reproduce what happened. **If it cannot reproduce a race it has
been given the true answers to, every counterfactual downstream of it is worthless.**
This is the single most important test in the project, and the strategy engine is not
built until it passes.

## The replay is leave-one-race-out, not in-sample

Replaying Hungary with a pace model that trained on Hungary, or with a pit-loss
distribution that includes Hungary's own stops, would be circuit-constant leakage
(architecture-proposal.md, Part 13, risk 5) and would report a flattering number that
says nothing about an unseen race. Both the pace model and the pit-loss sample are
refitted per replayed circuit with that circuit held out.

## What is fed in, and why that is still honest

Safety-car laps are supplied from observation rather than predicted. That is not a
concession: SC is exogenous by design (Challenge 3), and the question the gate asks is
"given the strategies *and the cautions*, does the race reconstruct?". It matters little
in practice here -- the current circuit set contains 27 caution laps in total, 0.2% of
all laps.

The scheduled race distance is likewise an input. It is a calendar fact, not an outcome.

## Errors are decomposed, not averaged away

The MVP does not model traffic (D5) or overtaking, so projections are clean-air times and
will run optimistic for a car that spent the race stuck behind someone. Rather than
caveat that in prose, ``lap_errors`` labels every lap by category and the per-category
error contributions sum exactly to the total race-time error. The cost of the omission
becomes a measured number, and the gate is read with it in view.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..data import store
from ..data.clean import is_under_caution
from ..models.pace import TARGET, add_reference_pace, feature_frame
from ..models.train import SELECTED
from ..sim.engine import SimComponents, SimulationResult, simulate_field
from ..sim.scenario import observed_plans
from ..sim.state import build_state

log = logging.getLogger(__name__)

#: **The gate.** Median absolute race-time error over the replayed window, as a share of
#: that window. Race time is the metric because the MVP is denominated in expected race
#: time (D5); the architecture doc fixes this explicitly -- "the Phase 7 replay gate is
#: assessed on race-time MAE only", with finish-order rank correlation deferred to
#: product-Phase 2 along with positions.
#:
#: 1% is the size of the decision the product exists to make: a pit stop costs ~22s
#: against a ~4,500s race, so ~0.5%. Reproduction error above the decision size would
#: swamp the signal the recommendation is built from.
GATE_RACE_TIME_PCT = 1.0

#: Diagnostics, reported alongside the gate but not gating. They localise *where* a
#: replay goes wrong rather than deciding whether it passed. The pace model's residual sd
#: is 0.73s, so a perfect simulator posts roughly 0.58s of clean-lap MAE.
DIAGNOSTIC_CLEAN_LAP_MAE_S = 1.0
#: Mean signed error per green clean lap. Under leave-one-race-out this is dominated by
#: the race-progress transfer error documented in the Phase 7 build log, which is a
#: consequence of holding one race per circuit rather than a simulator defect -- so it is
#: measured and reported, not gated on.
DIAGNOSTIC_CLEAN_LAP_BIAS_S = 0.30

#: Default decision laps to replay from. Early states have little observed pace to build
#: a reference from and a long horizon to compound it over; late states have the opposite.
#: Reporting the sweep rather than one number is the point.
DEFAULT_FROM_LAPS = (5, 15, 25, 35)


def _caution_laps(laps: pd.DataFrame) -> frozenset[int]:
    """Lap numbers run under SC/VSC/red -- the exogenous input the replay is handed."""
    flagged = is_under_caution(laps["track_status"])
    return frozenset(laps.loc[flagged, "lap_number"].astype(int).unique().tolist())


def _lap_category(laps: pd.DataFrame) -> pd.Series:
    """Label each lap by what dominates its time. Deliberately not ``exclusion_reason``.

    Same priority ordering, but derived from the raw columns, because the stored label
    answers "why was this lap excluded from *training*" and this asks "what explains this
    lap's *time*". They agree almost always; relying on that agreement would be the exact
    mistake the project has already made once.
    """
    cat = pd.Series("clean", index=laps.index, dtype="object")
    cat[laps["in_dirty_air"].to_numpy()] = "dirty_air"
    cat[is_under_caution(laps["track_status"]).to_numpy()] = "caution"
    cat[laps["is_pit_out"].to_numpy()] = "pit_out"
    cat[laps["is_pit_in"].to_numpy()] = "pit_in"
    cat[(laps["lap_number"] == 1).to_numpy()] = "lap_1"
    return cat


def loro_pace_model(circuit: str):
    """Fit the selected pace model on every trainable circuit except ``circuit``."""
    laps = store.read_table("laps")
    allowed = {c.key for c in config.TRAINING_CIRCUITS} - {circuit}
    laps = laps[laps["circuit"].isin(allowed)]
    laps = add_reference_pace(laps)
    train = laps[laps["is_clean"]].dropna(subset=[TARGET, "tyre_age"])
    return SELECTED().fit(feature_frame(train), train[TARGET])


@dataclass(frozen=True)
class ReplayResult:
    """One replayed race from one decision lap, with its errors already decomposed."""

    circuit: str
    from_lap: int
    to_lap: int
    n_sims: int
    laps: pd.DataFrame  #: per driver-lap: actual, predicted, error, category
    races: pd.DataFrame  #: per driver: elapsed-time error over the replayed window
    simulation: SimulationResult

    @property
    def clean(self) -> pd.DataFrame:
        return self.laps[self.laps["category"] == "clean"]

    def metrics(self) -> dict[str, float]:
        clean, races = self.clean, self.races
        return {
            "circuit": self.circuit,
            "from_lap": self.from_lap,
            "n_drivers": len(races),
            "clean_laps": len(clean),
            "clean_lap_mae_s": float(clean["error_s"].abs().mean()),
            "clean_lap_bias_s": float(clean["error_s"].mean()),
            "race_time_mae_s": float(races["error_s"].abs().mean()),
            "race_time_median_pct": float(races["abs_pct"].median()),
            "race_time_bias_s": float(races["error_s"].mean()),
            # Phase 8 baseline. i.i.d. per-lap sampling cancels over a race, so this is
            # expected to be far below 0.80 until hierarchical noise arrives.
            "race_time_cover_80": float(races["covered_80"].mean()),
        }

    def decomposition(self) -> pd.DataFrame:
        """Mean and total signed error by lap category.

        ``total_error_s`` sums across categories to the field's total race-time error,
        so the traffic bias is read off directly rather than argued about.
        """
        g = self.laps.groupby("category")["error_s"]
        out = pd.DataFrame({"n_laps": g.size(), "mean_error_s": g.mean(), "total_error_s": g.sum()})
        out["share_of_total"] = out["total_error_s"] / out["total_error_s"].sum()
        # Deliberately not rounded: the caller rounds for display, and a decomposition
        # that only adds up to three decimal places is not an arithmetic explanation.
        return out.sort_values("total_error_s")

    def passes(self) -> tuple[bool, str]:
        """Whether this replay clears the gate, and by how much."""
        pct = self.metrics()["race_time_median_pct"]
        return pct <= GATE_RACE_TIME_PCT, f"race-time error {pct:.2f}% vs {GATE_RACE_TIME_PCT}%"

    def diagnostics(self) -> list[str]:
        """Non-gating flags that say where a replay is drifting. Empty is good."""
        m, out = self.metrics(), []
        if m["clean_lap_mae_s"] > DIAGNOSTIC_CLEAN_LAP_MAE_S:
            out.append(f"clean-lap MAE {m['clean_lap_mae_s']:.3f}s")
        if abs(m["clean_lap_bias_s"]) > DIAGNOSTIC_CLEAN_LAP_BIAS_S:
            out.append(f"clean-lap bias {m['clean_lap_bias_s']:+.3f}s")
        return out


def replay_race(
    circuit: str,
    from_lap: int = 15,
    n_sims: int = 500,
    seed: int = config.SEED,
    pace_model=None,
    components: SimComponents | None = None,
) -> ReplayResult:
    """Replay one race from ``from_lap`` under the strategies actually executed.

    Args:
        circuit: Circuit key, as stored.
        from_lap: Decision lap L. The state is built from laps <= L; laps after it are
            projected and compared.
        n_sims: Monte Carlo draws. The comparison uses the median across draws, so this
            mainly controls how stable that median is.
        pace_model: Fitted model, held out of ``circuit``. Fitted here if omitted.
        components: Pit-loss and caution constants, likewise held out.
    """
    laps = store.race_laps(circuit)
    total_laps = store.race_total_laps(circuit)

    pace_model = pace_model or loro_pace_model(circuit)
    components = components or SimComponents.fit(pace_model, exclude_circuit=circuit)

    state = build_state(laps, lap=from_lap, total_laps=total_laps, pace_model=pace_model)
    plans = observed_plans(laps, after_lap=from_lap)
    simulation = simulate_field(
        state,
        plans,
        components,
        n_sims=n_sims,
        seed=seed,
        to_lap=total_laps,
        caution_laps=_caution_laps(laps[laps["lap_number"] > from_lap]),
        rival_policy="sim_pace",
    )

    lap_rows = _compare_laps(simulation, laps, from_lap, total_laps)
    race_rows = _compare_race_times(simulation, laps, from_lap, total_laps)
    return ReplayResult(circuit, from_lap, total_laps, n_sims, lap_rows, race_rows, simulation)


def _compare_laps(
    sim: SimulationResult, laps: pd.DataFrame, from_lap: int, to_lap: int
) -> pd.DataFrame:
    """Per driver-lap actual vs predicted, restricted to laps the driver actually ran."""
    predicted = np.median(sim.lap_times_s, axis=0)  # (n_drivers, n_laps)
    frame = pd.DataFrame(
        {
            "driver": np.repeat(sim.drivers, predicted.shape[1]),
            "lap_number": np.tile(sim.laps, len(sim.drivers)),
            "predicted_s": predicted.ravel(),
        }
    )

    window = laps[(laps["lap_number"] > from_lap) & (laps["lap_number"] <= to_lap)].copy()
    window["category"] = _lap_category(window)
    actual = window[
        ["driver", "lap_number", "lap_time_s", "category", "compound", "tyre_age"]
    ].rename(columns={"lap_time_s": "actual_s"})

    out = frame.merge(actual, on=["driver", "lap_number"], how="inner").dropna(subset=["actual_s"])
    out["error_s"] = out["predicted_s"] - out["actual_s"]
    return out.reset_index(drop=True)


def _compare_race_times(
    sim: SimulationResult, laps: pd.DataFrame, from_lap: int, to_lap: int
) -> pd.DataFrame:
    """Elapsed time from ``from_lap`` to each driver's own final lap, actual vs projected.

    Comparing over the laps a driver actually completed sidesteps having to predict
    retirements, which the MVP does not model and does not claim to. A driver who stopped
    on lap 30 contributes a 30-lap comparison, not a missing row.
    """
    rows = []
    for driver in sim.drivers:
        own = laps[(laps["driver"] == driver) & laps["cumulative_time_s"].notna()]
        end = int(min(own["lap_number"].max(), to_lap))
        if end <= from_lap:
            continue

        start_t = own.loc[own["lap_number"] == from_lap, "cumulative_time_s"]
        end_t = own.loc[own["lap_number"] == end, "cumulative_time_s"]
        if start_t.empty or end_t.empty:
            continue

        actual = float(end_t.iloc[0] - start_t.iloc[0])
        draws = sim.elapsed_s(driver, end)
        lo, hi = np.quantile(draws, [0.1, 0.9])
        rows.append(
            {
                "driver": driver,
                "laps_compared": end - from_lap,
                "actual_s": actual,
                "predicted_s": float(np.median(draws)),
                "error_s": float(np.median(draws)) - actual,
                "abs_pct": abs(float(np.median(draws)) - actual) / actual * 100,
                "covered_80": bool(lo <= actual <= hi),
            }
        )
    return pd.DataFrame(rows).sort_values("error_s").reset_index(drop=True)


def replay_all(
    circuits: tuple[str, ...] | None = None,
    from_laps: tuple[int, ...] = DEFAULT_FROM_LAPS,
    n_sims: int = 500,
) -> tuple[pd.DataFrame, list[ReplayResult]]:
    """Replay every trainable circuit from every decision lap in the sweep."""
    circuits = circuits or tuple(c.key for c in config.TRAINING_CIRCUITS)
    results, rows = [], []
    for circuit in circuits:
        # Fit once per circuit: the held-out model does not depend on the decision lap.
        model = loro_pace_model(circuit)
        components = SimComponents.fit(model, exclude_circuit=circuit)
        for from_lap in from_laps:
            r = replay_race(circuit, from_lap, n_sims, pace_model=model, components=components)
            results.append(r)
            rows.append(r.metrics())
    return pd.DataFrame(rows), results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", nargs="*", default=None)
    parser.add_argument("--from-laps", nargs="*", type=int, default=list(DEFAULT_FROM_LAPS))
    parser.add_argument("--n-sims", type=int, default=500)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary, results = replay_all(
        tuple(args.circuits) if args.circuits else None,
        tuple(args.from_laps),
        args.n_sims,
    )

    print("\n=== replay validation (leave-one-race-out) ===")
    print(summary.round(3).to_string(index=False))

    print("\n=== by decision lap (mean across circuits) ===")
    by_lap = summary.groupby("from_lap")[
        ["clean_lap_mae_s", "clean_lap_bias_s", "race_time_median_pct", "race_time_cover_80"]
    ].mean()
    print(by_lap.round(3).to_string())

    print("\n=== error decomposition by lap category, all replays pooled ===")
    pooled = pd.concat([r.laps.assign(circuit=r.circuit, from_lap=r.from_lap) for r in results])
    g = pooled.groupby("category")["error_s"]
    decomp = pd.DataFrame({"n_laps": g.size(), "mean_error_s": g.mean(), "total_error_s": g.sum()})
    decomp["share_of_total"] = decomp["total_error_s"] / decomp["total_error_s"].sum()
    print(decomp.sort_values("total_error_s").round(3).to_string())
    print(
        "\nTraffic is deliberately unmodelled (D5), so a negative dirty_air contribution\n"
        "is the size of that known bias, not a defect to be fixed here."
    )

    print(f"\n=== gate: median race-time error < {GATE_RACE_TIME_PCT}% ===")
    failures = {f"{r.circuit}@L{r.from_lap}": r.passes()[1] for r in results if not r.passes()[0]}
    print(f"passed {len(results) - len(failures)}/{len(results)} replays")
    for key, why in failures.items():
        print(f"  FAIL {key}: {why}")

    mature = [r for r in results if r.from_lap >= 15]
    mature_fail = [r for r in mature if not r.passes()[0]]
    print(f"\nfrom decision lap 15 onwards: {len(mature) - len(mature_fail)}/{len(mature)} passed")
    print("Lap-5 states carry almost no observed pace to build a reference from, and")
    print("strategy decisions do not cluster there; they are reported, not excused.")

    flagged = {f"{r.circuit}@L{r.from_lap}": r.diagnostics() for r in results if r.diagnostics()}
    if flagged:
        print("\n=== diagnostics (not gating) ===")
        for key, why in flagged.items():
            print(f"  {key}: {'; '.join(why)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
