"""Phase 11: freeze the decision grid to JSON the showcase can serve statically.

    uv run python -m apexstrategy.showcase.export
    uv run python -m apexstrategy.showcase.export --circuits hungary --laps 12 20 28 36 44

## Everything is leave-one-race-out, including the demo

The showcase's whole claim is that these are real numbers. A demo run with a pace model
that trained on the race being demonstrated would be a flattering lie told in a nice
typeface, so the pace model, the pit-loss sample, the noise components and the compound
caps are all refitted per circuit with that circuit held out, exactly as
``evaluation.replay`` does it. The exported payload records that it was done.

## What ends up in the file

Three payloads per race, matching the three MVP pages (Part 10):

- ``race`` -- stints, stops, cautions and the data-quality decomposition, all observed.
- ``decisions`` -- one entry per (decision lap, driver): the state as known at that lap,
  the ranked candidates, and the arithmetic explanation of the recommendation.
- ``provenance`` -- which class every series on the page belongs to, so the front end
  cannot silently render a projection as though it were a measurement.

## Size is a design constraint, not an afterthought

The full candidate set runs to ~100 plans per decision and the front end can only show a
dozen. Exporting all of them would put megabytes of JSON on the wire to render nothing, so
the ranking is truncated on export and the count it was truncated from travels with it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import store
from ..data.clean import is_under_caution
from ..sim.engine import SimComponents
from ..sim.scenario import Scenario, observed_plans
from ..sim.state import RaceState, build_state
from ..strategy.candidates import fit_max_stint_share, is_physically_plausible
from ..strategy.engine import recommend
from ..strategy.explain import caution_option_value, explain

log = logging.getLogger(__name__)

#: Decision laps the grid is frozen at. Early enough to show a state with almost no
#: observed pace, late enough to show one where the answer is nearly settled.
DEFAULT_DECISION_LAPS = (10, 20, 30, 40)

#: Runners a decision is computed for at each lap. The front end lets a visitor switch
#: between them; beyond the top handful the strategies stop being interesting and the
#: payload stops being small.
DEFAULT_DRIVERS_PER_LAP = 6

#: Candidates kept per decision. The ranking computes ~100; a table shows a dozen.
EXPORTED_CANDIDATES = 12

#: Draws per candidate. Matches the engine default so the exported numbers are the ones
#: the CLI prints, rather than a quieter version produced for the demo.
EXPORT_SIMS = 1000

DEFAULT_OUTPUT_DIR = config.PROJECT_ROOT / "web" / "public" / "data"


def _round(value, digits: int = 3):
    """Round for the wire. Float64 to 17 significant figures is most of the payload."""
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_round(v, digits) for v in value]
    if isinstance(value, float | np.floating):
        return None if not np.isfinite(value) else round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def race_payload(circuit: str) -> dict:
    """Observed race shape: stints, stops, cautions, and where the laps went.

    Every series here is measurement, which is why the data-quality decomposition sits in
    the same payload rather than on a diagnostics page nobody opens. The exclusion counts
    are the honest reading of how much of this race the models were allowed to learn from.
    """
    laps = store.race_laps(circuit)
    total_laps = store.race_total_laps(circuit)
    results = store.read_table("results")
    results = results[results["circuit"] == circuit]
    stints = store.read_table("stints")
    stints = stints[stints["circuit"] == circuit]
    pits = store.read_table("pits")
    pits = pits[pits["circuit"] == circuit]

    order = results.sort_values("finish_position", na_position="last")
    exclusions = laps.loc[~laps["is_clean"], "exclusion_reason"].value_counts().rename("n_laps")

    caution = sorted(
        laps.loc[is_under_caution(laps["track_status"]), "lap_number"].astype(int).unique().tolist()
    )
    circuit_row = store.read_table("circuits")
    circuit_row = circuit_row[
        (circuit_row["circuit"] == circuit) & (circuit_row["scope"] == "per_circuit")
    ]

    return _round(
        {
            "circuit": circuit,
            "season": int(laps["season"].iloc[0]),
            "round": int(laps["round"].iloc[0]),
            "total_laps": total_laps,
            "in_training_pool": circuit in {c.key for c in config.TRAINING_CIRCUITS},
            "drivers": [
                {
                    "code": str(r.driver),
                    "team": str(r.team),
                    "grid": None if pd.isna(r.grid_position) else int(r.grid_position),
                    "finish": None if pd.isna(r.finish_position) else int(r.finish_position),
                    "classified": str(r.classified_position),
                    "retired": bool(r.is_retired),
                    "laps_completed": int(r.laps_completed),
                }
                for r in order.itertuples()
            ],
            "stints": [
                {
                    "driver": str(s.driver),
                    "stint": int(s.stint_number),
                    "compound": str(s.compound),
                    "start_lap": int(s.start_lap),
                    "end_lap": int(s.end_lap),
                    "laps": int(s.n_laps),
                    # A stint that reached the flag is a lower bound on what the set could
                    # do, and the timeline says so rather than drawing it as a decision.
                    "censored": bool(s.is_censored),
                }
                for s in stints.sort_values(["driver", "stint_number"]).itertuples()
            ],
            "stops": [
                {
                    "driver": str(p.driver),
                    "lap": int(p.lap),
                    "stop": int(p.stop_number),
                    "compound_fitted": (
                        None if pd.isna(p.compound_fitted) else str(p.compound_fitted)
                    ),
                    "duration_s": None if pd.isna(p.pit_duration_s) else float(p.pit_duration_s),
                    "under_caution": bool(p.under_caution),
                }
                for p in pits.sort_values(["driver", "stop_number"]).itertuples()
            ],
            "caution_laps": caution,
            "data_quality": {
                "total_laps_recorded": int(len(laps)),
                "clean_laps": int(laps["is_clean"].sum()),
                "clean_yield": float(laps["is_clean"].mean()),
                "exclusions": [{"reason": str(k), "n_laps": int(v)} for k, v in exclusions.items()],
            },
            "constants": (
                {}
                if circuit_row.empty
                else {
                    "race_progress_s_per_lap": float(
                        circuit_row["race_progress_coef_s_per_lap"].iloc[0]
                    ),
                    "pit_loss_median_s": float(circuit_row["pit_loss_median_s"].iloc[0]),
                    "pit_loss_n": int(circuit_row["pit_loss_n"].iloc[0]),
                }
            ),
        }
    )


def pit_window(ranking) -> list[dict]:
    """Best achievable delta for each possible *next stop lap*, over every candidate.

    This is the graphic a strategist actually reads, and it falls out of the ranking for
    free. Collapsing ~100 plans onto the one axis that is live right now -- when to box --
    turns a table nobody can scan into a curve with a visible floor, and the width of that
    floor is the pit window. A ranked list hides the window; it shows twelve rows that are
    four laps and one compound choice wearing twelve different labels.
    """
    base = ranking.baseline.race_time_s
    best: dict[int, dict] = {}
    for e in ranking.evaluations:
        if not e.scenario.plan:
            continue
        lap = e.scenario.plan[0].lap
        delta = float((e.race_time_s - base).mean())
        if lap not in best or delta < best[lap]["delta_s"]:
            best[lap] = {
                "lap": lap,
                "delta_s": delta,
                "label": e.label,
                "n_stops": e.n_stops,
                "compound": e.scenario.plan[0].compound,
            }
    return [best[k] for k in sorted(best)]


def distinct_options(ranking) -> list[dict]:
    """The best plan of each genuinely different *shape*, not each different label.

    Measured on Hungary at lap 20: twelve exported candidates reduce to two decisions,
    HARD or MEDIUM on the next set. Nobody chooses between "box lap 22" and "box lap 23";
    they choose a shape and then a lap inside the window. Presenting near-duplicates as
    separate options manufactures a precision the simulation explicitly says it does not
    have -- four of those twelve were inside Monte Carlo noise of each other.
    """
    base = ranking.baseline.race_time_s

    def row(e) -> dict:
        d = e.race_time_s - base
        return {
            "n_stops": e.n_stops,
            "next_compound": e.scenario.plan[0].compound if e.scenario.plan else "none",
            "next_lap": e.scenario.plan[0].lap if e.scenario.plan else None,
            "label": e.label,
            "delta_s": float(d.mean()),
            "delta_se_s": float(d.std(ddof=1) / np.sqrt(len(d))),
            "p_faster": float((d < 0).mean()),
            "is_baseline": e is ranking.baseline,
        }

    best: dict[tuple, dict] = {}
    for e in ranking.evaluations:
        shape = (e.n_stops, e.scenario.plan[0].compound if e.scenario.plan else "none")
        candidate = row(e)
        if shape not in best or candidate["delta_s"] < best[shape]["delta_s"]:
            best[shape] = candidate

    # The plan that was actually run is always its own row, even when a better plan of the
    # same shape displaces it. Grouping absorbed it -- at Hungary lap 20 the executed
    # two-stop lost its slot to a two-stop two laps later -- and a comparison that hides
    # what the team did is not a comparison.
    if not any(r["is_baseline"] for r in best.values()):
        best[("baseline",)] = row(ranking.baseline)

    return sorted(best.values(), key=lambda r: r["delta_s"])


def _state_payload(state: RaceState, ego: str) -> dict:
    driver = state.by_code[ego]
    return {
        "lap": state.lap,
        "total_laps": state.total_laps,
        "track_status": state.track_status,
        "position": state.position_of(ego),
        "compound": driver.compound,
        "tyre_age": driver.tyre_age,
        "stops_made": driver.pit_stops,
        "compounds_used": sorted(driver.compounds_used),
        "gap_to_leader_s": driver.gap_to_leader_s,
        # How much this state actually knows about the car. At lap 10 it is often almost
        # nothing, and the front end says so instead of implying uniform confidence.
        "clean_laps_observed": driver.clean_laps_observed,
        "reference_pace_s": driver.reference_pace_s,
        "race_progress_adjustment_s_per_lap": state.race_progress_adjustment_s_per_lap,
    }


def decision_payload(
    state: RaceState,
    ego: str,
    components: SimComponents,
    plans: dict,
    max_stint_share: dict[str, float],
    n_sims: int = EXPORT_SIMS,
) -> dict | None:
    """One (lap, driver) decision: the state, the ranking, and why the top plan wins."""
    baseline = Scenario(ego=ego, plan=plans.get(ego, ()), label="as executed")
    try:
        ranking = recommend(
            state,
            ego,
            components,
            rival_plans=plans,
            n_sims=n_sims,
            baseline=baseline,
            max_stint_share=max_stint_share,
        )
    except ValueError as exc:  # no legal plan exists; a real state, not an error
        log.info("no candidates for %s at lap %d: %s", ego, state.lap, exc)
        return None

    payload = ranking.to_dict(EXPORTED_CANDIDATES)
    payload["state"] = _state_payload(state, ego)
    # A driver whose executed plan runs a stint past anything observed for that compound
    # gets a delta built on an extrapolation. The engine warns; the payload carries it so
    # the page can too, rather than printing a confident number over a shaky baseline.
    payload["baseline_in_support"] = bool(
        is_physically_plausible(state, ego, baseline.plan, max_stint_share)
    )
    payload["candidates_evaluated"] = len(ranking.evaluations)
    payload["candidates_shown"] = min(EXPORTED_CANDIDATES, len(ranking.evaluations))
    payload["indistinguishable_from_best"] = ranking.indistinguishable_from_best()["label"].tolist()

    payload["window"] = pit_window(ranking)
    payload["options"] = distinct_options(ranking)

    if ranking.best is not ranking.baseline:
        # What a safety car would do to the call. Exogenous by design (Challenge 3): this
        # is a contrast under an assumption the caller supplies, never a prediction, and it
        # leans on the two thinnest constants in the project. It is here because it is the
        # first thing a strategist asks about any plan, and answering "we do not model
        # that" while the number is computable would be a different kind of dishonesty.
        horizon = frozenset(range(state.lap + 1, min(state.lap + 11, state.total_laps + 1)))
        payload["caution"] = {
            "laps": sorted(horizon),
            "option_value_s": round(
                caution_option_value(
                    state,
                    ego,
                    ranking.best.scenario,
                    ranking.baseline.scenario,
                    components,
                    horizon,
                    rival_plans=plans,
                    n_sims=1000,
                ),
                3,
            ),
        }
    else:
        payload["caution"] = None

    if ranking.best is not ranking.baseline:
        why = explain(
            state,
            ego,
            ranking.best.scenario,
            ranking.baseline.scenario,
            components,
            rival_plans=plans,
        )
        payload["explanation"] = why.to_dict()
    else:
        payload["explanation"] = None
    return _round(payload)


def export_circuit(
    circuit: str,
    decision_laps: tuple[int, ...] = DEFAULT_DECISION_LAPS,
    drivers_per_lap: int = DEFAULT_DRIVERS_PER_LAP,
    n_sims: int = EXPORT_SIMS,
) -> dict:
    """Compute the whole grid for one race, with that race held out of every fit."""
    from ..evaluation.replay import loro_pace_model

    started = time.perf_counter()
    laps = store.race_laps(circuit)
    total_laps = store.race_total_laps(circuit)

    trainable = circuit in {c.key for c in config.TRAINING_CIRCUITS}
    decisions: list[dict] = []

    # Monaco (D3) and Silverstone (D10) are stored and excluded from fitting, so no
    # recommendation is offered for them. Showing one anyway would be the exact dishonesty
    # the exclusions exist to prevent -- and Silverstone would not even run, because a
    # third of its laps are on a compound the pace model has never seen. The race payload
    # is still exported: an excluded race is part of the story, not a gap in it.
    if not trainable:
        return {
            "schema": 1,
            "generated_by": "apexstrategy.showcase.export",
            "race": race_payload(circuit),
            "decisions": decisions,
            "method": _excluded_method(circuit),
            "provenance": PROVENANCE,
        }

    pace_model = loro_pace_model(circuit)
    components = SimComponents.fit(pace_model, exclude_circuit=circuit)
    max_stint_share = fit_max_stint_share(exclude_circuit=circuit)

    for lap in decision_laps:
        if lap >= total_laps:
            continue
        state = build_state(laps, lap=lap, total_laps=total_laps, pace_model=pace_model)
        plans = observed_plans(laps, after_lap=lap)
        for driver in [d.code for d in state.order()[:drivers_per_lap]]:
            entry = decision_payload(state, driver, components, plans, max_stint_share, n_sims)
            if entry is not None:
                decisions.append(entry)
        log.info("%s lap %d: %d decisions", circuit, lap, len(decisions))

    return {
        "schema": 1,
        "generated_by": "apexstrategy.showcase.export",
        "race": race_payload(circuit),
        "decisions": decisions,
        "method": {
            # Stated in the payload so it reaches the page, not just the repository.
            "held_out": circuit,
            "excluded": False,
            "in_training_pool": trainable,
            "note": (
                "Pace model, pit-loss sample, noise components and compound caps are all "
                "refitted with this circuit held out, so nothing shown here was fitted on "
                "the race it is shown against."
            ),
            "n_sims": n_sims,
            "objective": "expected_race_time_s",
            "rival_policy": "sim_pace",
            "not_modelled": [
                "Overtaking and track position (D5). Cars pass freely on cumulative time.",
                "Traffic on rejoin (D5). Projections are clean-air times and run optimistic "
                "for a car that would have been held up.",
                "Safety cars are never predicted (Challenge 3). They are an assumption you "
                "supply, and the grid here assumes the cautions that actually occurred.",
            ],
            "wall_time_s": round(time.perf_counter() - started, 2),
        },
        "provenance": PROVENANCE,
    }


def _excluded_method(circuit: str) -> dict:
    """Why a stored race gets no recommendation, in the words the decision was recorded in."""
    return {
        "held_out": circuit,
        "in_training_pool": False,
        "excluded": True,
        "reason": config.CIRCUITS_BY_KEY[circuit].note,
        "wall_time_s": 0.0,
        "note": (
            "This race is in the store and excluded from model fitting, so no strategy "
            "recommendation is offered for it. The observed race is shown in full."
        ),
    }


#: The four classes Part 10 makes a hard requirement, carried in the payload so the front
#: end renders from data rather than from a convention someone has to remember.
PROVENANCE = {
    "observed": {
        "label": "Observed",
        "meaning": "Measured. This actually happened.",
    },
    "predicted": {
        "label": "Predicted",
        "meaning": "Model output for the state as known at the decision lap.",
    },
    "simulated": {
        "label": "Simulated",
        "meaning": "Forward projection under an assumption, with its full distribution.",
    },
    "counterfactual": {
        "label": "Counterfactual",
        "meaning": "A strategy nobody ran. Never validated against an outcome.",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", nargs="*", default=None)
    parser.add_argument("--laps", nargs="*", type=int, default=list(DEFAULT_DECISION_LAPS))
    parser.add_argument("--drivers", type=int, default=DEFAULT_DRIVERS_PER_LAP)
    parser.add_argument("--n-sims", type=int, default=EXPORT_SIMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    circuits = tuple(args.circuits) if args.circuits else tuple(c.key for c in config.CIRCUITS)
    args.out.mkdir(parents=True, exist_ok=True)

    index = []
    for circuit in circuits:
        payload = export_circuit(circuit, tuple(args.laps), args.drivers, args.n_sims)
        path = args.out / f"{circuit}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        size_kb = path.stat().st_size / 1024
        print(
            f"{circuit:11s} {len(payload['decisions']):3d} decisions  "
            f"{size_kb:7.1f} KB  {payload['method']['wall_time_s']:6.1f}s"
        )
        index.append(
            {
                "circuit": circuit,
                "season": payload["race"]["season"],
                "round": payload["race"]["round"],
                "total_laps": payload["race"]["total_laps"],
                "in_training_pool": payload["race"]["in_training_pool"],
                "n_decisions": len(payload["decisions"]),
                "clean_yield": payload["race"]["data_quality"]["clean_yield"],
                "size_kb": round(size_kb, 1),
            }
        )

    index_path = args.out / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "demo_circuit": config.DEMO_CIRCUIT_KEY,
                "decision_laps": list(args.laps),
                "races": index,
                "provenance": PROVENANCE,
            },
            indent=2,
        )
    )
    print(f"\nwrote {len(index)} races + index.json to {args.out}")
    return 0
