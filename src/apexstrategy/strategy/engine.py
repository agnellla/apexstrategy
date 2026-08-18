"""Phase 9: the strategy engine -- generate, evaluate, rank, and say how sure it is.

    uv run python -m apexstrategy.strategy --circuit hungary --lap 20

Everything before this phase produces a *projection*. This is the layer that produces a
*decision*: given the race at lap L, which pit plan should this car run, by how much, and
with what downside. It is a thin wrapper around the Phase 7 simulator, which is the point
-- the credibility lives in the replay gate, and nothing here is allowed to add a claim
the simulator cannot support.

## Ranked on expected race time, per D5

Not finishing position and not championship points. The architecture doc's Part 9 argues
for points and is right about the general case: position value is wildly non-linear and
"expected finishing position 2.4" is an average over an ordinal scale. But points need
positions, positions need overtaking, and D5 defers both to product-Phase 2. Ranking on
expected race-time delta with a risk column is the doc's own stated consequence of D5, and
inventing a points number from a model that cannot resolve two cars meeting on track would
be the most confidently wrong output this system could produce.

## Candidates are compared paired, and that is the whole design

The absolute race time over a 50-lap projection has sd **33 s**, most of it shared: how
fast the track came to everyone today, how far the reference pace was off. The difference
between two sensible pit plans is a handful of seconds. Under independent draws the
comparison is hopeless -- a nine-second gap measured at sd 45.7 s reads as a coin flip.

So every candidate is simulated against the *same* draws, and all reported quantities are
functions of the paired difference. The shared luck cancels, the difference sharpens to
sd 2.9 s, and "P(this plan beats the recommendation)" becomes a statement about strategy
rather than about weather. ``engine.RandomAlignment`` is what makes the pairing survive a
change in stop count; without it the one-stop-versus-two-stop comparison, which is the
main thing anyone wants to know, silently decorrelates.

## Risk is a downside on the *difference*, not on the race time

The doc asks for CVaR@10% and is right that risk should be a number rather than the label
"High". Applied to the absolute race time, though, it mostly measures the shared
uncertainty every candidate carries equally -- ``cvar10_s`` runs within a second or two
across the whole table -- so the column ranks nothing. It is still reported, because it is
the quantity the doc names and because it is the one comparable across decision laps.

The column that carries the risk is ``cvar10_delta_s``: in the worst tenth of
simulations *for this plan against the baseline*, how much time does choosing it cost.
With the baseline set to the plan the car is on, that reads directly as "if the switch
goes badly, this is how badly", and it is independent of how many candidates were
enumerated.

``regret_s`` and ``cvar10_regret_s`` sit beside it, measured against whichever plan turned
out best in that same simulation -- the cost of not being clairvoyant. They are the only
risk numbers that stay non-zero for the recommendation itself, which is why they are here,
but **they grow with the size of the candidate set**: enumerating more plans gives the
oracle more to win with, so a hundred-candidate ranking posts larger regrets than a
twenty-candidate one over the same race. Compare them within a ranking, never across two.

Expected race time still does the ranking. Risk is reported beside it rather than folded
in, because collapsing the two into one number needs a risk preference nobody has measured.

## What the numbers do not include

``delta_se_s`` is the **Monte Carlo** standard error of a delta -- how much of it is an
artefact of having drawn 1,000 simulations rather than infinity. It is not model error.
The simulator's own accuracy is the Phase 7 replay result (median race-time error ~0.5% of
the projected window) and its calibration is the Phase 8 result; both are larger than this
and neither is reduced by drawing more simulations. A gap of 0.2 s between two candidates
is not a recommendation no matter how tight the standard error on it is.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..sim.engine import RandomAlignment, SimComponents, alignment_for, simulate_field
from ..sim.scenario import PitInstruction, RivalPolicy, Scenario
from ..sim.state import RaceState
from .candidates import (
    DEFAULT_MAX_STOPS,
    fit_max_stint_share,
    generate_candidates,
    is_physically_plausible,
    refine_candidates,
)

log = logging.getLogger(__name__)

#: Leading candidates whose next stop is re-searched to the lap. Six is enough to cover
#: both stop counts and a couple of compound choices without the refinement pass growing
#: larger than the coarse one it is sharpening.
DEFAULT_REFINE_TOP = 6

#: Tail fraction the risk column averages over. 10% is the doc's choice.
CVAR_TAIL = 0.10

#: Two candidates separated by less than this many Monte Carlo standard errors are not
#: distinguished by the simulation, whatever the ranking says. Reported, not hidden.
SEPARATION_SE = 2.0

#: Phase 9 exit criterion: 20 candidates evaluated inside this many seconds.
GATE_SECONDS_PER_20_CANDIDATES = 5.0


@dataclass(frozen=True)
class StrategyEvaluation:
    """One candidate's projected race time, as a full sample rather than a summary.

    The array is kept because every comparison in this module is paired -- a summary
    statistic per candidate would be enough to rank them and not enough to say how likely
    one is to beat another, which is the part that makes it a decision tool.
    """

    scenario: Scenario
    #: ``(n_sims,)`` elapsed seconds from the decision lap to the flag.
    race_time_s: np.ndarray

    @property
    def n_stops(self) -> int:
        return self.scenario.n_stops

    @property
    def label(self) -> str:
        return self.scenario.label or self.scenario.describe()

    @property
    def next_stop(self) -> PitInstruction | None:
        return self.scenario.plan[0] if self.scenario.plan else None


@dataclass(frozen=True)
class StrategyRanking:
    """Every evaluated candidate for one driver at one decision lap, ranked.

    JSON-serialisable through ``to_dict`` (D9): the showcase front-end is fed precomputed
    scenario grids, so nothing here may depend on a Python consumer.
    """

    race_id: str
    circuit: str
    ego: str
    from_lap: int
    to_lap: int
    n_sims: int
    seed: int
    rival_policy: RivalPolicy
    caution_laps: frozenset[int]
    evaluations: tuple[StrategyEvaluation, ...]
    #: Index into ``evaluations`` of the plan everything is measured against. Defaults to
    #: the ranked best, so a delta reads as "time lost against the recommendation"; set it
    #: to the plan the team is actually on to read it as "time gained by switching".
    baseline_index: int
    wall_time_s: float

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise ValueError("a ranking needs at least one evaluated candidate")

    @property
    def baseline(self) -> StrategyEvaluation:
        return self.evaluations[self.baseline_index]

    @property
    def best(self) -> StrategyEvaluation:
        """Lowest expected race time. The recommendation."""
        return self.evaluations[int(np.argmin(self._times().mean(axis=1)))]

    def _times(self) -> np.ndarray:
        """``(n_candidates, n_sims)``. Rows are paired: column j is one shared draw."""
        return np.vstack([e.race_time_s for e in self.evaluations])

    def table(self, top: int | None = None) -> pd.DataFrame:
        """Ranked candidates with their deltas, win probabilities and risk.

        Every column except ``median_s``/``mean_s``/``cvar10_s`` is a function of the
        paired difference, which is the only form in which these quantities mean anything
        -- see the module docstring.
        """
        times = self._times()
        base = times[self.baseline_index]
        delta = times - base[None, :]
        # Regret against whichever plan turned out best in that same simulation -- the cost
        # of choosing this one without clairvoyance. Needs the draws to be common across
        # rows, which is exactly what RandomAlignment buys.
        regret = times - times.min(axis=0)[None, :]
        n = times.shape[1]

        rows = []
        for i, e in enumerate(self.evaluations):
            d, r = delta[i], regret[i]
            rows.append(
                {
                    "label": e.label,
                    "n_stops": e.n_stops,
                    "next_stop": f"L{e.next_stop.lap} {e.next_stop.compound}"
                    if e.next_stop
                    else "-",
                    "median_s": float(np.median(times[i])),
                    "mean_s": float(times[i].mean()),
                    "delta_s": float(d.mean()),
                    "delta_se_s": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                    "p_faster": float((d < 0).mean()),
                    "cvar10_delta_s": float(np.mean(d[d >= np.quantile(d, 1 - CVAR_TAIL)])),
                    "regret_s": float(r.mean()),
                    "cvar10_regret_s": float(np.mean(r[r >= np.quantile(r, 1 - CVAR_TAIL)])),
                    "cvar10_s": float(
                        np.mean(times[i][times[i] >= np.quantile(times[i], 1 - CVAR_TAIL)])
                    ),
                    "is_baseline": i == self.baseline_index,
                }
            )

        out = pd.DataFrame(rows).sort_values("mean_s").reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out.head(top) if top else out

    def indistinguishable_from_best(self, n_se: float = SEPARATION_SE) -> pd.DataFrame:
        """Candidates the simulation cannot separate from the recommendation.

        A ranking that presents its first row as the answer when the second is within
        Monte Carlo noise of it is overstating what was measured. The honest output is
        "these are equivalent, pick on other grounds".
        """
        table = self.table()
        best = table.iloc[0]
        delta = table["mean_s"] - best["mean_s"]
        # Standard error of the difference between two paired candidates is bounded above
        # by the sum of their own, which avoids re-deriving every pairwise covariance.
        se = table["delta_se_s"] + best["delta_se_s"]
        return table[delta <= n_se * se]

    def to_dict(self, top: int | None = None) -> dict:
        """Precomputed-scenario JSON for the showcase front-end (D9)."""
        table = self.table(top)
        return {
            "race_id": self.race_id,
            "circuit": self.circuit,
            "ego": self.ego,
            "from_lap": self.from_lap,
            "to_lap": self.to_lap,
            "n_sims": self.n_sims,
            "seed": self.seed,
            # A number produced under frozen rivals means something different from one
            # produced under simulated rivals; the tier travels with it to the UI.
            "rival_policy": self.rival_policy,
            "caution_laps": sorted(self.caution_laps),
            "objective": "expected_race_time_s",
            "baseline": self.baseline.label,
            "recommendation": self.best.label,
            "n_candidates": len(self.evaluations),
            "wall_time_s": self.wall_time_s,
            "candidates": [
                {**s.to_dict(), **row}
                for s, row in zip(
                    [self._by_label()[r["label"]] for _, r in table.iterrows()],
                    table.to_dict(orient="records"),
                    strict=True,
                )
            ],
        }

    def _by_label(self) -> dict[str, Scenario]:
        return {e.label: e.scenario for e in self.evaluations}


def evaluate(
    state: RaceState,
    ego: str,
    candidates: Sequence[Scenario],
    components: SimComponents,
    *,
    rival_plans: dict[str, tuple[PitInstruction, ...]] | None = None,
    n_sims: int = 1000,
    seed: int = config.SEED,
    baseline: Scenario | None = None,
    caution_laps: frozenset[int] | None = None,
    rival_policy: RivalPolicy = "sim_pace",
    align: RandomAlignment | None = None,
) -> StrategyRanking:
    """Simulate every candidate against common draws and rank them on expected race time.

    Args:
        state: The race at lap L. Built behind the temporal firewall; nothing after L is
            visible to it or to anything downstream.
        ego: Driver the recommendation is for.
        candidates: Pit plans to compare. Generated by ``strategy.candidates``, but any
            legal set works -- an audit compares two hand-written plans through here.
        components: LORO-scoped pace model, pit-loss sample and noise model.
        rival_plans: What everyone else does. Under D5 no driver's lap time depends on
            another's, so this does not move ``ego``'s race time at all; it is carried so
            that the same call keeps working when product-Phase 2 couples them.
        baseline: Plan the deltas are measured against. Defaults to the ranked best, which
            reads as "time lost against the recommendation".
        caution_laps: Assumed safety-car laps, applied identically to every candidate.
            **Exogenous** (Challenge 3): an assumption to vary, never a prediction.
        align: Draw sizes, computed here if omitted. Supplying one lets a caller evaluate
            several batches -- a coarse pass and a refinement -- and still compare across
            them, because every batch then consumes the same stream.

    Returns:
        A ``StrategyRanking``. Its ``evaluations`` are in the order given, not ranked;
        ``table()`` does the ranking.
    """
    if not candidates:
        raise ValueError(f"no candidates to evaluate for {ego} at lap {state.lap}")
    if not state.by_code[ego].is_running:
        raise ValueError(f"{ego} is not running at lap {state.lap}")

    rivals = {d: p for d, p in (rival_plans or {}).items() if d != ego}
    plan_sets = [{**rivals, ego: s.plan} for s in candidates]
    align = align or alignment_for(plan_sets, state.lap)

    started = time.perf_counter()
    evaluations = tuple(
        StrategyEvaluation(
            scenario=scenario,
            race_time_s=simulate_field(
                state,
                plans,
                components,
                n_sims=n_sims,
                seed=seed,
                to_lap=state.total_laps,
                caution_laps=caution_laps,
                rival_policy=rival_policy,
                align=align,
            ).elapsed_s(ego),
        )
        for scenario, plans in zip(candidates, plan_sets, strict=True)
    )
    wall_time_s = time.perf_counter() - started

    return _rank(
        state, ego, evaluations, n_sims, seed, rival_policy, caution_laps, baseline, wall_time_s
    )


def recommend(
    state: RaceState,
    ego: str,
    components: SimComponents,
    *,
    rival_plans: dict[str, tuple[PitInstruction, ...]] | None = None,
    n_sims: int = 1000,
    seed: int = config.SEED,
    max_stops: int = DEFAULT_MAX_STOPS,
    refine_top: int = DEFAULT_REFINE_TOP,
    baseline: Scenario | None = None,
    max_stint_share: dict[str, float] | None = None,
    caution_laps: frozenset[int] | None = None,
    rival_policy: RivalPolicy = "sim_pace",
) -> StrategyRanking:
    """Generate, prune, evaluate on a coarse grid, refine the leaders, and rank the union.

    The two passes share one seed and one ``RandomAlignment``, so the refined plans are
    paired with the coarse ones and the whole set ranks as a single comparison. Evaluating
    the refinements against fresh draws would make the last stage of the search the one
    stage whose numbers could not be compared with the others.

    Args:
        max_stint_share: LORO-scoped compound caps. Fit with the circuit excluded when
            evaluating one that is in the training pool; omitting it is in-sample.
        baseline: Deltas are measured against this plan, which is added to the candidate
            set if the grid did not already contain it. Pass the plan actually executed to
            turn the ranking into a post-race audit (D4).
    """
    share = max_stint_share or fit_max_stint_share()
    coarse = generate_candidates(
        state, ego, max_stops=max_stops, max_stint_share=share, rival_policy=rival_policy
    )
    if baseline is not None:
        coarse = _with_baseline(coarse, baseline)
        # The baseline is exempt from the physical filter on purpose -- an audit has to be
        # able to evaluate what actually happened, and what actually happened is sometimes
        # a stint longer than anything else in the store. But then its projected time is an
        # extrapolation, and every delta measured against it inherits that. Said out loud
        # rather than folded silently into a headline number.
        if not is_physically_plausible(state, ego, baseline.plan, share):
            log.warning(
                "baseline %r runs a stint past anything observed for its compound; "
                "its projected time extrapolates the degradation term and the deltas "
                "measured against it should not be read as time on the table",
                baseline.describe(),
            )
    if not coarse:
        raise ValueError(f"no legal, plausible plan exists for {ego} at lap {state.lap}")

    rivals = {d: p for d, p in (rival_plans or {}).items() if d != ego}
    # Fixed before either pass. Refinement only moves a stop's lap, never its count, so
    # the coarse set already bounds how wide the draws have to be.
    align = alignment_for([{**rivals, ego: s.plan} for s in coarse], state.lap)

    def run(batch: Sequence[Scenario]) -> StrategyRanking:
        return evaluate(
            state,
            ego,
            batch,
            components,
            rival_plans=rivals,
            n_sims=n_sims,
            seed=seed,
            caution_laps=caution_laps,
            rival_policy=rival_policy,
            align=align,
        )

    first = run(coarse)

    leaders = [
        e.scenario
        for e in sorted(first.evaluations, key=lambda e: e.race_time_s.mean())[:refine_top]
    ]
    refined = refine_candidates(state, ego, leaders, max_stint_share=share, exclude=coarse)
    if not refined:
        return _rebase(first, baseline)

    second = run(refined)
    merged = first.evaluations + second.evaluations
    log.info("evaluated %d coarse + %d refined candidates", len(coarse), len(refined))
    return _rank(
        state,
        ego,
        merged,
        n_sims,
        seed,
        rival_policy,
        caution_laps,
        baseline,
        first.wall_time_s + second.wall_time_s,
    )


def _with_baseline(candidates: Sequence[Scenario], baseline: Scenario) -> list[Scenario]:
    """Add the baseline plan to the grid if it is not already on it.

    Deliberately unfiltered: the plan a car actually ran is a fact, and a post-race audit
    that refused to evaluate it because it broke a fitted cap would be useless. The caller
    warns when it does.
    """
    key = tuple((p.lap, p.compound) for p in baseline.plan)
    if any(tuple((p.lap, p.compound) for p in s.plan) == key for s in candidates):
        return list(candidates)
    return [*candidates, baseline]


def _rebase(ranking: StrategyRanking, baseline: Scenario | None) -> StrategyRanking:
    from dataclasses import replace

    return replace(ranking, baseline_index=_baseline_index(ranking.evaluations, baseline))


def _baseline_index(evaluations: Sequence[StrategyEvaluation], baseline: Scenario | None) -> int:
    """Index of the plan deltas are measured against; the ranked best when unspecified."""
    if baseline is None:
        return int(np.argmin([e.race_time_s.mean() for e in evaluations]))
    key = tuple((p.lap, p.compound) for p in baseline.plan)
    for i, e in enumerate(evaluations):
        if tuple((p.lap, p.compound) for p in e.scenario.plan) == key:
            return i
    raise ValueError(f"baseline {baseline.describe()!r} was not among the evaluated candidates")


def _rank(
    state: RaceState,
    ego: str,
    evaluations: tuple[StrategyEvaluation, ...],
    n_sims: int,
    seed: int,
    rival_policy: RivalPolicy,
    caution_laps: frozenset[int] | None,
    baseline: Scenario | None,
    wall_time_s: float,
) -> StrategyRanking:
    return StrategyRanking(
        race_id=state.race_id,
        circuit=state.circuit,
        ego=ego,
        from_lap=state.lap,
        to_lap=state.total_laps,
        n_sims=n_sims,
        seed=seed,
        rival_policy=rival_policy,
        caution_laps=frozenset(caution_laps or ()),
        evaluations=evaluations,
        baseline_index=_baseline_index(evaluations, baseline),
        wall_time_s=wall_time_s,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

PRINTED_COLUMNS = (
    "rank",
    "label",
    "n_stops",
    "median_s",
    "delta_s",
    "delta_se_s",
    "p_faster",
    "cvar10_delta_s",
    "cvar10_regret_s",
)


def _loro_components(circuit: str) -> tuple[object, SimComponents]:
    """Pace model and simulation constants with ``circuit`` held out.

    Imported from the evaluation layer rather than reimplemented: a demo run on a circuit
    the model trained on would flatter every number in the table, and the strategy engine
    is the last place that should be quietly in-sample. (Phase 12's audit runs the same
    way, which is why this is worth the slightly odd import direction.)
    """
    from ..evaluation.replay import loro_pace_model

    model = loro_pace_model(circuit)
    return model, SimComponents.fit(model, exclude_circuit=circuit)


def main(argv: list[str] | None = None) -> int:
    from ..data import store
    from ..sim.scenario import observed_plans
    from ..sim.state import build_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", default=config.DEMO_CIRCUIT_KEY)
    parser.add_argument("--lap", type=int, default=20, help="decision lap L")
    parser.add_argument("--driver", default=None, help="defaults to the leader at lap L")
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--max-stops", type=int, default=DEFAULT_MAX_STOPS)
    parser.add_argument("--top", type=int, default=12, help="rows to print")
    parser.add_argument(
        "--baseline",
        choices=("best", "observed"),
        default="observed",
        help="measure deltas against the ranked best, or against what was actually run",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    laps = store.race_laps(args.circuit)
    total_laps = store.race_total_laps(args.circuit)
    model, components = _loro_components(args.circuit)
    state = build_state(laps, lap=args.lap, total_laps=total_laps, pace_model=model)

    ego = args.driver or state.order()[0].code
    driver = state.by_code[ego]
    plans = observed_plans(laps, after_lap=args.lap)
    baseline = (
        Scenario(ego=ego, plan=plans.get(ego, ()), label="as executed")
        if args.baseline == "observed"
        else None
    )

    ranking = recommend(
        state,
        ego,
        components,
        rival_plans=plans,
        n_sims=args.n_sims,
        max_stops=args.max_stops,
        baseline=baseline,
        max_stint_share=fit_max_stint_share(exclude_circuit=args.circuit),
    )

    print(f"\n=== {args.circuit} lap {args.lap}/{total_laps}, {ego} ({driver.team}) ===")
    print(
        f"P{state.position_of(ego)}, {driver.compound} age {driver.tyre_age}, "
        f"{driver.pit_stops} stop(s) made, used {sorted(driver.compounds_used)}, "
        f"reference pace {driver.reference_pace_s:.3f}s "
        f"from {driver.clean_laps_observed} clean laps"
    )
    print(f"constants and pace model fitted with {args.circuit} held out (leave-one-race-out)")

    print(f"\n=== {len(ranking.evaluations)} candidates, ranked on expected race time (D5) ===")
    table = ranking.table(args.top)
    print(table[list(PRINTED_COLUMNS)].round(3).to_string(index=False))
    print(
        f"\ndelta_s and p_faster are measured against '{ranking.baseline.label}'; "
        "regret is against whichever plan won that same simulation."
    )

    close = ranking.indistinguishable_from_best()
    print(f"\n=== recommendation: {ranking.best.label} ===")
    if len(close) > 1:
        print(
            f"{len(close)} plans sit within {SEPARATION_SE:.0f} Monte Carlo standard errors "
            "of each other and are not separated by this simulation:"
        )
        print("  " + "; ".join(close["label"].tolist()))
    print(
        "delta_se_s is Monte Carlo error only. Simulator accuracy is the Phase 7 replay\n"
        "result (~0.5% of the projected window) and is larger; more draws do not shrink it."
    )

    per_20 = ranking.wall_time_s / len(ranking.evaluations) * 20
    ok = per_20 <= GATE_SECONDS_PER_20_CANDIDATES
    print(
        f"\n=== Phase 9 exit criterion: 20 candidates < {GATE_SECONDS_PER_20_CANDIDATES:.0f}s ==="
    )
    print(
        f"{len(ranking.evaluations)} candidates x {args.n_sims} sims in "
        f"{ranking.wall_time_s:.2f}s -> {per_20:.2f}s per 20"
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1
