"""Phase 10: why the engine recommends what it recommends -- as arithmetic.

    uv run python -m apexstrategy.strategy --circuit hungary --lap 20 --explain

The brief asks for explanations and suggests SHAP. SHAP on the pace model explains *the
pace model*, which is not what anyone was asking about: the recommendation comes out of
the simulator, and the pace model is one of its inputs. An attribution method applied to
the wrong artefact is worse than no explanation, because it is persuasive.

So the delta is decomposed directly (Challenge 6). The simulator adds every effect in
seconds, so the pieces are already there -- ``LapBreakdown`` keeps the addends and this
module differences them::

    Hungary, lap 20, VER -- two-stop against one-stop:   -7.01 s

      fresher rubber        -31.20 s
      compound choice        -0.63 s
      pit-lane time         +24.30 s
      sampling residue       +0.53 s
      ─────────────────────────────
      net                    -7.01 s

The extra stop costs 24.3 s in the pit lane and buys 31.2 s of fresher rubber. The sum is
not a fit or an approximation; it is the same arithmetic the simulator did, regrouped.

## Why the reference pace and the race-progress trend do not appear

Because they cancel, exactly. Both plans run the same laps of the same race, so the
driver's reference level and the in-race progress correction enter both totals identically
and vanish in the subtraction. That is worth knowing rather than hiding: the two largest
terms in an *absolute* projection contribute nothing to a *comparison*, which is the same
reason Phase 9 pairs the draws. What is left is genuinely about the strategy.

## The pace difference is split without asking the model how it works

A per-lap pace difference mixes two causes: the plans put a different *compound* on the
car and a different *tyre age* under it. Splitting them by reading the fitted coefficients
would work today -- ``pace_delta ~ C(compound) + C(compound):tyre_age + lap_number`` is
about as separable as a model gets -- and would silently become wrong the moment a model
with interactions replaced it.

Instead both orderings are evaluated and averaged, which needs nothing but ``predict``::

    age      = ½[ mu(cA, ageA) - mu(cA, ageB) ] + ½[ mu(cB, ageA) - mu(cB, ageB) ]
    compound = ½[ mu(cA, ageA) - mu(cB, ageA) ] + ½[ mu(cA, ageB) - mu(cB, ageB) ]

For two factors that is the Shapley value, and it sums to ``mu(cA, ageA) - mu(cB, ageB)``
identically -- not approximately, but as an algebraic identity you can check by eye.

## The sampling row is the honesty check, not a component

The hierarchical noise is mean zero, so its contribution to a difference between two plans
is zero in expectation and Monte Carlo residue in practice. Reporting it is what stops the
decomposition from quietly attributing residue to a cause: **if the sampling row is not
small against the headline, the headline is not settled and the answer is more draws.**

It is not always small. Two plans with different stop counts are exposed to different
numbers of tyre sets, and the stint-level effect on the extra set does not cancel — the
same irreducible 14.9 s of spread ``RandomAlignment`` documents. At 4,000 draws that
leaves a few tenths of a second on the row, which is why the default here is far above the
1,000 the ranking uses.

## Terms this decomposition does not contain

Stated rather than omitted, because the doc's own worked example contains both:

- **Traffic on rejoin.** Not modelled at all (D5). The replay validation measures the cost
  of the omission at -0.07 s per dirty-air lap, about -11% of total replay error, and it
  is one-sided: projections are optimistic for a car that would have been stuck behind
  someone. A plan that rejoins into traffic is flattered by every number here.
- **Safety-car option value.** Cautions are exogenous and never predicted (Challenge 3),
  so there is no distribution to take an expectation over. What can honestly be produced
  is a *contrast* under an assumed caution -- ``caution_option_value`` -- and it inherits
  two thin constants: the caution lap multiplier rests on a single VSC, and the caution
  pit multiplier is assumed rather than fitted because the store contains no
  green-flag-to-caution stop pairs at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..sim.engine import RandomAlignment, SimComponents, alignment_for, build_schedule
from ..sim.engine import simulate_field as _simulate
from ..sim.scenario import PitInstruction, RivalPolicy, Scenario
from ..sim.state import RaceState

#: Columns that describe the *set* on the car, as opposed to which compound it is. They
#: move together in the counterfactual swap because they are all derived from when the
#: tyre was fitted -- pulling ``tyre_age`` across while leaving ``is_fresh_tyre`` behind
#: would build a frame describing a car that cannot exist.
AGE_COLUMNS = ("tyre_age", "is_fresh_tyre", "stint_number")

#: How far the reconstructed components may miss the simulated total before it counts as a
#: defect rather than float noise. Double precision over ~50 laps x 4,000 draws leaves far
#: less than this; anything larger means a term was dropped.
RECONCILIATION_TOLERANCE_S = 1e-6

#: Draws for an explanation. Well above the ranking's 1,000 because the sampling residue
#: falls as 1/sqrt(n) and it is the row that decides whether the rest can be believed.
DEFAULT_EXPLAIN_SIMS = 4000

#: Measured in the Phase 7 replay decomposition, and quoted here so the omission travels
#: with the explanation rather than living only in a design document.
TRAFFIC_BIAS_S_PER_DIRTY_AIR_LAP = -0.07


@dataclass(frozen=True)
class Explanation:
    """A strategy delta split into causes that sum back to it.

    Sign convention follows the rest of the engine: negative is faster, so a negative
    ``total_s`` means ``plan`` beats ``baseline`` and a negative component is a term
    working in the plan's favour.
    """

    ego: str
    circuit: str
    from_lap: int
    to_lap: int
    plan: Scenario
    baseline: Scenario
    n_sims: int
    seed: int
    caution_laps: frozenset[int]
    #: Mean of ``T(plan) - T(baseline)`` over the paired draws. The headline.
    total_s: float
    #: Running fresher or older rubber, holding compound fixed. Usually the big one.
    tyre_age_s: float
    #: Running a different compound, holding tyre age fixed.
    compound_s: float
    #: Stops made under one plan and not the other, charged at the bootstrapped loss.
    pit_lane_s: float
    #: Mean-zero in expectation; its size is how unsettled the headline is.
    sampling_s: float

    @property
    def residual_s(self) -> float:
        """What the components fail to explain. Zero to float precision, or it is a bug."""
        return self.total_s - (
            self.tyre_age_s + self.compound_s + self.pit_lane_s + self.sampling_s
        )

    @property
    def is_settled(self) -> bool:
        """Whether the sampling residue is small enough for the split to be worth reading."""
        return abs(self.sampling_s) <= 0.25 * abs(self.total_s) or abs(self.sampling_s) < 0.1

    def table(self) -> pd.DataFrame:
        """The decomposition, with the total as its last row so the sum is visible."""
        rows = [
            ("fresher rubber", self.tyre_age_s),
            ("compound choice", self.compound_s),
            ("pit-lane time", self.pit_lane_s),
            ("sampling residue", self.sampling_s),
            ("net", self.total_s),
        ]
        out = pd.DataFrame(rows, columns=["component", "seconds"])
        out["share_of_net"] = out["seconds"] / self.total_s if self.total_s else np.nan
        out.loc[out["component"] == "net", "share_of_net"] = 1.0
        return out

    def to_dict(self) -> dict:
        """JSON for the showcase front-end (D9), omissions included rather than implied."""
        return {
            "ego": self.ego,
            "circuit": self.circuit,
            "from_lap": self.from_lap,
            "to_lap": self.to_lap,
            "plan": self.plan.to_dict(),
            "baseline": self.baseline.to_dict(),
            "n_sims": self.n_sims,
            "seed": self.seed,
            "caution_laps": sorted(self.caution_laps),
            "total_s": self.total_s,
            "components": {
                "tyre_age_s": self.tyre_age_s,
                "compound_s": self.compound_s,
                "pit_lane_s": self.pit_lane_s,
                "sampling_s": self.sampling_s,
            },
            "residual_s": self.residual_s,
            "is_settled": self.is_settled,
            "not_included": [
                {
                    "term": "traffic on rejoin",
                    "why": "not modelled (D5); projections are clean-air times",
                    "measured_bias_s_per_dirty_air_lap": TRAFFIC_BIAS_S_PER_DIRTY_AIR_LAP,
                    "direction": "optimistic",
                },
                {
                    "term": "safety-car option value",
                    "why": "cautions are exogenous and never predicted (Challenge 3)",
                    "available_as": "caution_option_value(), under an assumed caution",
                },
            ],
        }


def explain(
    state: RaceState,
    ego: str,
    plan: Scenario,
    baseline: Scenario,
    components: SimComponents,
    *,
    rival_plans: dict[str, tuple[PitInstruction, ...]] | None = None,
    n_sims: int = DEFAULT_EXPLAIN_SIMS,
    seed: int = config.SEED,
    caution_laps: frozenset[int] | None = None,
    rival_policy: RivalPolicy = "sim_pace",
    align: RandomAlignment | None = None,
) -> Explanation:
    """Decompose ``plan`` minus ``baseline`` into causes that sum back to the difference.

    Both plans are simulated against common random draws, as everywhere else in this
    layer: without the pairing the reference-pace and race-progress terms would not cancel
    draw-by-draw and the decomposition would be dominated by shared luck.

    Raises:
        ValueError: if the components fail to reconstruct the simulated total. That is a
            dropped term, not a tolerance to widen.
    """
    rivals = {d: p for d, p in (rival_plans or {}).items() if d != ego}
    plans = [{**rivals, ego: s.plan} for s in (plan, baseline)]
    align = align or alignment_for(plans, state.lap)

    runs = [
        _simulate(
            state,
            p,
            components,
            n_sims=n_sims,
            seed=seed,
            to_lap=state.total_laps,
            caution_laps=caution_laps,
            rival_policy=rival_policy,
            align=align,
            capture=True,
        )
        for p in plans
    ]
    a, b = runs
    i = a.drivers.index(ego)
    assert a.breakdown is not None and b.breakdown is not None  # capture=True

    total_s = float((a.elapsed_s(ego) - b.elapsed_s(ego)).mean())
    pit_lane_s = float((a.breakdown.pit_s[:, i].sum(1) - b.breakdown.pit_s[:, i].sum(1)).mean())
    sampling_s = float((a.breakdown.noise_s[:, i].sum(1) - b.breakdown.noise_s[:, i].sum(1)).mean())

    tyre_age_s, compound_s = _split_pace(
        state, ego, plan, baseline, rivals, components, caution_laps
    )

    out = Explanation(
        ego=ego,
        circuit=state.circuit,
        from_lap=state.lap,
        to_lap=state.total_laps,
        plan=plan,
        baseline=baseline,
        n_sims=n_sims,
        seed=seed,
        caution_laps=frozenset(caution_laps or ()),
        total_s=total_s,
        tyre_age_s=tyre_age_s,
        compound_s=compound_s,
        pit_lane_s=pit_lane_s,
        sampling_s=sampling_s,
    )
    if abs(out.residual_s) > RECONCILIATION_TOLERANCE_S:
        raise ValueError(
            f"decomposition does not reconcile: {out.residual_s:+.3e}s unexplained of "
            f"{total_s:+.3f}s. A term is missing from the split, not from the tolerance."
        )
    return out


def _split_pace(
    state: RaceState,
    ego: str,
    plan: Scenario,
    baseline: Scenario,
    rivals: dict[str, tuple[PitInstruction, ...]],
    components: SimComponents,
    caution_laps: frozenset[int] | None,
) -> tuple[float, float]:
    """Shapley split of the modelled pace difference into tyre age and compound.

    The two halves sum to ``mu_a - mu_b`` as an algebraic identity, so the reconciliation
    check in ``explain`` is really testing that the *simulator's* pace term equals this
    one -- that is, that the reference level and the race-progress trend cancelled between
    the plans as claimed. They do, and if they ever stop, this is what notices.
    """
    from ..models.pace import feature_frame

    sched_a, sched_b = (
        _ego_schedule(state, {**rivals, ego: s.plan}, ego) for s in (plan, baseline)
    )
    predict = components.pace_model.predict

    mu_aa = predict(feature_frame(sched_a))
    mu_bb = predict(feature_frame(sched_b))
    # Each plan's compound carrying the other's tyre state, which is the counterfactual
    # the Shapley average needs and never a car that could exist on track.
    mu_a_with_b_age = predict(feature_frame(_swap(sched_a, sched_b, AGE_COLUMNS)))
    mu_b_with_a_age = predict(feature_frame(_swap(sched_b, sched_a, AGE_COLUMNS)))

    age = 0.5 * ((mu_aa - mu_a_with_b_age) + (mu_b_with_a_age - mu_bb))
    compound = 0.5 * ((mu_aa - mu_b_with_a_age) + (mu_a_with_b_age - mu_bb))

    # A caution scales the whole lap, the model's mean included, so the split has to be
    # scaled the same way or it will not reconcile with what the simulator charged.
    weight = _caution_multiplier(state, sched_a["lap_number"].to_numpy(), components, caution_laps)
    return float((age * weight).sum()), float((compound * weight).sum())


def _ego_schedule(
    state: RaceState, plans: dict[str, tuple[PitInstruction, ...]], ego: str
) -> pd.DataFrame:
    sched = build_schedule(state, plans, state.total_laps)
    return sched[sched["driver"] == ego].sort_values("lap_number").reset_index(drop=True)


def _swap(base: pd.DataFrame, other: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = base.copy()
    for column in columns:
        out[column] = other[column].to_numpy()
    return out


def _caution_multiplier(
    state: RaceState,
    laps: np.ndarray,
    components: SimComponents,
    caution_laps: frozenset[int] | None,
) -> np.ndarray:
    weight = np.ones(len(laps))
    if caution_laps:
        weight[np.isin(laps, sorted(caution_laps))] = components.caution_lap_multiplier
    return weight


def caution_option_value(
    state: RaceState,
    ego: str,
    plan: Scenario,
    baseline: Scenario,
    components: SimComponents,
    caution_laps: frozenset[int],
    **kwargs,
) -> float:
    """How much the plan's advantage changes if a caution appears on ``caution_laps``.

    Negative means the caution helps this plan relative to the baseline -- the "SC option
    value" of the architecture doc's worked example, produced honestly.

    **Read the caveats before quoting the number.** Cautions are exogenous by design and
    are never predicted (Challenge 3), so this is a contrast under an assumption the caller
    supplies, not an expectation over anything. Worse, both constants it leans on are the
    thinnest in the project: the caution lap multiplier was fitted on a single VSC, and the
    caution pit multiplier is *assumed*, because the store contains no
    green-flag-to-caution stop pairs to fit it from. The sign of this number is more
    trustworthy than its size, and it needs multi-year data before it is more than
    indicative.
    """
    under = explain(state, ego, plan, baseline, components, caution_laps=caution_laps, **kwargs)
    green = explain(state, ego, plan, baseline, components, caution_laps=None, **kwargs)
    return under.total_s - green.total_s
