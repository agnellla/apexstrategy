"""Phase 7: the Monte Carlo race simulator.

Given a ``RaceState`` at lap L and a pit plan per driver, project every remaining lap and
return a distribution over race times. The strategy engine (Phase 9) is a wrapper around
this; the replay validation (``evaluation.replay``) is the gate it has to pass first.

## Why there is no lap loop

The architecture doc specifies a loop over laps, vectorised over sims, because steps 4-5
of the per-lap update (dirty air, overtake resolution) couple drivers together and must
be resolved in track order. D5 defers both to product-Phase 2, and with them gone **no
driver's lap time depends on any other driver's**. The tyre schedule is deterministic
given the plan, so the entire race is one vectorised expression over
``(n_sims, n_drivers, n_laps)`` and the lap loop buys nothing but time.

This is a consequence of a scoping decision, not a shortcut, and it reverses when
overtaking arrives: ``build_schedule`` already produces the per-lap frame the loop would
walk, so restoring one is local to ``simulate_field``.

## What is modelled here, and what is deliberately not

| Effect | Treatment |
|---|---|
| Pace, degradation, race progress | Pace model (Phase 5) for the mean |
| Uncertainty | Hierarchical variance components (Phase 8, ``sim.noise``) |
| Pit stop cost | Bootstrapped from observed green-flag stops, split across in/out lap |
| Safety car / VSC laps | **Exogenous input.** Never predicted (Challenge 3) |
| Traffic / dirty air | **Not modelled** (D5). Projections are clean-air times |
| Overtaking, positions | **Not modelled** (D5). Cars pass freely on cumulative time |

The traffic omission is the single largest known bias in the MVP, and it is one-sided:
projected race times are optimistic for a car that would really have been stuck behind
someone. Rather than caveat it in prose, ``evaluation.replay`` measures it -- the replay
error is decomposed by lap category, so the cost of the omission is a number.

## Where the uncertainty comes from

Phase 7 drew the spread from the pace model's own ``predict_dist``, which bootstraps
per-lap residuals independently. Over a 40-lap horizon those average out and race-level
80% intervals covered 11-28% of outcomes -- the variance collapse of Challenge 5.

Phase 8 replaced it with ``sim.noise.NoiseModel``, which samples each effect at the level
it lives at, and coverage moved to 0.81. The pace model is now called for its *mean* and
the noise model for the spread; ``predict_dist`` remains the fallback when no noise model
is supplied, purely so the Phase 7 result stays reproducible. Point estimates are
identical either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from ..data import store
from .noise import NoiseModel
from .scenario import PitInstruction, RivalPolicy
from .state import RaceState


class PaceModel(Protocol):
    """All the simulator needs from a pace model.

    Structural rather than a base class, so ``models.pace`` does not have to import from
    ``sim`` to satisfy it and a caller can pass a stub without inheriting anything. It also
    documents the seam D11 is fought over: any model that beats the linear baseline drops
    in here and nowhere else.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_dist(self, X: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray: ...


#: Fallback share of a stop's cost charged to the in-lap rather than the out-lap, used
#: only when there is no model to fit the split against. ``fit_pit_loss`` measures it.
DEFAULT_PIT_IN_LAP_SHARE = 0.18

#: Lap-time multiplier while a caution is out. Measured at 1.21 on the 27 caution laps in
#: the store -- all of them Austria's late VSC, so this is a thin estimate on one event.
#: It is close to inert for the current circuit set (0.2% of laps) and becomes
#: load-bearing only when Phase 9 starts varying safety-car assumptions.
DEFAULT_CAUTION_LAP_MULTIPLIER = 1.21

#: A stop taken under caution costs a fraction of a green-flag one, because the whole
#: field is slowed while the stopping car loses only the pit lane. **This is an assumed
#: value, not a fitted one:** the store contains no green-flag-to-caution stop pairs to
#: fit it from (Monaco's 16 stops were taken during a red-flag stoppage, where the cost is
#: zero for a different reason). Flagged rather than quietly fitted on unsuitable data.
ASSUMED_CAUTION_PIT_MULTIPLIER = 0.45


def fit_pit_loss(laps: pd.DataFrame, pace_model: PaceModel) -> tuple[np.ndarray, float]:
    """Measure pit loss *in the simulator's own currency*, and how it splits across laps.

    Phase 4's ``estimate_pit_loss`` charges a stop against the driver's local median clean
    pace. The simulator charges it against ``reference_pace + pace_model.predict(lap)``,
    which is not the same reference: the out-lap runs a tyre at age 1, and the model
    expects a fresh tyre to be fast. Reusing the Phase 4 number therefore credits the
    out-lap with a fresh-tyre bonus it never actually enjoyed, and under-charges the stop.

    Measured, that mismatch was worth **-1.7s on every out-lap** in the first replay run
    -- the single largest error contributor, and a bug in the coupling rather than in
    either estimate on its own. Fitting the loss as the residual over what the model
    predicts for those two laps makes the two definitions consistent by construction::

        excess_in  = t_in  - (reference + model(in-lap))
        excess_out = t_out - (reference + model(out-lap))
        loss       = excess_in + excess_out

    Returns the per-stop losses and the in-lap share of the median stop.
    """
    from ..data.clean import is_under_caution
    from ..data.schema import DRY_COMPOUNDS
    from ..models.pace import add_reference_pace, feature_frame

    laps = add_reference_pace(laps).sort_values(["race_id", "driver", "lap_number"])
    laps = laps[laps["reference_pace_s"].notna() & laps["lap_time_s"].notna()]
    # Dry only, before predicting rather than after. The pace model has never seen an
    # INTERMEDIATE or WET compound level, so handing it one fails inside statsmodels with
    # an unhelpful error. True today only because the training circuits happen to be dry;
    # stated here so adding a wet-phase circuit does not turn it into a puzzle.
    laps = laps[laps["compound"].isin(DRY_COMPOUNDS)]
    laps = laps.assign(
        excess_s=laps["lap_time_s"]
        - laps["reference_pace_s"]
        - pace_model.predict(feature_frame(laps.assign(tyre_age=laps["tyre_age"].fillna(1.0))))
    )

    key = ["race_id", "driver"]
    laps["next_excess_s"] = laps.groupby(key)["excess_s"].shift(-1)
    laps["next_is_pit_out"] = laps.groupby(key)["is_pit_out"].shift(-1)
    laps["next_compound"] = laps.groupby(key)["compound"].shift(-1)

    stops = laps[
        laps["is_pit_in"]
        & laps["next_is_pit_out"].astype("boolean").fillna(False).astype(bool)
        & laps["next_excess_s"].notna()
        & ~is_under_caution(laps["track_status"])
        & laps["compound"].isin(DRY_COMPOUNDS)
        & laps["next_compound"].isin(DRY_COMPOUNDS)
    ]
    losses = (stops["excess_s"] + stops["next_excess_s"]).to_numpy()
    share = float(np.median(stops["excess_s"]) / np.median(losses))
    return losses, share


@dataclass(frozen=True)
class RandomAlignment:
    """Declared draw sizes that make the random stream independent of the pit plan.

    Two strategies are compared by the *difference* in their race times, and that
    difference only carries information if both were run against the same luck.
    Everything the two share -- how fast the track came to the field today, how far the
    reference pace was off, whether this was a good set of tyres -- then cancels in the
    subtraction, and what survives is the strategy.

    Measured on Hungary from lap 20 over 2,000 sims, a lap-30 stop against a lap-36 stop:
    the difference has sd **2.9 s** under common draws against **45.7 s** under
    independent ones, on a mean difference of 8.9 s. Independent draws call a
    nine-second gap a coin flip.

    Passing both runs the same seed is not sufficient on its own. ``NoiseModel.sample``
    draws one block per *stint* and ``_apply_pit_losses`` one per *stop*, so a plan with
    more stops than any rival draws a larger array, leaves the generator in a different
    place, and every component after it -- including the per-lap noise -- decorrelates.
    Same race and lap, a one-stop against a two-stop with the field frozen: sd **25.5 s**
    on the same seed, **14.9 s** once the sizes are declared.

    That 14.9 s does not go away with better bookkeeping, and should not. A two-stop plan
    is genuinely exposed to one more set of tyres than a one-stop, and the stint-level
    effect on that extra set is a real risk the plans do not share. Pairing removes the
    luck the two strategies have in common; what it leaves is the part that is actually
    about the strategy, which is what the CVaR column then measures.

    Both blocks are drawn at the largest size any compared plan needs and indexed down, so
    the stream depends on the state and the seed alone. ``None`` leaves the Phase 7/8 draw
    order untouched, which is why the replay and calibration numbers stay reproducible.
    """

    #: Stints per driver: the one running at lap L plus one started by each future stop.
    stint_slots: int
    #: Future stops per driver.
    stop_slots: int


def alignment_for(
    plan_sets: Iterable[dict[str, tuple[PitInstruction, ...]]], from_lap: int
) -> RandomAlignment:
    """Sizes wide enough for every plan set that is going to be compared with the others.

    Sized over *all* drivers, not just the one whose strategy varies: the noise blocks are
    rectangular over the field, so a rival with three stops sets the floor as surely as
    the ego does.
    """
    stops = max(
        (
            sum(1 for p in plan if p.lap > from_lap)
            for plans in plan_sets
            for plan in plans.values()
        ),
        default=0,
    )
    return RandomAlignment(stint_slots=stops + 1, stop_slots=stops)


@dataclass(frozen=True)
class SimComponents:
    """Everything the simulator needs beyond the state and the plans.

    Fitted through ``SimComponents.fit(exclude_circuit=...)`` so that a replay of circuit
    X never consumes constants estimated from circuit X. Circuit constants are model
    parameters and leak exactly like model weights do (Part 13, risk 5).
    """

    pace_model: PaceModel
    #: Empirical green-flag pit losses, bootstrapped rather than fitted to a distribution:
    #: the tail of slow stops is the part that matters for risk and a Gaussian loses it.
    pit_loss_s: np.ndarray
    pit_in_lap_share: float = DEFAULT_PIT_IN_LAP_SHARE
    caution_lap_multiplier: float = DEFAULT_CAUTION_LAP_MULTIPLIER
    caution_pit_multiplier: float = ASSUMED_CAUTION_PIT_MULTIPLIER
    #: Phase 8 hierarchical noise. ``None`` falls back to the model's own i.i.d.
    #: ``predict_dist``, which is the Phase 7 behaviour and is badly under-dispersed at
    #: race level -- kept so the Phase 7 gate stays reproducible and the A/B is one flag.
    noise: NoiseModel | None = None

    @classmethod
    def fit(
        cls, pace_model: PaceModel, exclude_circuit: str | None = None, **overrides
    ) -> SimComponents:

        # Race-level admission, not circuit-level. Filtering by circuit let a dry running
        # of a venue admit a wet one: Sao Paulo 2024 (100% wet), Montreal 2024 (66.5%),
        # Zandvoort 2023 (24.4%) and Silverstone 2024 (24.3%) were all entering the
        # pit-loss and noise fits, which is exactly the contamination D10 excluded by hand
        # and which put Silverstone's pit loss at 41.4 s against 19.3 s dry.
        laps = store.fittable_laps()
        laps = laps[store.usable_races(laps)]
        if exclude_circuit is not None:
            laps = laps[laps["circuit"] != exclude_circuit]

        losses, share = fit_pit_loss(laps, pace_model)
        if losses.size < 20:
            raise ValueError(f"only {losses.size} pit stops available after exclusions")
        return cls(
            pace_model=pace_model,
            pit_loss_s=losses,
            **{
                "pit_in_lap_share": share,
                "noise": NoiseModel.fit(laps, pace_model),
                **overrides,
            },
        )

    def sample_pit_loss(self, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.pit_loss_s, size=size, replace=True)


def build_schedule(
    state: RaceState,
    plans: dict[str, tuple[PitInstruction, ...]],
    to_lap: int,
) -> pd.DataFrame:
    """Expand each running driver's plan into a per-lap tyre schedule.

    Deterministic across simulations: a pit plan fixes which compound is on the car at
    every lap, and nothing sampled here changes it. That is what lets the whole race
    vectorise -- only the *time* of each lap is random, never the state.

    ``PitInstruction(lap=P)`` means the driver pits at the end of lap P, so lap P is the
    in-lap on the old tyre and lap P+1 is the out-lap on the new one at age 1. That
    matches the store's convention, where ``is_pit_in`` marks P and ``is_pit_out`` marks
    P+1 with ``tyre_age == 1``.
    """
    rows = []
    for d in state.running:
        stops = {p.lap: p.compound for p in plans.get(d.code, ()) if state.lap < p.lap <= to_lap}
        compound, age, stint = d.compound, d.tyre_age, d.stint_number

        for lap in range(state.lap + 1, to_lap + 1):
            fresh = lap - 1 in stops
            if fresh:
                compound, age, stint = stops[lap - 1], 1, stint + 1
            else:
                age += 1
            rows.append(
                {
                    "driver": d.code,
                    "lap_number": lap,
                    "compound": compound,
                    "tyre_age": float(age),
                    "stint_number": stint,
                    "is_fresh_tyre": fresh,
                    "is_pit_in": lap in stops,
                    "is_pit_out": fresh,
                    "circuit": state.circuit,
                    "track_temp": state.track_temp,
                    "air_temp": state.air_temp,
                }
            )
    return pd.DataFrame(rows).sort_values(["driver", "lap_number"]).reset_index(drop=True)


@dataclass(frozen=True)
class LapBreakdown:
    """The additive pieces every projected lap time was assembled from, in seconds.

    ``pace_s + noise_s + pit_s`` reconstructs ``lap_times_s`` to floating-point round-off
    (~1e-14 s), by construction rather than by regression: the simulator adds each effect
    in seconds, so keeping the addends is enough to explain the total. That is what makes
    a Phase 10 explanation *arithmetically true* (Challenge 6) rather than an attribution
    method's opinion about one. The round-off is the caution multiplier being applied to
    the sum and divided back out of the parts; making it bitwise would mean changing the
    order of operations in the default path, which is not worth 1e-14 s.

    Captured only on request -- three more ``(n_sims, n_drivers, n_laps)`` arrays is 24 MB
    at 1,000 sims, which is worth paying for the two plans being compared and not for the
    hundred being ranked.
    """

    #: ``multiplier * (reference + model mean + race-progress adjustment)``. Deterministic
    #: given the plan: everything the simulator claims to *know* about the lap.
    pace_s: np.ndarray
    #: ``multiplier * hierarchical noise``. Mean zero, so its contribution to a difference
    #: between two plans is Monte Carlo residue and a measure of how settled that
    #: difference is -- see ``strategy.explain``.
    noise_s: np.ndarray
    #: Pit-loss charged to the in-lap and out-lap of each stop.
    pit_s: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    """Sampled lap times for every running driver over the projected laps.

    ``lap_times_s`` has shape ``(n_sims, n_drivers, n_laps)`` and stays in memory; it is
    the raw material for ablation-based explanations (Phase 10). ``to_dict`` emits
    summary statistics only, because the showcase front-end is fed precomputed JSON (D9)
    and a 10 MB array is not that.
    """

    race_id: str
    circuit: str
    from_lap: int
    to_lap: int
    drivers: tuple[str, ...]
    lap_times_s: np.ndarray
    plans: dict[str, tuple[PitInstruction, ...]]
    seed: int
    rival_policy: RivalPolicy = "sim_pace"
    caution_laps: frozenset[int] = field(default_factory=frozenset)
    #: Present only when ``simulate_field(capture=True)``. The addends of ``lap_times_s``.
    breakdown: LapBreakdown | None = None

    @property
    def n_sims(self) -> int:
        return self.lap_times_s.shape[0]

    @property
    def laps(self) -> np.ndarray:
        return np.arange(self.from_lap + 1, self.to_lap + 1)

    def _index(self, driver: str) -> int:
        return self.drivers.index(driver)

    def elapsed_s(self, driver: str, through_lap: int | None = None) -> np.ndarray:
        """Projected elapsed time from ``from_lap`` through ``through_lap``, per sim."""
        through = self.to_lap if through_lap is None else through_lap
        n = through - self.from_lap
        if not 0 <= n <= self.to_lap - self.from_lap:
            raise ValueError(f"lap {through} is outside the projected window")
        return self.lap_times_s[:, self._index(driver), :n].sum(axis=1)

    def summary(self, through_lap: int | None = None) -> pd.DataFrame:
        """Per-driver projected elapsed time: median, interval, and downside risk."""
        rows = []
        for d in self.drivers:
            t = self.elapsed_s(d, through_lap)
            q10, q50, q90 = np.quantile(t, [0.1, 0.5, 0.9])
            rows.append(
                {
                    "driver": d,
                    "median_s": q50,
                    "p10_s": q10,
                    "p90_s": q90,
                    # Mean of the worst 10% of draws: "risk" as a number rather than a
                    # label. Slower is worse, so the risk tail is the upper one.
                    "cvar10_s": float(t[t >= q90].mean()),
                    "n_stops": len(self.plans.get(d, ())),
                }
            )
        return pd.DataFrame(rows).sort_values("median_s").reset_index(drop=True)

    def to_dict(self, through_lap: int | None = None) -> dict:
        return {
            "race_id": self.race_id,
            "circuit": self.circuit,
            "from_lap": self.from_lap,
            "to_lap": self.to_lap,
            "n_sims": self.n_sims,
            "seed": self.seed,
            # Travels with the result to the UI: a number produced under frozen rivals
            # means something different from one produced under simulated rivals.
            "rival_policy": self.rival_policy,
            "caution_laps": sorted(self.caution_laps),
            "plans": {
                d: [{"lap": p.lap, "compound": p.compound} for p in plan]
                for d, plan in self.plans.items()
            },
            "drivers": self.summary(through_lap).to_dict(orient="records"),
        }


def simulate_field(
    state: RaceState,
    plans: dict[str, tuple[PitInstruction, ...]],
    components: SimComponents,
    n_sims: int = 1000,
    seed: int = 0,
    to_lap: int | None = None,
    caution_laps: frozenset[int] | None = None,
    rival_policy: RivalPolicy = "sim_pace",
    align: RandomAlignment | None = None,
    capture: bool = False,
) -> SimulationResult:
    """Project every running driver to ``to_lap`` under their given plan.

    Args:
        state: The race at lap L. Nothing after lap L is visible to this function.
        plans: Pit plan per driver code. Stops at or before ``state.lap`` are ignored.
        components: Fitted pace model and pit-loss distribution, LORO-scoped by caller.
        n_sims: Monte Carlo draws. 1,000 gives ~1.6pp standard error on a probability
            near 0.5, which is well inside model error; 5,000 for final reports.
        seed: Recorded on the result. Identical inputs must give identical output.
        to_lap: Defaults to the scheduled race distance.
        caution_laps: Laps run under SC/VSC. **Exogenous** -- an assumption supplied by
            the caller or replayed from observation, never a prediction (Challenge 3).
        align: Declared draw sizes, so that two plans run under the same seed consume the
            same random numbers and their difference is paired. Required whenever results
            from different plans are subtracted; see ``RandomAlignment``.
        capture: Keep the addends of every lap time in ``result.breakdown``. Costs three
            more full-size arrays, so it is for the two plans being explained rather than
            the hundred being ranked.

    Returns:
        A ``SimulationResult`` carrying the full ``(n_sims, n_drivers, n_laps)`` sample.
    """
    from ..models.pace import feature_frame

    to_lap = state.total_laps if to_lap is None else to_lap
    caution = frozenset(caution_laps or ())
    if to_lap <= state.lap:
        raise ValueError(f"nothing to simulate: to_lap={to_lap} <= state.lap={state.lap}")

    sched = build_schedule(state, plans, to_lap)
    drivers = tuple(sched["driver"].unique())
    n_laps = to_lap - state.lap
    rng = np.random.default_rng(seed)

    X = feature_frame(sched.assign(driver=sched["driver"].astype(str)))
    reference = np.array([state.by_code[d].reference_pace_s for d in drivers])

    if components.noise is None:
        if capture:
            raise ValueError(
                "component capture needs a noise model: the predict_dist fallback draws "
                "the mean and the spread together and they cannot be separated afterwards"
            )
        # Phase 7 behaviour: the model's own i.i.d. residual bootstrap. Correct on the
        # mean, far too confident on the spread, and kept only for comparison.
        deltas = components.pace_model.predict_dist(X, n_sims, rng)
        lap_times = deltas.reshape(n_sims, len(drivers), n_laps)
    else:
        mu = components.pace_model.predict(X).reshape(len(drivers), n_laps)
        noise = components.noise.sample(
            sched,
            drivers,
            {d: state.by_code[d].clean_laps_observed for d in drivers},
            state.lap,
            n_sims,
            rng,
            n_stint_slots=align.stint_slots if align else None,
        )
        lap_times = mu[None, :, :] + noise
    lap_times = lap_times + reference[None, :, None]

    # The part of race progress this race is showing that the pooled model does not
    # expect, measured in-race from laps <= L (see state.estimate_race_progress_adjustment).
    # Anchored at state.lap, which is where the reference pace was anchored too.
    if state.race_progress_adjustment_s_per_lap:
        offsets = np.arange(1, n_laps + 1) * state.race_progress_adjustment_s_per_lap
        lap_times += offsets[None, None, :]

    multiplier = np.ones(n_laps)
    if caution:
        under = np.isin(np.arange(state.lap + 1, to_lap + 1), sorted(caution))
        multiplier[under] = components.caution_lap_multiplier
        lap_times[:, :, under] *= components.caution_lap_multiplier

    breakdown = None
    if capture:
        # Separating rather than recomputing: the noise field is already in hand, and the
        # caution multiplier scaled it along with everything else, so scaling it the same
        # way here reproduces its contribution exactly. `pace_s` is then the remainder,
        # which is what keeps the three parts summing to the total by construction.
        noise_s = noise * multiplier[None, None, :]
        pace_s = lap_times - noise_s
        before_pit = lap_times.copy()

    _apply_pit_losses(lap_times, sched, drivers, state.lap, caution, components, rng, align)

    if capture:
        breakdown = LapBreakdown(pace_s=pace_s, noise_s=noise_s, pit_s=lap_times - before_pit)

    return SimulationResult(
        race_id=state.race_id,
        circuit=state.circuit,
        from_lap=state.lap,
        to_lap=to_lap,
        drivers=drivers,
        lap_times_s=lap_times,
        plans={d: tuple(p for p in plans.get(d, ()) if p.lap > state.lap) for d in drivers},
        seed=seed,
        rival_policy=rival_policy,
        caution_laps=caution,
        breakdown=breakdown,
    )


def _apply_pit_losses(
    lap_times: np.ndarray,
    sched: pd.DataFrame,
    drivers: tuple[str, ...],
    from_lap: int,
    caution_laps: frozenset[int],
    components: SimComponents,
    rng: np.random.Generator,
    align: RandomAlignment | None = None,
) -> None:
    """Charge each stop's cost across its in-lap and out-lap, in place.

    One loss is drawn per (sim, stop) and *split* between the two laps rather than drawn
    twice, because the in-lap and out-lap excesses are two halves of one physical event.
    Drawing independently would halve the variance of a stop's cost.

    Under ``align`` the losses come from one block indexed by (driver, stop ordinal)
    instead of one draw per stop in schedule order, so that a driver's second stop meets
    the same pit crew whether the plan in front of it had one stop or three.
    """
    stops = sched.loc[sched["is_pit_in"], ["driver", "lap_number"]].sort_values(
        ["driver", "lap_number"]
    )
    n_sims = lap_times.shape[0]
    index = {d: i for i, d in enumerate(drivers)}
    share = components.pit_in_lap_share

    # Drawn before the empty check and at the declared size, so how much of the stream a
    # plan consumes does not depend on how many stops the plan happens to contain.
    block = None
    if align is not None:
        needed = int(stops.groupby("driver").size().max()) if len(stops) else 0
        if needed > align.stop_slots:
            raise ValueError(f"a driver has {needed} stops but stop_slots={align.stop_slots}")
        block = components.sample_pit_loss((n_sims, len(drivers), align.stop_slots), rng)

    if stops.empty:
        return

    for row in stops.assign(ordinal=stops.groupby("driver").cumcount()).itertuples():
        loss = (
            components.sample_pit_loss((n_sims,), rng)
            if block is None
            else block[:, index[row.driver], row.ordinal]
        )
        if row.lap_number in caution_laps:
            loss = loss * components.caution_pit_multiplier

        d = index[row.driver]
        in_lap = row.lap_number - from_lap - 1
        out_lap = in_lap + 1
        if out_lap < lap_times.shape[2]:
            lap_times[:, d, in_lap] += share * loss
            lap_times[:, d, out_lap] += (1 - share) * loss
        else:
            # The stop's out-lap falls outside the window; charge the whole cost to the
            # in-lap so the total is conserved rather than silently truncated.
            lap_times[:, d, in_lap] += loss
