"""Phase 9: which strategies are worth simulating at all.

The candidate space is a Cartesian product of pit lap, compound and stop count, and it
explodes: from lap 20 of a 70-lap race, two stops anywhere on an every-third-lap grid with
three compounds each is over four thousand plans. Simulating them is affordable -- 0.03 s
apiece -- but the output is not, because a ranking with four thousand rows is not a
recommendation. **Pruning is a product requirement before it is a performance one.**

Three filters, in the order they bite:

1. **The two-compound rule (regulatory, hard).** A dry race requires two different dry
   compounds. This is not a tidying-up constraint -- it is frequently the entire reason a
   stop happens when it does, and a system that forgets it recommends illegal strategies
   with a straight face. Delegated to ``Scenario.is_legal``.
2. **Stint length (physical).** No stint shorter than ``MIN_STINT_LAPS`` or longer than
   anything that compound has been seen to do.
3. **A decision window on the *next* stop.** Later stops sit on a coarse grid; the near
   one gets the fine grid, then ``refine_candidates`` sharpens the leaders.

## The stint the car is already on counts

``Scenario.is_legal`` measures the first stint from the decision lap, which makes an
immediate stop look like a zero-lap stint and rejects it. That is the wrong constraint
here: a driver twenty laps into a set can pit next lap, and an undercut is exactly that
move -- it is the Hungary 2024 demo (D4). What is physically bounded is the *set's* total
life, so the filters below measure stints from where the tyre was fitted, adding
``tyre_age`` to the segment still to run. ``is_legal`` is still called, for the compound
rule it owns.

## Maximum stint length is fitted, not asserted

Taken from the store as the longest stint each compound has been seen to complete, as a
share of race distance -- shares rather than laps because the circuit set spans 53 to 71
laps. Measured over the five trainable circuits: HARD 0.586, MEDIUM 0.485, SOFT 0.379.

Two things about this number are worth being clear on:

- It is a **lower bound on what is possible**, not an upper bound. Many of those stints
  ended at the flag (``is_censored``), so the tyre was still going. Using it as a cap is
  therefore conservative, and the conservatism is one-directional: it can hide a viable
  strategy, never invent one.
- It is fitted per compound and **pooled across circuits**, because with one race per
  circuit a per-circuit maximum is a single observation.

Because a wrongly excluded candidate is invisible while a wrongly included one merely
ranks badly and gets seen, the filter is deliberately set at the observed maximum rather
than a quantile of it.

**The cap has to bind on "stay out" too**, which is easy to get wrong because that plan
has no stop to filter. It is the plan most likely to break the pace model: the degradation
term is linear in tyre age and will happily extrapolate a 54-lap stint out of 41 laps of
evidence, then report the phantom degradation as a reason to pit. Measured on Hamilton at
Hungary lap 40, exempting it was worth a fictitious 28.6 s.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise

import pandas as pd

from ..data import store
from ..data.schema import DRY_COMPOUNDS
from ..sim.scenario import MIN_STINT_LAPS, PitInstruction, RivalPolicy, Scenario
from ..sim.state import RaceState

#: How far ahead the *next* stop is searched. Beyond this the question stops being "when
#: do we pit" and becomes "what shape is the rest of the race", which the stop-count axis
#: already answers -- and the state will have been rebuilt long before the window closes.
DEFAULT_WINDOW_LAPS = 20

#: Grid spacing for the next stop, and for the ones after it. The near stop is the
#: decision, so it gets the fine grid; a later stop is a placeholder that will be re-decided
#: at its own decision lap, and pretending to pick it to the lap would be false precision.
DEFAULT_STEP = 4
DEFAULT_LATER_STEP = 8

#: Remaining stops considered. Three further stops has no support in the store -- the
#: longest race run anywhere in it is four stints -- and would be a recommendation the
#: model has never seen anyone make.
DEFAULT_MAX_STOPS = 2

#: Laps either side of a leading candidate's stops that ``refine_candidates`` re-searches.
DEFAULT_REFINE_SPREAD = 2


def fit_max_stint_share(exclude_circuit: str | None = None) -> dict[str, float]:
    """Longest observed stint per dry compound, as a share of race distance.

    LORO-scoped like every other circuit constant: evaluating a strategy at Hungary with a
    cap fitted on Hungary's own stints is the same leak as fitting the pace model on it
    (Part 13, risk 5).
    """
    from .. import config

    stints = store.read_table("stints")
    stints = stints[stints["season"] == config.TRAINING_SEASON]
    results = store.read_table("results")
    results = results[results["season"] == config.TRAINING_SEASON]
    allowed = {c.key for c in config.TRAINING_CIRCUITS} - {exclude_circuit}

    total_laps = results.groupby("circuit")["total_laps"].max()
    stints = stints[stints["circuit"].isin(allowed) & stints["compound"].isin(DRY_COMPOUNDS)]
    share = stints["n_laps"] / stints["circuit"].map(total_laps)
    by_compound = share.groupby(stints["compound"]).max()

    # A compound with no stints left after the exclusion falls back to the widest cap
    # rather than to zero, which would silently forbid it everywhere.
    widest = float(by_compound.max())
    return {c: float(by_compound.get(c, widest)) for c in sorted(DRY_COMPOUNDS)}


def stint_lengths(state: RaceState, ego: str, plan: Sequence[PitInstruction]) -> list[int]:
    """Total life of every set the plan uses, counting laps already run on the current one.

    The first entry is the set on the car now, so it starts at ``tyre_age`` rather than
    zero. That is what makes an immediate stop representable -- see the module docstring.
    """
    boundaries = [state.lap, *(p.lap for p in plan), state.total_laps]
    lengths = [b - a for a, b in pairwise(boundaries)]
    lengths[0] += state.by_code[ego].tyre_age
    return lengths


def stint_compounds(state: RaceState, ego: str, plan: Sequence[PitInstruction]) -> list[str]:
    """The compound on each of those sets, starting with the one fitted now."""
    return [state.by_code[ego].compound, *(p.compound for p in plan)]


def uses_two_dry_compounds(scenario: Scenario, already_used: frozenset[str]) -> bool:
    """The regulatory filter, on its own.

    ``Scenario.is_legal`` bundles this with a stint-length bound measured from the
    decision lap, which is the wrong bound here (module docstring). This is the half that
    is genuinely a rule rather than a physical estimate.
    """
    return len(scenario.compounds_after(already_used) & DRY_COMPOUNDS) >= 2


def is_physically_plausible(
    state: RaceState,
    ego: str,
    plan: Sequence[PitInstruction],
    max_stint_share: dict[str, float],
) -> bool:
    """Whether every set the plan uses runs a length that compound has been seen to run."""
    lengths = stint_lengths(state, ego, plan)
    caps = [max_stint_share[c] * state.total_laps for c in stint_compounds(state, ego, plan)]

    # The set already on the car can be past anything observed whatever we would have
    # preferred, so its cap is raised to the shortest length still *reachable* -- one more
    # lap, since the earliest a stop can happen is the end of the next one. Reachable, not
    # chosen: raising it to the plan's own length instead would exempt the no-stop plan
    # from the cap entirely, which is the trap the module docstring records.
    caps[0] = max(caps[0], state.by_code[ego].tyre_age + 1)

    return all(MIN_STINT_LAPS <= n <= cap for n, cap in zip(lengths, caps, strict=True))


def _plans(
    state: RaceState,
    n_stops: int,
    first_laps: Sequence[int],
    later_laps: Sequence[int],
    compounds: Sequence[str],
) -> Iterator[tuple[PitInstruction, ...]]:
    """Every ordered stop sequence of length ``n_stops`` on the two grids."""

    def extend(plan: tuple[PitInstruction, ...]) -> Iterator[tuple[PitInstruction, ...]]:
        if len(plan) == n_stops:
            yield plan
            return
        after = plan[-1].lap if plan else state.lap
        for lap in first_laps if not plan else later_laps:
            if lap <= after:
                continue
            for compound in compounds:
                yield from extend((*plan, PitInstruction(lap, compound)))

    return extend(())


def generate_candidates(
    state: RaceState,
    ego: str,
    *,
    max_stops: int = DEFAULT_MAX_STOPS,
    window_laps: int = DEFAULT_WINDOW_LAPS,
    step: int = DEFAULT_STEP,
    later_step: int = DEFAULT_LATER_STEP,
    compounds: Iterable[str] | None = None,
    max_stint_share: dict[str, float] | None = None,
    rival_policy: RivalPolicy = "sim_pace",
) -> list[Scenario]:
    """Legal, physically plausible pit plans for ``ego`` from the state's decision lap.

    Args:
        state: The race at lap L. Only ``ego``'s plan varies; the field's is the caller's.
        ego: Driver code the recommendation is for. Must be running.
        max_stops: Further stops considered, zero included when it is legal.
        window_laps: How far ahead the *next* stop is searched.
        step: Grid spacing for the next stop; ``later_step`` for the ones after it.
        max_stint_share: Per-compound cap from ``fit_max_stint_share``. Fitted here if
            omitted, which is in-sample -- pass a LORO-scoped one when evaluating a
            circuit that is in the training pool.

    Returns:
        Scenarios sorted by stop count then by stop lap, deduplicated. Empty means every
        plan in the grid was illegal or implausible, which is itself informative.
    """
    driver = state.by_code[ego]
    if not driver.is_running:
        raise ValueError(f"{ego} is not running at lap {state.lap}")

    compounds = tuple(sorted(compounds if compounds is not None else DRY_COMPOUNDS))
    share = max_stint_share or fit_max_stint_share()

    # A stop needs a stint after it, so the last one worth considering leaves room for the
    # shortest plausible run to the flag.
    last = state.total_laps - MIN_STINT_LAPS
    first_laps = list(range(state.lap + 1, min(state.lap + window_laps, last) + 1, step))
    later_laps = list(range(state.lap + 1, last + 1, later_step))

    out: dict[tuple[tuple[int, str], ...], Scenario] = {}
    for n_stops in range(max_stops + 1):
        for plan in _plans(state, n_stops, first_laps, later_laps, compounds):
            scenario = Scenario(ego=ego, plan=plan, rival_policy=rival_policy)
            if not uses_two_dry_compounds(scenario, driver.compounds_used):
                continue
            if not is_physically_plausible(state, ego, plan, share):
                continue
            out[tuple((p.lap, p.compound) for p in plan)] = scenario

    if not out:
        # Last resort, and only reached when the grid is genuinely empty -- a handful of
        # laps from the flag, where no stop leaves a stint worth running. Staying out is
        # then not a strategy, it is the only thing physically left, so it is offered even
        # though the stint is past anything observed. The engine has to return an answer
        # here rather than an exception; that the answer is an extrapolation is a property
        # of the situation, not a choice this function is making.
        to_the_flag = Scenario(ego=ego, plan=(), rival_policy=rival_policy)
        if uses_two_dry_compounds(to_the_flag, driver.compounds_used):
            return [to_the_flag]

    return sorted(out.values(), key=lambda s: (s.n_stops, [(p.lap, p.compound) for p in s.plan]))


def refine_candidates(
    state: RaceState,
    ego: str,
    around: Iterable[Scenario],
    *,
    spread: int = DEFAULT_REFINE_SPREAD,
    max_stint_share: dict[str, float] | None = None,
    exclude: Iterable[Scenario] = (),
) -> list[Scenario]:
    """Re-search each leading plan's **next** stop to the lap, within ``spread`` either side.

    The coarse grid finds the right shape -- one stop or two, and roughly when -- cheaply,
    because a two-lap error in a stop costs well under a second. This pass buys that back
    on the handful of plans actually in contention, which is the doc's coarse-then-refine
    (Part 9).

    Only the next stop moves. Refining a stop thirty laps away to the lap would be false
    precision by the module's own argument: the state will be rebuilt several times before
    that stop arrives, and the plan will be re-decided with laps of evidence this one does
    not have. It also keeps the pass linear in the number of stops rather than exponential
    -- refining both stops of six two-stop plans at +-2 generates 150 combinations and
    swamped the coarse set it was meant to sharpen.

    Plans in ``exclude`` (normally the coarse set) are dropped, so the caller can evaluate
    the union without paying for anything twice.
    """
    share = max_stint_share or fit_max_stint_share()
    seen = {tuple((p.lap, p.compound) for p in s.plan) for s in exclude}
    out: dict[tuple[tuple[int, str], ...], Scenario] = {}

    for scenario in around:
        if not scenario.plan:
            continue
        first, rest = scenario.plan[0], scenario.plan[1:]
        for offset in range(-spread, spread + 1):
            plan = (PitInstruction(first.lap + offset, first.compound), *rest)
            key = tuple((p.lap, p.compound) for p in plan)
            if key in seen or key in out:
                continue
            laps = [p.lap for p in plan]
            if laps != sorted(set(laps)):
                continue  # the offset pushed the next stop onto or past the one after it
            if not (state.lap < plan[0].lap and plan[-1].lap <= state.total_laps):
                continue
            if not is_physically_plausible(state, ego, plan, share):
                continue
            out[key] = Scenario(ego=ego, plan=plan, rival_policy=scenario.rival_policy)

    return sorted(out.values(), key=lambda s: (s.n_stops, [(p.lap, p.compound) for p in s.plan]))


def describe_grid(candidates: Sequence[Scenario]) -> pd.DataFrame:
    """How the surviving candidates are distributed over stop count. A pruning sanity check."""
    frame = pd.DataFrame({"n_stops": [s.n_stops for s in candidates]})
    return frame.value_counts().rename("n_candidates").reset_index().sort_values("n_stops")
