"""Phase 6: ``RaceState`` -- the single source of truth handed to the simulator.

## The temporal firewall

``build_state`` truncates its input frame to laps <= L as its *first* action, and every
field is derived from that truncated frame. A future lap cannot influence the state
because no future lap is ever in scope. This is the architecture doc's "make leakage
impossible rather than detected" (Part 13), and it is asserted by a property test that
overwrites every lap after L with garbage and requires the state to be unchanged.

The one input that is not derived from race laps is ``total_laps``, which is the
*scheduled* race distance -- a pre-race fact from the calendar, not an outcome. It is a
required argument rather than something looked up inside, so that a caller cannot
accidentally pass a post-race lap count.

## Closing the Phase 5 reference-pace gap

The pace model predicts ``lap_time - reference_pace``, and in training the reference was
the median of a driver's clean laps over the *whole race*. Live, only laps <= L exist,
and their raw median is biased slow: early laps carry fuel and green track.

``estimate_reference_pace`` removes that bias by inverting the model instead of ignoring
it. For each observed clean lap it computes ``lap_time - model.predict(that lap)``, which
is an estimate of the same whole-race reference the model was trained against, and takes
the median. The model's own race-progress and degradation terms do the de-trending.

Measured over all five trainable circuits at decision laps 5-35 (n=397 driver-states),
against the whole-race reference as truth:

| Estimator | Bias | MAE |
|---|---|---|
| raw median of clean laps (Phase 5 behaviour) | +0.790 s | 0.810 s |
| **de-trended through the model** | **+0.045 s** | **0.463 s** |

A pleasant property falls out of this: a *constant* per-circuit offset in the pace model
cancels exactly. If the model is uniformly ``c`` seconds off at some circuit, the implied
reference absorbs ``-c`` and the projected lap time adds it back. Only the *shape* of the
model's error over tyre age and race distance survives -- which is what the replay
validation goes on to measure.

## The fallback matters more than the estimator

At lap 15 only 74% of drivers have three clean laps to their name, and at lap 5 just 9%
-- a car stuck in traffic generates almost none. An early version fell back to the
*field's* reference pace, which hands a backmarker the pace of the field and was the
single largest source of error in the first replay run.

The replacement uses the fact that traffic is a **one-sided** penalty: a lap in a wake is
slower than clean-air pace, never faster. So the fast tail of a driver's own green,
non-pit laps estimates their clean-air pace, and the fallback takes a low quantile of the
de-trended implied references rather than their median. On the drivers that actually need
it, the 10th percentile is the best of the quantiles tried (bias +0.23 s, MAE 0.75 s
against +0.66 s / 0.95 s for the median).

## What "retired" means at lap L

At lap L nobody knows who is about to retire, and the state must not either. A driver is
treated as running if their most recent line crossing is recent relative to the leader's:
a lapped car still crosses the line every lap, a stopped car does not. The consequence is
that a car which retires during lap L+1 still reads as running at lap L -- which is
correct, because that is genuinely all the pit wall knows at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..data.clean import (
    STATUS_RED,
    STATUS_SAFETY_CAR,
    STATUS_VSC_DEPLOYED,
    STATUS_VSC_ENDING,
    STATUS_YELLOW,
)

TrackStatus = Literal["green", "yellow", "vsc", "sc", "red"]

#: A driver whose last line crossing is older than this many of their own lap times,
#: measured against the leader's crossing on lap L, is treated as retired. One full lap
#: plus a margin: a lapped car is at most ~1 lap behind the leader in elapsed time, a
#: stopped car falls behind without bound. Deliberately generous -- calling a running car
#: retired removes it from the projection entirely, which is the worse error.
RETIREMENT_GRACE_LAPS = 1.6

#: Minimum clean laps before a driver's own clean-air pace is estimated from them
#: directly. Below this the traffic fallback runs instead.
MIN_LAPS_FOR_REFERENCE = 3

#: Shrinkage weight for the in-race race-progress correction: the estimate is multiplied
#: by ``n / (n + SHRINK)`` where ``n`` is the number of clean laps it was fitted on. At
#: lap 5 the field has produced a handful of laps and the slope is mostly noise about to
#: be extrapolated over 50 laps; by lap 25 there are hundreds and it is real signal.
#:
#: Chosen by sweeping the full replay suite (5 circuits x 4 decision laps). The unshrunk
#: estimate is *worse than no correction at all*, which is the whole case for shrinking:
#:
#: | shrink | clean-lap MAE | abs bias | race-time error |
#: |---|---|---|---|
#: | 0 (unshrunk) | 1.075 s | 0.739 s | 0.877% |
#: | 150 | 0.881 s | 0.485 s | 0.620% |
#: | **300** | **0.890 s** | **0.485 s** | **0.599%** |
#: | 600 | 0.909 s | 0.491 s | 0.592% |
#: | none (correction off) | 0.960 s | 0.538 s | 0.636% |
RACE_PROGRESS_SHRINK_LAPS = 300

#: Hard bound on that correction. The per-circuit race-progress coefficients measured in
#: Phase 4 span -0.027 to -0.096 s/lap, so a *residual* beyond this is a fitting artefact,
#: not a track. Clipping protects against a wild extrapolation from an early state.
MAX_RACE_PROGRESS_ADJUSTMENT = 0.06

#: Quantile of a driver's de-trended green laps used when they have too few clean ones.
#: Low because traffic only ever makes a lap slower: the fast tail of a driver's green
#: laps is the part run in something close to clear air. Selected by measurement over the
#: five trainable circuits -- 0.1 beat 0.2/0.3/0.5 on both bias and MAE.
TRAFFIC_FALLBACK_QUANTILE = 0.1

#: Track-status severity, most severe first. A lap's status is the concatenation of every
#: code active during it ("1254" = AllClear+Yellow+SC+Red), so this must be a substring
#: search in priority order, never an equality test.
_STATUS_PRIORITY: tuple[tuple[str, TrackStatus], ...] = (
    (STATUS_RED, "red"),
    (STATUS_SAFETY_CAR, "sc"),
    (STATUS_VSC_DEPLOYED, "vsc"),
    (STATUS_VSC_ENDING, "vsc"),
    (STATUS_YELLOW, "yellow"),
)


@dataclass(frozen=True)
class DriverState:
    """One car, as known at lap L.

    ``position`` is deliberately absent: it is derived from ``gap_to_leader_s`` by
    ``RaceState.order``. Storing both invites the two to contradict each other, and
    sorting by elapsed gap is the only definition of order the simulator uses.
    """

    code: str
    team: str
    gap_to_leader_s: float
    compound: str
    tyre_age: int
    stint_number: int
    compounds_used: frozenset[str]
    pit_stops: int
    laps_completed: int
    status: Literal["running", "retired"]
    #: Whole-race reference pace, estimated from laps <= L. The level the pace model's
    #: predicted deltas are added to.
    reference_pace_s: float

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def to_dict(self) -> dict:
        """JSON-serialisable form. The showcase front-end is fed precomputed JSON (D9),
        so nothing in this layer may depend on a Python consumer."""
        d = self.__dict__.copy()
        d["compounds_used"] = sorted(self.compounds_used)
        return d


@dataclass(frozen=True)
class RaceState:
    """The race at lap L. Immutable, self-contained, and free of future information."""

    race_id: str
    circuit: str
    season: int
    round: int
    lap: int
    total_laps: int
    track_status: TrackStatus
    drivers: tuple[DriverState, ...]
    air_temp: float
    track_temp: float
    rainfall: bool
    #: Elapsed session time at which the leader crossed the line on lap L. Race times are
    #: reported relative to this, so a projection never depends on the session clock.
    leader_time_s: float = field(default=0.0)
    #: Residual race progress this race is showing beyond what the pace model expects,
    #: in s/lap, measured from laps <= L. Applied by the simulator relative to ``lap``.
    #: Zero when no pace model was supplied to ``build_state``.
    race_progress_adjustment_s_per_lap: float = 0.0

    @property
    def laps_remaining(self) -> int:
        return self.total_laps - self.lap

    @property
    def by_code(self) -> dict[str, DriverState]:
        return {d.code: d for d in self.drivers}

    @property
    def running(self) -> tuple[DriverState, ...]:
        return tuple(d for d in self.drivers if d.is_running)

    def order(self) -> tuple[DriverState, ...]:
        """Running cars sorted by gap to the leader. The only definition of position."""
        return tuple(sorted(self.running, key=lambda d: d.gap_to_leader_s))

    def position_of(self, code: str) -> int:
        for i, d in enumerate(self.order(), start=1):
            if d.code == code:
                return i
        raise KeyError(f"{code} is not running at lap {self.lap}")

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "drivers"}
        d["drivers"] = [x.to_dict() for x in self.drivers]
        return d


def resolve_track_status(codes: pd.Series) -> TrackStatus:
    """Collapse concatenated FastF1 status codes to the most severe state present."""
    joined = "".join(codes.dropna().astype(str).unique())
    for code, label in _STATUS_PRIORITY:
        if code in joined:
            return label
    return "green"


def _green_racing_laps(observed: pd.DataFrame) -> pd.DataFrame:
    """Laps whose time reflects racing, traffic included: no pit lap, start, or caution.

    Wider than ``is_clean``, which additionally drops dirty air. That extra breadth is
    the whole point of the fallback -- for a driver stuck in a train these are the only
    laps that exist.
    """
    from ..data.clean import is_under_caution
    from ..data.schema import DRY_COMPOUNDS

    return observed[
        observed["lap_time_s"].notna()
        & (observed["lap_number"] > 1)
        & ~observed["is_pit_in"]
        & ~observed["is_pit_out"]
        & ~is_under_caution(observed["track_status"])
        & observed["compound"].isin(DRY_COMPOUNDS)
    ]


def implied_reference(laps: pd.DataFrame, pace_model) -> np.ndarray:
    """``lap_time - model(lap)`` for each lap: this lap's estimate of the race reference.

    If the model were perfect this would be flat across a race. Every deviation the
    replay validation finds shows up here first.
    """
    from ..models.pace import feature_frame

    return laps["lap_time_s"].to_numpy() - pace_model.predict(feature_frame(laps))


def estimate_race_progress_adjustment(observed: pd.DataFrame, pace_model) -> float:
    """How much faster per lap this race is trending than the pace model expects.

    The pace model carries a single pooled race-progress coefficient, but the true rate is
    a track property that Phase 4 measured varying from -0.027 s/lap (Austria) to
    -0.096 s/lap (Suzuka). Under leave-one-race-out the model cannot know which it is
    facing, and the replay validation showed that mismatch to be the *dominant* source of
    projection error -- the per-circuit residual slope tracks the Phase 4 coefficient gap
    at r = 0.993.

    It is, however, observable from the race in progress: regress each lap's implied
    reference on lap number, and a non-zero slope is exactly the part of race progress the
    model is missing. This uses only laps <= L, so it stays inside the firewall, and it is
    what a race engineer does when the track comes to them faster than expected.

    Estimated field-wide rather than per driver, because fuel burn and track evolution are
    shared by everyone, and pooling turns twenty noisy slopes into one usable one. Driver
    level is removed first so a fast car cannot tilt the slope.
    """
    clean = observed[observed["is_clean"] & observed["lap_time_s"].notna()]
    if len(clean) < MIN_LAPS_FOR_REFERENCE * 4 or clean["lap_number"].nunique() < 4:
        return 0.0

    implied = pd.Series(implied_reference(clean, pace_model), index=clean.index)
    centred = implied - implied.groupby(clean["driver"]).transform("median")
    slope = float(np.polyfit(clean["lap_number"], centred, 1)[0])

    shrunk = slope * len(clean) / (len(clean) + RACE_PROGRESS_SHRINK_LAPS)
    return float(np.clip(shrunk, -MAX_RACE_PROGRESS_ADJUSTMENT, MAX_RACE_PROGRESS_ADJUSTMENT))


def estimate_reference_pace(
    observed: pd.DataFrame,
    pace_model=None,
    field_fallback: float | None = None,
    progress_adjustment: float = 0.0,
    anchor_lap: int | None = None,
) -> float:
    """Estimate a driver's reference pace from laps <= L, anchored at ``anchor_lap``.

    With a model, each usable lap contributes ``lap_time - predicted_delta``, an estimate
    of the same whole-race median the model was trained against. Clean laps give the
    median of those; when there are too few, the fast tail of the driver's green laps
    stands in, because traffic is a one-sided penalty (see the module docstring).

    ``progress_adjustment`` de-trends those estimates back to the anchor lap, so that a
    reference built from early laps is not systematically off when applied to late ones.
    The simulator re-applies the same slope going forward.

    Without a model this degrades to the raw median of clean laps -- the Phase 5
    behaviour, kept so the state object is usable without one, and honestly worse.
    """
    anchor = observed["lap_number"].max() if anchor_lap is None else anchor_lap

    def anchored(laps: pd.DataFrame) -> np.ndarray:
        # Each lap's implied reference, moved to what it would have been at the anchor.
        return implied_reference(laps, pace_model) - progress_adjustment * (
            laps["lap_number"].to_numpy() - anchor
        )

    clean = observed[observed["is_clean"] & observed["lap_time_s"].notna()]

    if len(clean) >= MIN_LAPS_FOR_REFERENCE:
        if pace_model is None:
            return float(clean["lap_time_s"].median())
        return float(np.median(anchored(clean)))

    green = _green_racing_laps(observed)
    if len(green) >= MIN_LAPS_FOR_REFERENCE and pace_model is not None:
        return float(np.quantile(anchored(green), TRAFFIC_FALLBACK_QUANTILE))
    if len(green) >= MIN_LAPS_FOR_REFERENCE:
        return float(green["lap_time_s"].quantile(TRAFFIC_FALLBACK_QUANTILE))

    # Last resort: barely any racing laps exist yet (lap 2 or 3 of the race, or a driver
    # who has done nothing but pit). The field's pace is wrong for a slow car, but at
    # this point nothing better is observable.
    if field_fallback is not None and np.isfinite(field_fallback):
        return float(field_fallback)
    usable = observed[observed["lap_time_s"].notna() & (observed["lap_number"] > 1)]
    return float(usable["lap_time_s"].median()) if len(usable) else float("nan")


def _driver_state(
    observed: pd.DataFrame,
    lap: int,
    leader_time_s: float,
    leader_lap_time_s: float,
    pace_model,
    field_reference_s: float,
    progress_adjustment: float,
) -> DriverState:
    """Build one ``DriverState`` from that driver's laps <= L."""
    last = observed.iloc[-1]
    n_completed = int(last["lap_number"])
    # A driver can legitimately have no timed lap at all: Suzuka 2024 was red-flagged on
    # lap 1 and the cars eliminated in that incident never set one. Fall back to the
    # leader's pace so the retirement check below still has a scale to work with.
    timed = observed["lap_time_s"].dropna()
    own_lap_time = float(timed.median()) if len(timed) else leader_lap_time_s

    # Elapsed-time recency against the leader's lap-L crossing. A lapped car crosses the
    # line every lap; a stopped car does not.
    behind_s = leader_time_s - float(last["cumulative_time_s"])
    running = behind_s <= RETIREMENT_GRACE_LAPS * own_lap_time

    # gap_to_leader_s at the driver's own last lap, plus a full lap of the leader's pace
    # for every lap they are down. Lapped cars must sort behind the whole lead lap.
    laps_down = lap - n_completed
    gap = float(last["gap_to_leader_s"]) + laps_down * leader_lap_time_s

    return DriverState(
        code=str(last["driver"]),
        team=str(last["team"]),
        gap_to_leader_s=gap,
        compound=str(last["compound"]),
        tyre_age=int(last["tyre_age"]) if pd.notna(last["tyre_age"]) else 1,
        stint_number=int(last["stint_number"]),
        compounds_used=frozenset(observed["compound"].dropna().astype(str).unique()),
        pit_stops=int(observed["is_pit_in"].sum()),
        laps_completed=n_completed,
        status="running" if running else "retired",
        reference_pace_s=estimate_reference_pace(
            observed, pace_model, field_reference_s, progress_adjustment, anchor_lap=lap
        ),
    )


def build_state(
    laps: pd.DataFrame,
    lap: int,
    total_laps: int,
    pace_model=None,
    race_progress_adjustment: float | None = None,
) -> RaceState:
    """Assemble the race state at ``lap`` from a single race's laps frame.

    Args:
        laps: Normalised laps for exactly one race. May contain later laps -- they are
            discarded before anything is computed, which is the firewall.
        lap: The decision lap L. State reflects the race with lap L completed.
        total_laps: Scheduled race distance. A calendar fact, not a race outcome.
        pace_model: Optional fitted pace model, used only to de-trend the reference-pace
            estimate. Passing one closes the Phase 5 reference gap; omitting it falls
            back to the raw truncated median.

    Raises:
        ValueError: if the frame spans more than one race, or nobody reached ``lap``.
    """
    if laps["race_id"].nunique() != 1:
        raise ValueError(f"build_state expects one race, got {laps['race_id'].unique().tolist()}")

    # THE FIREWALL. Everything below reads only `observed`.
    observed = laps[laps["lap_number"] <= lap].sort_values(["driver", "lap_number"])
    if observed.empty:
        raise ValueError(f"no laps at or before lap {lap}")

    on_lap = observed[observed["lap_number"] == lap]
    if on_lap.empty:
        raise ValueError(f"nobody completed lap {lap}; the race may have been shorter")

    leader_time_s = float(on_lap["cumulative_time_s"].min())
    recent = observed[observed["lap_number"] > max(1, lap - 5)]
    leader_lap_time_s = float(recent["lap_time_s"].median())

    if race_progress_adjustment is not None:
        progress_adjustment = race_progress_adjustment
    elif pace_model is not None:
        progress_adjustment = estimate_race_progress_adjustment(observed, pace_model)
    else:
        progress_adjustment = 0.0

    # Field-level reference for drivers with too few clean laps of their own to trust.
    field_clean = observed[observed["is_clean"]]
    field_reference_s = (
        estimate_reference_pace(field_clean, pace_model, None, progress_adjustment, anchor_lap=lap)
        if len(field_clean) >= MIN_LAPS_FOR_REFERENCE
        else leader_lap_time_s
    )

    drivers = tuple(
        _driver_state(
            g,
            lap,
            leader_time_s,
            leader_lap_time_s,
            pace_model,
            field_reference_s,
            progress_adjustment,
        )
        for _, g in observed.groupby("driver", sort=True)
    )

    head = on_lap.iloc[0]
    return RaceState(
        race_id=str(head["race_id"]),
        circuit=str(head["circuit"]),
        season=int(head["season"]),
        round=int(head["round"]),
        lap=lap,
        total_laps=total_laps,
        track_status=resolve_track_status(on_lap["track_status"]),
        drivers=drivers,
        air_temp=float(on_lap["air_temp"].median()),
        track_temp=float(on_lap["track_temp"].median()),
        rainfall=bool(on_lap["rainfall"].fillna(False).any()),
        leader_time_s=leader_time_s,
        race_progress_adjustment_s_per_lap=progress_adjustment,
    )
