"""Phase 6: what the simulator is asked to evaluate.

A scenario is deliberately minimal: the ego driver's remaining pit plan, plus the
assumptions the plan is judged under. Pace modes, driver instructions and per-stint
push/conserve settings are **not** here -- they multiply the search space and cannot be
validated against anything the data contains.

## Rival fidelity is declared, not assumed

``rival_policy`` names which tier of rival behaviour a result was produced under
(architecture-proposal.md, Challenge 2). It travels with the result all the way to the
UI, because a number produced under frozen rivals means something different from one
produced under simulated rivals, and a tool that hides the difference is misleading.

| Tier | Rivals | Valid for |
|---|---|---|
| ``frozen`` | replay their observed lap times and pit laps | short-horizon undercut questions |
| ``sim_pace`` | driven by the pace model, pit laps fixed to observed | full-race projections |
| ``reactive`` | respond to the ego strategy | not implemented (Phase 2+) |

## The two-compound rule is a hard constraint, not a preference

A dry race requires two different dry compounds. It is frequently the entire reason a
stop happens when it does, so a system that forgets it recommends illegal strategies with
a straight face. ``Scenario.is_legal`` checks it against the compounds already used,
which is why ``DriverState`` carries ``compounds_used``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import pandas as pd

from ..data.schema import DRY_COMPOUNDS

RivalPolicy = Literal["frozen", "sim_pace", "reactive"]

#: Shortest stint worth simulating. Below this a stop is not a strategy, it is damage
#: control, and the pace model has no data there.
MIN_STINT_LAPS = 5


@dataclass(frozen=True, order=True)
class PitInstruction:
    """Pit at the end of ``lap``; ``compound`` runs from ``lap + 1``."""

    lap: int
    compound: str

    def __post_init__(self) -> None:
        if self.compound not in DRY_COMPOUNDS:
            raise ValueError(f"{self.compound} is not a dry compound; the MVP is dry-only")


@dataclass(frozen=True)
class Scenario:
    """An ego pit plan plus the assumptions it is evaluated under."""

    ego: str
    plan: tuple[PitInstruction, ...] = ()
    rival_policy: RivalPolicy = "frozen"
    #: ``None`` means "use the circuit's empirical rate". Exogenous by design: safety
    #: cars are not predicted here (Challenge 3), they are assumed and varied.
    sc_rate_per_lap: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if list(self.plan) != sorted(self.plan):
            raise ValueError("plan must be sorted by lap")
        laps = [p.lap for p in self.plan]
        if len(set(laps)) != len(laps):
            raise ValueError(f"two stops on the same lap: {laps}")

    @property
    def n_stops(self) -> int:
        return len(self.plan)

    def compounds_after(self, already_used: frozenset[str]) -> frozenset[str]:
        return already_used | {p.compound for p in self.plan}

    def is_legal(self, already_used: frozenset[str], total_laps: int, from_lap: int) -> bool:
        """Two different dry compounds by the flag, and no implausibly short stint."""
        if len(self.compounds_after(already_used) & DRY_COMPOUNDS) < 2:
            return False
        boundaries = [from_lap, *[p.lap for p in self.plan], total_laps]
        return all(b - a >= MIN_STINT_LAPS for a, b in pairwise(boundaries))

    def to_dict(self) -> dict:
        return {
            "ego": self.ego,
            "plan": [{"lap": p.lap, "compound": p.compound} for p in self.plan],
            "rival_policy": self.rival_policy,
            "sc_rate_per_lap": self.sc_rate_per_lap,
            "label": self.label or self.describe(),
        }

    def describe(self) -> str:
        if not self.plan:
            return "no further stops"
        return " -> ".join(f"L{p.lap} {p.compound}" for p in self.plan)


def observed_plans(laps: pd.DataFrame, after_lap: int = 0) -> dict[str, tuple[PitInstruction, ...]]:
    """Recover the pit plan each driver actually executed, from the laps frame.

    The ground truth the replay validation feeds back in: if the simulator cannot
    reproduce a race it has been handed the real strategies for, no counterfactual it
    produces afterwards is worth anything.

    Pit laps are read from ``is_pit_in``; the compound fitted is the compound on the
    following lap, which is what the car actually ran. Stops onto a wet compound are
    dropped rather than raising -- the MVP is dry-only, and Silverstone is in the store.
    """
    plans: dict[str, tuple[PitInstruction, ...]] = {}
    for driver, g in laps.sort_values("lap_number").groupby("driver", sort=True):
        by_lap = g.set_index("lap_number")["compound"]
        stops = [
            PitInstruction(int(lap), str(by_lap[lap + 1]))
            for lap in g.loc[g["is_pit_in"], "lap_number"].astype(int)
            if lap > after_lap and (lap + 1) in by_lap.index and by_lap[lap + 1] in DRY_COMPOUNDS
        ]
        plans[str(driver)] = tuple(sorted(stops))
    return plans
