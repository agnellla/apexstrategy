"""Clean-lap classification.

Two principles carried over from the earlier ``pitwindow`` project, both deliberate:

1. **Flag, never drop.** Excluded laps stay in the table with ``is_clean=False``. The
   strategy audit still needs to know a pit-in lap happened even though its time is not
   representative of pace, and a dropped row leaves no trace for the data-quality view.
2. **Record *why*.** ``exclusion_reason`` carries the first applicable reason rather
   than a bare boolean, so clean-lap yield can be decomposed. Yield, and what drives it,
   is the Phase 2 exit criterion.

Every rule here depends only on information available at that lap. Excluding a lap
because of something that happened later would be outcome-informed cleaning
(architecture-proposal.md, Part 13, risk 7).
"""

from __future__ import annotations

import pandas as pd

from .schema import WET_COMPOUNDS

# FastF1 track-status codes. A lap's TrackStatus is the concatenation of every code
# active during it, e.g. "671" = VSCDeployed + VSCEnding + AllClear.
STATUS_YELLOW = "2"
STATUS_SAFETY_CAR = "4"
STATUS_RED = "5"
STATUS_VSC_DEPLOYED = "6"
STATUS_VSC_ENDING = "7"


#: Every code that means the field was slowed. Yellow is excluded: it is local, costs a
#: lift rather than a delta-time, and is handled as an ordinary pace exclusion.
CAUTION_CODES = (STATUS_SAFETY_CAR, STATUS_VSC_DEPLOYED, STATUS_VSC_ENDING, STATUS_RED)


def _has_status(track_status: pd.Series, code: str) -> pd.Series:
    return track_status.fillna("").astype(str).str.contains(code, regex=False)


def is_under_caution(track_status: pd.Series) -> pd.Series:
    """Whether a safety car, VSC, or red flag was active during each lap.

    Read from ``track_status`` directly and never from ``exclusion_reason``: that label
    is prioritised and lossy, and deriving caution state from it has already mislabelled
    16 Monaco red-flag pit stops as green-flag once in this project.
    """
    flagged = pd.Series(False, index=track_status.index)
    for code in CAUTION_CODES:
        flagged |= _has_status(track_status, code)
    return flagged


def _first_timed_lap_after_red_flag(laps: pd.DataFrame) -> pd.Series:
    """Flag each driver's first timed lap following a red-flag suspension.

    A restart runs through a formation procedure, so the first timed lap afterwards is
    far off representative pace but carries an ordinary ``AllClear`` track status by the
    time it completes -- every other rule lets it through. At Suzuka 2024 six such laps
    survived cleaning at 33-53% above the drivers' own race medians, and they were enough
    to drag that circuit's Phase 4 race-progress coefficient to roughly three times the
    pooled value.

    The rule keys off race-control state and the *presence* of a lap time, never its
    value. Excluding a lap for being slow would be outcome-informed cleaning
    (architecture-proposal.md, Part 13, risk 7); this is not that.
    """
    red = _has_status(laps["track_status"], STATUS_RED)
    flagged = pd.Series(False, index=laps.index)

    for _, idx in laps.groupby(["race_id", "driver"], sort=False).groups.items():
        group = laps.loc[idx]
        red_laps = group.loc[red.loc[idx], "lap_number"]
        if red_laps.empty:
            continue
        after = group[(group["lap_number"] > red_laps.max()) & group["lap_time_s"].notna()]
        if not after.empty:
            flagged.loc[after["lap_number"].idxmin()] = True

    return flagged


def classify_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_clean`` and ``exclusion_reason`` to a normalised laps frame.

    Expects the normalised column names (``lap_time_s``, ``is_pit_in``, ...), not raw
    FastF1 ones. Reasons are applied in the order listed in ``schema.EXCLUSION_REASONS``,
    so each excluded lap is attributed to exactly one cause and the counts partition
    cleanly.

    ``dirty_air`` is not applied here -- it needs gaps, which arrive in Phase 3.
    """
    laps = laps.copy()
    reason = pd.Series("clean", index=laps.index, dtype="object")

    def mark(mask: pd.Series, label: str) -> None:
        # Only laps still considered clean can take a new reason, which is what makes
        # the ordering meaningful and keeps the reasons mutually exclusive.
        reason.loc[(reason == "clean") & mask.fillna(False)] = label

    status = laps["track_status"]

    mark(laps["lap_time_s"].isna(), "no_lap_time")
    # Lap 1 is a standing start plus first-lap chaos: never representative of pace.
    mark(laps["lap_number"] == 1, "lap_1")
    mark(laps["is_pit_in"], "pit_in")
    mark(laps["is_pit_out"], "pit_out")
    mark(_has_status(status, STATUS_RED), "red_flag")
    if {"race_id", "driver"}.issubset(laps.columns):
        mark(_first_timed_lap_after_red_flag(laps), "restart")
    mark(_has_status(status, STATUS_SAFETY_CAR), "safety_car")
    mark(
        _has_status(status, STATUS_VSC_DEPLOYED) | _has_status(status, STATUS_VSC_ENDING),
        "vsc",
    )
    # pitwindow excluded SC/VSC/red but kept yellows. A local yellow still forces a
    # meaningful lift, so it is excluded here.
    mark(_has_status(status, STATUS_YELLOW), "yellow")
    mark(laps["is_deleted"], "deleted")
    mark(~laps["is_accurate"], "inaccurate")
    # MVP is dry-race only (Part 14, assumption 8).
    mark(laps["compound"].isin(WET_COMPOUNDS), "wet_compound")
    # Lowest priority: a lap spent in another car's wake is not clean-air pace. Applied
    # last so that a lap which is *also* a pit lap or under caution is attributed to the
    # stronger cause, keeping the dirty-air count interpretable as "otherwise clean".
    if "in_dirty_air" in laps.columns:
        mark(laps["in_dirty_air"], "dirty_air")

    laps["exclusion_reason"] = reason
    laps["is_clean"] = reason == "clean"
    return laps


def yield_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean-lap yield per race, decomposed by exclusion reason.

    This is the Phase 2 deliverable and feeds the dashboard's data-quality panel.
    """
    counts = laps.groupby(["race_id", "exclusion_reason"]).size().unstack(fill_value=0).astype(int)
    counts["total"] = counts.sum(axis=1)
    counts["clean_pct"] = (100 * counts.get("clean", 0) / counts["total"]).round(1)
    return counts.sort_values("clean_pct")
