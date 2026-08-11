"""Turn a raw FastF1 session into the six normalised tables.

Lap times are stored **raw**. The earlier ``pitwindow`` project applied a fuel
correction at ingest using a hardcoded 0.035 s/kg against a 100 kg start load, which
meant every downstream model silently inherited an unfitted constant -- and, because
that correction is itself a function of lap number, any error in it transferred directly
into every within-stint degradation slope as a constant bias. Fuel correction is a
*fitted* per-circuit transform applied in Phase 4; see architecture-proposal.md,
Challenge 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import (
    STATUS_RED,
    STATUS_SAFETY_CAR,
    STATUS_VSC_DEPLOYED,
    STATUS_VSC_ENDING,
    classify_laps,
)
from .gaps import add_gaps
from .load import RawSession

#: Non-numeric ``ClassifiedPosition`` values mean the driver was not classified:
#: R=retired, D=disqualified, W=withdrawn, N=not classified, E=excluded.
_NOT_CLASSIFIED = frozenset({"R", "D", "W", "N", "E"})


def _with_ids(df: pd.DataFrame, raw: RawSession) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "circuit", raw.circuit)
    df.insert(0, "round", raw.round)
    df.insert(0, "season", raw.season)
    df.insert(0, "race_id", raw.race_id)
    return df


def _seconds(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s).dt.total_seconds()


def build_results(raw: RawSession, laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver, including how their race ended.

    This table is what makes correct stint censoring possible. Without it, "this stint
    did not end on the final lap" gets read as "this stint ended in a pit stop", which
    mislabels both retirements and lapped finishers.
    """
    res = raw.session.results
    laps_done = laps.groupby("driver")["lap_number"].max()

    classified = res["ClassifiedPosition"].astype(str).str.strip().str.upper()

    out = pd.DataFrame(
        {
            "driver": res["Abbreviation"].astype(str),
            "team": res["TeamName"].astype(str),
            "grid_position": pd.to_numeric(res["GridPosition"], errors="coerce"),
            "finish_position": pd.to_numeric(res["Position"], errors="coerce"),
            "classified_position": classified,
            "status": res["Status"].astype(str),
            "points": pd.to_numeric(res["Points"], errors="coerce"),
            "is_retired": classified.isin(_NOT_CLASSIFIED).to_numpy(),
        }
    ).reset_index(drop=True)

    out["laps_completed"] = out["driver"].map(laps_done).fillna(0).astype(int)
    out["total_laps"] = raw.total_laps
    return _with_ids(out, raw)


def build_weather(raw: RawSession) -> pd.DataFrame:
    w = raw.session.weather_data
    out = pd.DataFrame(
        {
            "time_s": _seconds(w["Time"]),
            "track_temp": pd.to_numeric(w["TrackTemp"], errors="coerce"),
            "air_temp": pd.to_numeric(w["AirTemp"], errors="coerce"),
            "humidity": pd.to_numeric(w["Humidity"], errors="coerce"),
            "wind_speed": pd.to_numeric(w["WindSpeed"], errors="coerce"),
            "rainfall": w["Rainfall"].astype(bool),
        }
    ).drop_duplicates(subset="time_s")
    return _with_ids(out.reset_index(drop=True), raw)


def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Nearest-timestamp merge of weather onto laps.

    The explicit float cast on both join keys is carried over from ``pitwindow``, where
    pandas versions disagreed on timedelta resolution (ns vs us) depending on how each
    Series was constructed and silently produced a misaligned merge.
    """
    cols = ["track_temp", "air_temp", "humidity", "wind_speed", "rainfall"]
    if weather.empty or laps["cumulative_time_s"].isna().all():
        for c in cols:
            laps[c] = pd.NA
        return laps

    left = laps.sort_values("cumulative_time_s").copy()
    right = weather[["time_s", *cols]].sort_values("time_s").copy()
    left["_k"] = left["cumulative_time_s"].astype("float64")
    right["_k"] = right["time_s"].astype("float64")

    merged = pd.merge_asof(left.dropna(subset=["_k"]), right, on="_k", direction="nearest").drop(
        columns=["_k", "time_s"]
    )

    # Laps with no timestamp cannot be matched; re-attach them with null weather.
    unmatched = left[left["_k"].isna()].drop(columns=["_k"])
    if not unmatched.empty:
        for c in cols:
            unmatched[c] = pd.NA
        merged = pd.concat([merged, unmatched], ignore_index=True)

    return merged.sort_values(["driver", "lap_number"]).reset_index(drop=True)


def build_laps(raw: RawSession, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per lap, raw times, cleaned-flagged."""
    src = raw.session.laps

    df = pd.DataFrame(
        {
            "driver": src["Driver"].astype(str),
            "team": src["Team"].astype(str),
            "lap_number": pd.to_numeric(src["LapNumber"], errors="coerce").astype("Int64"),
            "lap_time_s": _seconds(src["LapTime"]),
            "cumulative_time_s": _seconds(src["Time"]),
            "compound": src["Compound"].astype(str).str.upper().fillna("UNKNOWN"),
            "tyre_age": pd.to_numeric(src["TyreLife"], errors="coerce"),
            "is_fresh_tyre": src["FreshTyre"],
            "stint_number": pd.to_numeric(src["Stint"], errors="coerce").astype("Int64"),
            "position": pd.to_numeric(src["Position"], errors="coerce"),
            "is_pit_in": src["PitInTime"].notna(),
            "is_pit_out": src["PitOutTime"].notna(),
            "track_status": src["TrackStatus"].astype(str),
            "is_deleted": src.get("Deleted", pd.Series(False, index=src.index))
            .fillna(False)
            .astype(bool),
            "is_accurate": src.get("IsAccurate", pd.Series(True, index=src.index))
            .fillna(False)
            .astype(bool),
        }
    ).reset_index(drop=True)

    df = df.dropna(subset=["lap_number", "stint_number"])
    df["lap_number"] = df["lap_number"].astype(int)
    df["stint_number"] = df["stint_number"].astype(int)
    df["compound"] = df["compound"].replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})
    df["is_fresh_tyre"] = df["is_fresh_tyre"].astype("boolean")

    df = _with_ids(df, raw)
    # Gaps before cleaning: dirty-air exclusion depends on them.
    df = add_gaps(df)
    df = classify_laps(df)
    df = df.drop(columns=["is_deleted", "is_accurate"])
    return _merge_weather(df, weather)


def build_stints(raw: RawSession, laps: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per stint, with a correctly censored ``end_reason``.

    A stint ends in a ``pit`` unless it is the driver's last, in which case it ended
    either at the flag (``race_end``) or with a ``retirement``. Only ``pit`` is an
    uncensored observation of a strategy-driven stint length (D1).
    """
    retired = dict(zip(results["driver"], results["is_retired"], strict=True))

    agg = (
        laps.groupby(["driver", "stint_number"])
        .agg(
            compound=("compound", "first"),
            start_lap=("lap_number", "min"),
            end_lap=("lap_number", "max"),
            n_laps=("lap_number", "size"),
            start_tyre_age=("tyre_age", "min"),
        )
        .reset_index()
    )

    last_stint = agg.groupby("driver")["stint_number"].transform("max")
    is_final = agg["stint_number"] == last_stint
    driver_retired = agg["driver"].map(retired).fillna(False).astype(bool)

    agg["end_reason"] = np.where(
        ~is_final, "pit", np.where(driver_retired, "retirement", "race_end")
    )
    agg["is_censored"] = agg["end_reason"] != "pit"
    return _with_ids(agg, raw)


def build_pits(raw: RawSession, laps_src: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """One row per pit stop, timed pit-in to pit-out.

    ``pit_duration_s`` is total pit-lane time (entry to exit), not stationary time --
    it is the quantity the strategy layer trades against, though it still excludes the
    time lost decelerating and re-accelerating on track. The full circuit-level pit loss
    is fitted in Phase 4.

    ``under_caution`` is read from ``track_status`` rather than from
    ``exclusion_reason``: the latter is a *prioritised* label, so a red-flagged lap 1
    is recorded as ``"lap_1"`` and its caution state would be lost. A stop made under
    a red flag or safety car costs a fraction of a green-flag stop, so mislabelling it
    would corrupt the Phase 4 pit-loss fit.
    """
    src = laps_src[["Driver", "LapNumber", "PitInTime", "PitOutTime"]].copy()
    src.columns = ["driver", "lap_number", "pit_in", "pit_out"]
    src = src.dropna(subset=["lap_number"])
    src["lap_number"] = src["lap_number"].astype(int)
    src["driver"] = src["driver"].astype(str)

    indexed = laps.set_index(["driver", "lap_number"])
    caution = indexed["track_status"].fillna("").astype(str)
    compound = indexed["compound"]

    rows = []
    for driver, g in src.sort_values("lap_number").groupby("driver"):
        g = g.reset_index(drop=True)
        in_laps = g[g["pit_in"].notna()]
        for stop_number, (_, r) in enumerate(in_laps.iterrows(), start=1):
            lap = int(r["lap_number"])
            nxt = g[g["lap_number"] == lap + 1]
            duration = np.nan
            if not nxt.empty and pd.notna(nxt.iloc[0]["pit_out"]):
                duration = (
                    pd.to_timedelta(nxt.iloc[0]["pit_out"]) - pd.to_timedelta(r["pit_in"])
                ).total_seconds()
            rows.append(
                {
                    "driver": driver,
                    "stop_number": stop_number,
                    "lap": lap,
                    "pit_duration_s": duration,
                    "compound_fitted": compound.get((driver, lap + 1)),
                    "under_caution": any(
                        code in caution.get((driver, lap), "")
                        for code in (
                            STATUS_SAFETY_CAR,
                            STATUS_RED,
                            STATUS_VSC_DEPLOYED,
                            STATUS_VSC_ENDING,
                        )
                    ),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "driver",
            "stop_number",
            "lap",
            "pit_duration_s",
            "compound_fitted",
            "under_caution",
        ],
    )
    out["under_caution"] = out["under_caution"].astype(bool)
    return _with_ids(out, raw)


def build_track_status(raw: RawSession, laps: pd.DataFrame) -> pd.DataFrame:
    """Track-status intervals mapped from session time onto lap numbers.

    Lap attribution uses the leader's elapsed time (the minimum cumulative time at each
    lap), so ``start_lap`` is the lap the leader was on when the status changed. Lapped
    cars were on an earlier lap at that moment -- an approximation that is adequate for
    scenario-level SC modelling but not for per-car lap attribution.
    """
    ts = raw.session.track_status
    if ts is None or len(ts) == 0:
        return _with_ids(pd.DataFrame(columns=["status", "message", "start_lap", "end_lap"]), raw)

    leader = laps.groupby("lap_number")["cumulative_time_s"].min().dropna().sort_index()
    lap_no = leader.index.to_numpy()
    lap_t = leader.to_numpy()

    t = _seconds(ts["Time"]).to_numpy()
    start_lap = np.clip(np.searchsorted(lap_t, t, side="right"), 0, lap_no.max())
    end_lap = np.append(start_lap[1:], lap_no.max())

    out = pd.DataFrame(
        {
            "status": ts["Status"].astype(str).to_numpy(),
            "message": ts["Message"].astype(str).to_numpy(),
            "start_lap": start_lap.astype(int),
            "end_lap": np.maximum(end_lap, start_lap).astype(int),
        }
    ).drop_duplicates(subset=["start_lap", "status"], keep="first")
    return _with_ids(out.reset_index(drop=True), raw)


def normalize_session(raw: RawSession) -> dict[str, pd.DataFrame]:
    """Build all six tables for one race."""
    weather = build_weather(raw)
    laps = build_laps(raw, weather)
    results = build_results(raw, laps)
    return {
        "laps": laps,
        "stints": build_stints(raw, laps, results),
        "pits": build_pits(raw, raw.session.laps, laps),
        "track_status": build_track_status(raw, laps),
        "weather": weather,
        "results": results,
    }
