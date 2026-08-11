"""Track-position gaps, derived from cumulative lap time.

FastF1 gives no gap column. Gaps are reconstructed by differencing cumulative session
time between drivers *at the same lap number*, which is the time between them as they
cross the line on that lap.

Two limits of that construction, both real and both documented rather than hidden:

- **Lapped traffic is invisible.** A car a lap down can be physically alongside the
  leader while its same-lap-number gap reads as a full lap. So ``gap_ahead_s`` measures
  the gap to the nearest car *on the same lap*, not necessarily the nearest car on track.
  For dirty-air detection this under-counts: some laps genuinely spent in traffic are not
  flagged. It never produces a false positive, which is the safer direction of error.
- **Gaps are sampled once per lap**, at the line. A car caught and passed mid-lap leaves
  no trace if the gap has reopened by the line.

Adequate for stint-level pace and strategy work; not precise enough to present as
official timing data.
"""

from __future__ import annotations

import pandas as pd

#: A car within this many seconds ahead is treated as putting the following car in
#: dirty air. Roughly the range over which aerodynamic wake measurably costs lap time;
#: DRS detection sits at 1.0s. Sensitivity to this value is reported by
#: ``dirty_air_sensitivity``, because it is a judgement call, not a measured constant.
DIRTY_AIR_THRESHOLD_S = 1.5

_GROUP = ["race_id", "lap_number"]


def add_gaps(laps: pd.DataFrame, threshold_s: float = DIRTY_AIR_THRESHOLD_S) -> pd.DataFrame:
    """Add ``gap_to_leader_s``, ``gap_ahead_s``, ``derived_position``, ``in_dirty_air``.

    ``in_dirty_air`` for lap N uses the gap measured at the end of lap **N-1**, i.e. the
    traffic state the car entered the lap in. This is deliberate and matters: using the
    end-of-lap gap would be selection-biased, because a car having a fast lap closes on
    the car ahead and would be flagged, while the same car having a slow lap would drop
    back and would not. Excluding preferentially-fast laps would bias the pace model
    downward. The entering gap is causally prior to the lap time it explains.
    """
    laps = laps.sort_values([*_GROUP, "cumulative_time_s"]).copy()
    grouped = laps.groupby(_GROUP, sort=False)["cumulative_time_s"]

    laps["gap_to_leader_s"] = laps["cumulative_time_s"] - grouped.transform("min")
    # Within a group sorted ascending by cumulative time, diff() is the gap to the car
    # immediately ahead. The leader gets NaN, which is correct: nobody is ahead.
    laps["gap_ahead_s"] = grouped.diff()
    laps["derived_position"] = grouped.cumcount() + 1

    laps["entering_gap_s"] = (
        laps.sort_values(["race_id", "driver", "lap_number"])
        .groupby(["race_id", "driver"], sort=False)["gap_ahead_s"]
        .shift(1)
        .reindex(laps.index)
    )
    laps["in_dirty_air"] = (laps["entering_gap_s"] < threshold_s).fillna(False)

    return laps.sort_values(["race_id", "driver", "lap_number"]).reset_index(drop=True)


def position_agreement(laps: pd.DataFrame) -> pd.DataFrame:
    """How often the reconstructed order matches FastF1's reported ``position``.

    The Phase 3 exit criterion. Disagreement is expected around pit stops and lapped
    cars -- FastF1's ``Position`` is the official classification at that lap, which
    accounts for cars on different laps, whereas the reconstruction orders purely by
    elapsed time at the same lap number. A high but imperfect agreement rate is the
    expected result; a low one means the reconstruction is broken.
    """
    both = laps.dropna(subset=["position", "derived_position"])
    both = both.assign(agrees=both["position"].astype(int) == both["derived_position"])
    out = both.groupby("circuit")["agrees"].agg(["mean", "size"])
    out.columns = ["agreement", "laps"]
    out["agreement"] = (100 * out["agreement"]).round(1)
    return out.sort_values("agreement")


def dirty_air_sensitivity(
    laps: pd.DataFrame, thresholds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
) -> pd.DataFrame:
    """Share of otherwise-clean laps lost to dirty air at each candidate threshold.

    Exists so the threshold is chosen with its cost visible rather than by assertion.
    """
    base = laps[laps["exclusion_reason"].isin(["clean", "dirty_air"])]
    rows = []
    for t in thresholds:
        flagged = (base["entering_gap_s"] < t).fillna(False)
        rows.append(
            {
                "threshold_s": t,
                "laps_excluded": int(flagged.sum()),
                "pct_of_otherwise_clean": round(100 * flagged.mean(), 1),
            }
        )
    return pd.DataFrame(rows)
