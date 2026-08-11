"""Unit tests for gap reconstruction, on synthetic races."""

from __future__ import annotations

import pandas as pd

from apexstrategy.data.gaps import add_gaps


def _race(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    """(driver, lap_number, cumulative_time_s) -> a minimal race frame."""
    return pd.DataFrame(rows, columns=["driver", "lap_number", "cumulative_time_s"]).assign(
        race_id="R"
    )


def test_leader_has_zero_gap_and_null_gap_ahead() -> None:
    out = add_gaps(_race([("A", 1, 100.0), ("B", 1, 102.0), ("C", 1, 105.0)]))
    leader = out.query("driver == 'A'").iloc[0]
    assert leader["gap_to_leader_s"] == 0.0
    assert pd.isna(leader["gap_ahead_s"])
    assert leader["derived_position"] == 1


def test_gaps_are_relative_to_the_car_ahead_not_the_leader() -> None:
    out = add_gaps(_race([("A", 1, 100.0), ("B", 1, 102.0), ("C", 1, 105.0)])).set_index("driver")
    assert out.loc["C", "gap_to_leader_s"] == 5.0
    assert out.loc["C", "gap_ahead_s"] == 3.0  # to B, not to A


def test_exactly_one_leader_per_lap() -> None:
    out = add_gaps(_race([("A", 1, 100.0), ("B", 1, 102.0), ("A", 2, 200.0), ("B", 2, 201.0)]))
    assert out["gap_ahead_s"].isna().sum() == 2  # one per lap


def test_order_follows_time_not_input_order() -> None:
    out = add_gaps(_race([("C", 1, 105.0), ("A", 1, 100.0), ("B", 1, 102.0)]))
    order = out.sort_values("derived_position")["driver"].tolist()
    assert order == ["A", "B", "C"]


def test_positions_swap_when_times_do() -> None:
    out = add_gaps(
        _race([("A", 1, 100.0), ("B", 1, 102.0), ("A", 2, 205.0), ("B", 2, 203.0)])
    ).set_index(["driver", "lap_number"])
    assert out.loc[("A", 1), "derived_position"] == 1
    assert out.loc[("B", 2), "derived_position"] == 1


class TestDirtyAir:
    """The entering-gap choice is the load-bearing decision here."""

    def test_uses_entering_gap_not_end_of_lap_gap(self) -> None:
        """B is 3.0s back entering lap 2, then closes to 0.5s by the line.

        Lap 2 must NOT be flagged: B ran that lap in clear air. Flagging it would
        select against fast laps, since closing up is what a fast lap looks like,
        and would bias the pace model downward.
        """
        out = add_gaps(
            _race(
                [
                    ("A", 1, 100.0),
                    ("B", 1, 103.0),  # 3.0s back at end of lap 1
                    ("A", 2, 200.0),
                    ("B", 2, 200.5),  # closed to 0.5s at end of lap 2
                ]
            )
        ).set_index(["driver", "lap_number"])
        assert not out.loc[("B", 2), "in_dirty_air"]

    def test_flags_the_lap_after_closing_up(self) -> None:
        """Having ended lap 2 only 0.5s back, B starts lap 3 in A's wake."""
        out = add_gaps(
            _race(
                [
                    ("A", 1, 100.0),
                    ("B", 1, 103.0),
                    ("A", 2, 200.0),
                    ("B", 2, 200.5),
                    ("A", 3, 300.0),
                    ("B", 3, 301.0),
                ]
            )
        ).set_index(["driver", "lap_number"])
        assert out.loc[("B", 3), "in_dirty_air"]

    def test_leader_is_never_in_dirty_air(self) -> None:
        out = add_gaps(
            _race([("A", 1, 100.0), ("B", 1, 100.5), ("A", 2, 200.0), ("B", 2, 200.5)])
        ).set_index(["driver", "lap_number"])
        assert not out.loc[("A", 2), "in_dirty_air"]

    def test_first_lap_is_never_flagged(self) -> None:
        """No previous lap exists, so there is no entering gap to judge."""
        out = add_gaps(_race([("A", 1, 100.0), ("B", 1, 100.2)]))
        assert not out["in_dirty_air"].any()

    def test_threshold_is_respected(self) -> None:
        """B enters lap 2 exactly 2.0s back: excluded at a 1.0s cutoff, not at 3.0s."""
        rows = [("A", 1, 100.0), ("B", 1, 102.0), ("A", 2, 200.0), ("B", 2, 202.0)]
        key = ("B", 2)
        tight = add_gaps(_race(rows), threshold_s=1.0).set_index(["driver", "lap_number"])
        loose = add_gaps(_race(rows), threshold_s=3.0).set_index(["driver", "lap_number"])
        assert not tight.loc[key, "in_dirty_air"]
        assert loose.loc[key, "in_dirty_air"]
