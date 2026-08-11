"""Unit tests for clean-lap classification, on synthetic laps."""

from __future__ import annotations

import pandas as pd
import pytest

from apexstrategy.data.clean import classify_laps


def _laps(**overrides) -> pd.DataFrame:
    base = {
        "lap_number": 10,
        "lap_time_s": 90.0,
        "is_pit_in": False,
        "is_pit_out": False,
        "track_status": "1",
        "is_deleted": False,
        "is_accurate": True,
        "compound": "MEDIUM",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _reason(**overrides) -> str:
    return classify_laps(_laps(**overrides))["exclusion_reason"].iloc[0]


def test_normal_lap_is_clean() -> None:
    assert _reason() == "clean"
    assert classify_laps(_laps())["is_clean"].iloc[0]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"lap_time_s": None}, "no_lap_time"),
        ({"lap_number": 1}, "lap_1"),
        ({"is_pit_in": True}, "pit_in"),
        ({"is_pit_out": True}, "pit_out"),
        ({"track_status": "5"}, "red_flag"),
        ({"track_status": "4"}, "safety_car"),
        ({"track_status": "6"}, "vsc"),
        ({"track_status": "7"}, "vsc"),
        ({"track_status": "2"}, "yellow"),
        ({"is_deleted": True}, "deleted"),
        ({"is_accurate": False}, "inaccurate"),
        ({"compound": "INTERMEDIATE"}, "wet_compound"),
        ({"compound": "WET"}, "wet_compound"),
    ],
)
def test_each_exclusion_reason(overrides: dict, expected: str) -> None:
    assert _reason(**overrides) == expected


def test_concatenated_track_status_is_detected() -> None:
    """FastF1 concatenates every code active during a lap, e.g. '671' or '1254'."""
    assert _reason(track_status="671") == "vsc"
    assert _reason(track_status="1254") == "red_flag"


def test_reasons_are_mutually_exclusive_and_prioritised() -> None:
    """A lap matching several rules takes exactly the highest-priority one.

    This is why ``exclusion_reason`` must never be used to answer a different question
    (e.g. 'was this lap under caution?') -- the label is lossy by construction. Monaco
    2024 lap 1 was red-flagged but is labelled ``lap_1``.
    """
    assert _reason(lap_number=1, track_status="1254", is_pit_in=True) == "lap_1"


def test_yield_partitions_the_frame() -> None:
    laps = pd.concat(
        [_laps(), _laps(lap_number=1), _laps(is_pit_in=True), _laps(track_status="4")],
        ignore_index=True,
    )
    out = classify_laps(laps)
    assert len(out) == 4
    assert out["is_clean"].sum() == 1
    assert set(out["exclusion_reason"]) == {"clean", "lap_1", "pit_in", "safety_car"}
