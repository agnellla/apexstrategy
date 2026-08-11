"""Integration tests against the built Parquet store.

Skipped if the store has not been built. Run ``uv run python -m apexstrategy.data.build``
first. These are the tests that catch a bad build, as opposed to bad logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from apexstrategy import config
from apexstrategy.data import store
from apexstrategy.data.schema import ALL_TABLES, DRY_COMPOUNDS

pytestmark = pytest.mark.skipif(not store.table_path("laps").exists(), reason="store not built")


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return store.read_all()


def test_every_table_satisfies_its_contract(tables: dict[str, pd.DataFrame]) -> None:
    for spec in ALL_TABLES:
        spec.validate_frame(tables[spec.name].reindex(columns=list(spec.column_names)))


def test_all_seven_circuits_present(tables: dict[str, pd.DataFrame]) -> None:
    assert set(tables["laps"]["circuit"]) == set(config.CIRCUITS_BY_KEY)


def test_clean_laps_are_dry_compounds_only(tables: dict[str, pd.DataFrame]) -> None:
    """MVP is dry-race only; Silverstone 2024 had a wet phase that must be excluded."""
    clean = tables["laps"].query("is_clean")
    assert set(clean["compound"]) <= DRY_COMPOUNDS


def test_stint_retirements_agree_with_results(tables: dict[str, pd.DataFrame]) -> None:
    """The bug this guards: inferring 'ended by pit' from lap numbers alone marks
    every lapped finisher and every retirement as a pit stop."""
    from_stints = tables["stints"].query("end_reason == 'retirement'").groupby("race_id").size()
    from_results = tables["results"].query("is_retired").groupby("race_id").size()
    pd.testing.assert_series_equal(
        from_stints.sort_index(), from_results.sort_index(), check_names=False
    )


def test_lapped_finishers_are_not_retirements(tables: dict[str, pd.DataFrame]) -> None:
    """Bahrain 2024 classified all 20 drivers; 11 were lapped, none retired."""
    bahrain = tables["results"].query("circuit == 'bahrain'")
    assert bahrain["is_retired"].sum() == 0
    assert (bahrain["laps_completed"] < bahrain["total_laps"]).sum() > 0


def test_only_pit_ended_stints_are_uncensored(tables: dict[str, pd.DataFrame]) -> None:
    stints = tables["stints"]
    assert (stints["is_censored"] == (stints["end_reason"] != "pit")).all()


def test_each_driver_has_exactly_one_final_stint(tables: dict[str, pd.DataFrame]) -> None:
    finals = (
        tables["stints"]
        .query("end_reason in ['race_end', 'retirement']")
        .groupby(["race_id", "driver"])
        .size()
    )
    assert (finals == 1).all()


def test_red_flag_pit_stops_are_flagged_under_caution(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Regression test. ``under_caution`` was originally derived from the prioritised
    ``exclusion_reason``, so Monaco's red-flag stops (labelled ``lap_1``) came back
    green-flagged -- which would have poisoned the Phase 4 pit-loss fit.
    """
    monaco = tables["pits"].query("circuit == 'monaco'")
    long_stops = monaco[monaco["pit_duration_s"] > 300]
    assert len(long_stops) > 0, "expected Monaco 2024 red-flag stops in the data"
    assert long_stops["under_caution"].all()


def test_green_flag_pit_durations_are_physically_plausible(
    tables: dict[str, pd.DataFrame],
) -> None:
    green = tables["pits"].query("not under_caution").dropna(subset=["pit_duration_s"])
    medians = green.groupby("circuit")["pit_duration_s"].median()
    assert medians.between(15, 40).all(), f"implausible pit-lane times:\n{medians}"


def test_stint_laps_reconcile_with_laps_table(tables: dict[str, pd.DataFrame]) -> None:
    from_laps = tables["laps"].groupby(["race_id", "driver"]).size()
    from_stints = tables["stints"].groupby(["race_id", "driver"])["n_laps"].sum()
    pd.testing.assert_series_equal(
        from_laps.sort_index(), from_stints.sort_index(), check_names=False
    )


def test_lap_times_are_raw_not_fuel_corrected(tables: dict[str, pd.DataFrame]) -> None:
    """Fuel correction is a fitted Phase 4 transform, deliberately not baked in here.

    A fuel-corrected series trends flat-to-up across a race; a raw one trends *down*
    as the car burns off fuel. Checking the sign of that trend detects a correction
    silently reappearing at ingest.
    """
    clean = tables["laps"].query("is_clean and circuit == 'barcelona'")
    early = clean.query("lap_number <= 15")["lap_time_s"].median()
    late = clean.query("lap_number >= 45")["lap_time_s"].median()
    assert late < early, "raw lap times should fall across a race as fuel burns off"


def test_training_laps_excludes_monaco_and_unclean() -> None:
    """D3 is enforced at the store boundary so no model can forget it."""
    train = store.training_laps()
    assert "monaco" not in set(train["circuit"])
    assert train["is_clean"].all()
    assert len(set(train["circuit"])) == 6
