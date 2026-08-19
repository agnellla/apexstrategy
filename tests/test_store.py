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
    """Bahrain 2024 classified all 20 drivers; 11 were lapped, none retired.

    Season-filtered because the store holds more than one now. This test was implicitly
    single-season and started failing the moment a held-out season arrived, which is the
    cheapest possible reminder that ``circuit`` stopped being a unique race key.
    """
    bahrain = tables["results"].query("circuit == 'bahrain' and season == @config.SEASON")
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


def test_training_laps_excludes_untrainable_circuits_and_unclean() -> None:
    """Enforced at the store boundary so no model can forget it.

    Monaco is out as a pathological outlier (D3); Silverstone because 2024 was a
    mixed-conditions race (D10). Five circuits remain.
    """
    train = store.training_laps()
    assert {"monaco", "britain"}.isdisjoint(set(train["circuit"]))
    assert train["is_clean"].all()
    assert len(set(train["circuit"])) == 5


def test_exactly_one_leader_per_lap(tables: dict[str, pd.DataFrame]) -> None:
    """Null gap_ahead_s must occur exactly once per (race, lap) -- the leader."""
    laps = tables["laps"]
    leaders = laps["gap_ahead_s"].isna().sum()
    groups = laps.groupby(["race_id", "lap_number"]).ngroups
    assert leaders == groups


def test_derived_position_matches_reported(tables: dict[str, pd.DataFrame]) -> None:
    """Consistency check, not independent validation: FastF1's `position` comes from
    the same timing feed. It confirms the ordering logic, not the underlying times."""
    from apexstrategy.data.gaps import position_agreement

    assert (position_agreement(tables["laps"])["agreement"] > 99).all()


def test_gaps_are_non_negative(tables: dict[str, pd.DataFrame]) -> None:
    laps = tables["laps"]
    assert (laps["gap_to_leader_s"].dropna() >= 0).all()
    assert (laps["gap_ahead_s"].dropna() >= 0).all()


def test_a_driver_already_leading_is_never_flagged_dirty_air(
    tables: dict[str, pd.DataFrame],
) -> None:
    """The invariant holds on the PREVIOUS lap's position, not the current one.

    Because ``in_dirty_air`` uses the gap entering the lap, a car that started the lap
    in second place and took the lead during it is correctly flagged -- it really did
    run part of that lap in a wake. In the built store, 9 laps are flagged for a car
    classified as leader, and all 9 were P2 or P3 the lap before. A car that was
    already leading has no entering gap at all, so it can never be flagged.
    """
    laps = tables["laps"].sort_values(["race_id", "driver", "lap_number"])
    prev_position = laps.groupby(["race_id", "driver"])["derived_position"].shift(1)
    already_leading = laps[prev_position == 1]
    assert not already_leading["in_dirty_air"].any()


def test_clean_laps_are_in_clear_air(tables: dict[str, pd.DataFrame]) -> None:
    """The point of the dirty-air filter: `is_clean` must mean clean-AIR pace."""
    clean = tables["laps"].query("is_clean")
    assert not clean["in_dirty_air"].any()


def test_config_dry_race_exclusions_match_the_data(tables: dict[str, pd.DataFrame]) -> None:
    """Encode-then-verify, applied to scope. Config asserts which circuits are dry
    enough to fit on; the data must agree. Silverstone 2024 is 24.3% wet laps."""
    from apexstrategy.models.circuit_constants import dry_circuits

    measured_dry = dry_circuits(tables["laps"])
    for circuit in config.CIRCUITS:
        if circuit.use_for_training:
            assert circuit.key in measured_dry, (
                f"{circuit.key} is marked trainable but the data says it was not a dry race"
            )
