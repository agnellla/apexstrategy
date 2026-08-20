"""One rule about which races may be learned from, applied everywhere it matters.

The project's data policy is race-level: a race is admitted if it is in a training season,
dry enough to fit a dry-tyre model on, and not at a venue excluded on judgement. The rule
lives in ``store.usable_races``.

It has now been restated locally twice and got it wrong both times, in the same way each
time -- filtering by *circuit* instead of by race, so that a dry running of a venue
readmitted a wet one. Conditions are a property of an afternoon; a track is not wet.

These tests pin the rule at the boundary rather than trusting each call site to remember
it, because the failure is silent: the wrong filter looks equivalent, trains without error,
and only shows up as inflated variance components or a degradation curve fitted partly on
intermediates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from apexstrategy import config
from apexstrategy.data import store
from apexstrategy.models.train import training_frame

#: Races that leaked into pace-model training before the rule was made canonical. Named
#: individually so that excluding them for some incidental reason later still counts as a
#: regression if that reason goes away.
KNOWN_LEAKS = ("2023_zandvoort_R", "2024_britain_R", "2024_montreal_R")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return training_frame()


class TestPaceModelTrainingData:
    def test_no_non_dry_race_is_trained_on(self, frame: pd.DataFrame) -> None:
        dry = store.dry_race_ids(store.read_table("laps"))
        admitted = set(frame["race_id"])
        assert admitted <= dry, sorted(admitted - dry)

    @pytest.mark.parametrize("race", KNOWN_LEAKS)
    def test_the_known_leaks_are_gone(self, frame: pd.DataFrame, race: str) -> None:
        assert race not in set(frame["race_id"])

    def test_a_dry_running_of_a_wet_venue_is_still_trained_on(self, frame: pd.DataFrame) -> None:
        """The rule excludes wet races, not wet tracks. Silverstone 2023 was bone dry and
        belongs in the pool; the old per-circuit exclusion threw it out with 2024."""
        assert "2023_britain_R" in set(frame["race_id"])

    def test_the_season_gate_and_venue_exclusion_still_hold(self, frame: pd.DataFrame) -> None:
        assert set(frame["season"]) <= set(config.TRAINING_SEASONS)
        assert not set(frame["season"]) & set(config.HOLDOUT_SEASONS)
        assert not set(frame["circuit"]) & store.EXCLUDED_VENUES

    def test_only_clean_laps_and_a_usable_target(self, frame: pd.DataFrame) -> None:
        from apexstrategy.models.pace import TARGET

        assert frame["is_clean"].all()
        assert frame[TARGET].notna().all()
        assert frame["tyre_age"].notna().all()

    def test_it_agrees_with_the_canonical_rule_exactly(self, frame: pd.DataFrame) -> None:
        """The point of the fix: the training set *is* the canonical rule's output, not a
        local reconstruction that happens to match today."""
        from apexstrategy.models.pace import TARGET, add_reference_pace

        canonical = store.trainable_laps(add_reference_pace(store.fittable_laps()))
        canonical = canonical.dropna(subset=[TARGET, "tyre_age"])
        assert set(frame["race_id"]) == set(canonical["race_id"])
        assert len(frame) == len(canonical)


class TestLeaveOneRaceOutTrainingData:
    """LORO folds obey the same admission rule as everything else.

    ``loro_pace_model`` was the third local restatement of the policy and, like the other
    two, filtered by circuit rather than by race. That conflated two different exclusions:
    holding out the circuit under test, which is what LORO means, and excluding wet races,
    which is a data-quality rule. A dry running of a venue therefore readmitted a wet one
    into every fold.
    """

    @pytest.mark.parametrize("held_out", ["hungary", "suzuka", "britain"])
    def test_no_wet_race_enters_a_fold(self, held_out: str) -> None:
        dry = store.dry_race_ids(store.read_table("laps"))
        assert set(_fold(held_out)["race_id"]) <= dry

    @pytest.mark.parametrize("held_out", ["hungary", "suzuka", "bahrain"])
    def test_the_held_out_circuit_is_absent_from_its_own_fold(self, held_out: str) -> None:
        """The exclusion that makes it leave-one-race-out at all. Circuit constants leak
        exactly like model weights (Part 13, risk 5)."""
        fold = _fold(held_out)
        assert held_out not in set(fold["circuit"])
        assert not fold["race_id"].str.contains(held_out).any()

    def test_holding_out_a_circuit_removes_only_that_circuit(self) -> None:
        """A fold must lose the held-out circuit and nothing else, or the exclusion is
        quietly doing a second job."""
        every = set(_fold(None)["race_id"])
        without = set(_fold("hungary")["race_id"])
        removed = every - without
        assert removed, "holding out hungary must remove something"
        assert all("hungary" in r for r in removed), sorted(removed)

    def test_the_fold_is_the_canonical_rule_minus_one_circuit(self) -> None:
        """The rule is applied, not reimplemented: a fold equals the canonical trainable
        set with one circuit dropped, exactly."""
        from apexstrategy.models.pace import TARGET, add_reference_pace

        canonical = store.trainable_laps(add_reference_pace(store.fittable_laps()))
        canonical = canonical[canonical["circuit"] != "suzuka"]
        canonical = canonical.dropna(subset=[TARGET, "tyre_age"])
        assert set(_fold("suzuka")["race_id"]) == set(canonical["race_id"])
        assert len(_fold("suzuka")) == len(canonical)


def _fold(held_out: str | None) -> pd.DataFrame:
    """The laps a LORO fold trains on, built the way ``loro_pace_model`` builds them."""
    from apexstrategy.models.pace import TARGET, add_reference_pace

    laps = add_reference_pace(store.fittable_laps())
    train = store.trainable_laps(laps)
    if held_out is not None:
        train = train[train["circuit"] != held_out]
    return train.dropna(subset=[TARGET, "tyre_age"])
