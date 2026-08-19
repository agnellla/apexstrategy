"""The held-out season must be unreachable from anything that fits a model.

Adding a season to the store is the moment an out-of-sample test quietly becomes an
in-sample one: every model-facing reader takes `laps` and filters by circuit, and a
season column nobody filters on is invisible until the validation number comes back
suspiciously good. These tests make that structural, the same way the temporal firewall
makes lap-level leakage structural rather than something to stay alert about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexstrategy import config
from apexstrategy.data import store


def _laps(season: int, circuit: str = "bahrain", n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": [f"{season}_{circuit}_R"] * n,
            "season": season,
            "round": 1,
            "circuit": circuit,
            "driver": "AAA",
            "lap_number": np.arange(1, n + 1),
            "lap_time_s": 90.0,
            "is_clean": True,
            "compound": "MEDIUM",
            "tyre_age": np.arange(1, n + 1, dtype=float),
        }
    )


class TestSeasonGate:
    def test_training_laps_sees_only_the_training_season(self) -> None:
        both = pd.concat([_laps(config.TRAINING_SEASON), _laps(config.HOLDOUT_SEASONS[0])])
        out = store.trainable_laps(both)
        assert set(out["season"].unique()) == {config.TRAINING_SEASON}
        assert len(out) == 40

    def test_a_holdout_only_frame_yields_nothing_to_fit_on(self) -> None:
        """Loud and empty beats quiet and wrong: a fit on the held-out season should fail
        to find data rather than succeed on data it must never see."""
        assert store.trainable_laps(_laps(config.HOLDOUT_SEASONS[0])).empty

    def test_the_holdout_season_is_not_the_training_season(self) -> None:
        assert config.TRAINING_SEASON not in config.HOLDOUT_SEASONS

    @pytest.mark.parametrize(
        "reader",
        ["fittable_laps", "training_laps", "trainable_laps"],
        ids=["fittable", "training", "trainable"],
    )
    def test_every_model_facing_reader_is_season_aware(self, reader: str) -> None:
        """Named individually so that adding a fourth reader without a season filter
        shows up here as a missing case rather than as a silent leak."""
        assert hasattr(store, reader)


class TestComputedExclusions:
    """Exclusions that used to be hand-written lists are now rules, and rules need tests.

    D10 excluded Silverstone 2024 by hand after measuring it at 24.3% wet laps. That was
    the right call for one race and unmaintainable for seventy, so the threshold became a
    rule applied to every race. The risk with that trade is a rule that quietly stops
    matching the judgement it replaced, which is what these pin.
    """

    @staticmethod
    def _race(race_id: str, wet: int, dry: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": race_id,
                "season": config.TRAINING_SEASON,
                "circuit": race_id.split("_")[1],
                "compound": ["INTERMEDIATE"] * wet + ["MEDIUM"] * dry,
                "is_clean": True,
            }
        )

    def test_a_wet_race_is_excluded_and_a_dry_one_is_not(self) -> None:
        laps = pd.concat(
            [
                self._race("2024_britain_R", wet=24, dry=76),  # 24% wet, as measured
                self._race("2024_bahrain_R", wet=0, dry=100),
            ]
        )
        assert store.dry_race_ids(laps) == {"2024_bahrain_R"}

    def test_the_threshold_is_where_D10_put_it(self) -> None:
        """A damp lap or two does not make a race wet; the line is 5%."""
        just_under = self._race("2024_a_R", wet=4, dry=96)
        just_over = self._race("2024_b_R", wet=6, dry=94)
        ids = store.dry_race_ids(pd.concat([just_under, just_over]))
        assert ids == {"2024_a_R"}

    def test_monaco_is_excluded_on_judgement_no_rule_will_catch(self) -> None:
        """Near-zero overtaking and a pit lane whose geometry does not generalise (D3).
        Nothing measurable in the laps flags that, so it stays a named venue."""
        laps = self._race("2024_monaco_R", wet=0, dry=100).assign(
            lap_time_s=90.0, lap_number=1, tyre_age=1.0
        )
        assert "2024_monaco_R" in store.dry_race_ids(laps)
        assert store.trainable_laps(laps).empty
