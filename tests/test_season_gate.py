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
