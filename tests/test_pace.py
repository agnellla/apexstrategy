"""Unit tests for the pace model and its feature contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexstrategy.models.base import ResidualSampler
from apexstrategy.models.pace import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    GBMPaceModel,
    LinearPaceBaseline,
    add_reference_pace,
    feature_frame,
)


def _laps(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A synthetic race with a known 0.05 s/lap degradation rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for driver in ("A", "B"):
        base = 90.0 if driver == "A" else 91.0
        for stint, (compound, start) in enumerate([("MEDIUM", 1), ("HARD", 31)], start=1):
            for age in range(1, n // 4 + 1):
                lap = start + age - 1
                rows.append(
                    {
                        "race_id": "R",
                        "circuit": "testtrack",
                        "driver": driver,
                        "compound": compound,
                        "stint_number": stint,
                        "tyre_age": float(age),
                        "lap_number": lap,
                        "track_temp": 40.0,
                        "air_temp": 25.0,
                        "is_fresh_tyre": age == 1,
                        "entering_gap_s": 5.0,
                        "is_clean": True,
                        "lap_time_s": base + 0.05 * age + rng.normal(0, 0.05),
                    }
                )
    return pd.DataFrame(rows)


class TestReferencePace:
    def test_target_is_centred_per_driver_race(self) -> None:
        out = add_reference_pace(_laps())
        medians = out.groupby(["race_id", "driver"])[TARGET].median()
        assert np.allclose(medians, 0.0, atol=1e-9)

    def test_reference_differs_between_drivers(self) -> None:
        """B is a second slower; the reference must absorb that, not the target."""
        out = add_reference_pace(_laps())
        refs = out.groupby("driver")["reference_pace_s"].first()
        assert abs((refs["B"] - refs["A"]) - 1.0) < 0.15

    def test_reference_uses_only_clean_laps(self) -> None:
        laps = _laps()
        laps.loc[laps.index[:5], ["is_clean", "lap_time_s"]] = [False, 300.0]
        out = add_reference_pace(laps)
        assert out["reference_pace_s"].max() < 120.0


class TestFeatureContract:
    def test_endogenous_columns_are_excluded(self) -> None:
        """position and gap_to_leader_s are known at prediction time but endogenous:
        a slow car falls back. Feeding them to the simulator would make lap time depend
        on an ordering that is itself an output of lap times."""
        cols = set(feature_frame(_laps()).columns)
        assert not cols & {"position", "derived_position", "gap_to_leader_s", "gap_ahead_s"}

    def test_no_outcome_columns_leak_in(self) -> None:
        cols = set(feature_frame(_laps()).columns)
        assert not cols & {"lap_time_s", TARGET, "reference_pace_s", "is_clean"}

    def test_expected_features_present(self) -> None:
        cols = set(feature_frame(_laps()).columns)
        assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOL_FEATURES) <= cols

    def test_traffic_feature_only_when_requested(self) -> None:
        assert "entering_gap_s" not in feature_frame(_laps())
        assert "entering_gap_s" in feature_frame(_laps(), include_traffic=True)

    def test_clear_air_encoded_as_large_gap_not_null(self) -> None:
        """No car ahead is information, not missing data."""
        laps = _laps()
        laps.loc[laps.index[0], "entering_gap_s"] = np.nan
        X = feature_frame(laps, include_traffic=True)
        assert X["entering_gap_s"].notna().all()
        assert X["entering_gap_s"].iloc[0] > 50


@pytest.mark.parametrize("factory", [LinearPaceBaseline, GBMPaceModel])
class TestModels:
    def test_recovers_known_degradation(self, factory) -> None:
        laps = add_reference_pace(_laps())
        model = factory().fit(feature_frame(laps), laps[TARGET])
        pred = model.predict(feature_frame(laps))
        slope = np.polyfit(laps["tyre_age"], pred, 1)[0]
        assert 0.02 < slope < 0.08, f"synthetic truth is 0.05 s/lap, model implies {slope}"

    def test_predict_dist_shape(self, factory) -> None:
        laps = add_reference_pace(_laps())
        model = factory().fit(feature_frame(laps), laps[TARGET])
        draws = model.predict_dist(feature_frame(laps), n=25, rng=np.random.default_rng(0))
        assert draws.shape == (25, len(laps))

    def test_predict_dist_is_seed_deterministic(self, factory) -> None:
        laps = add_reference_pace(_laps())
        model = factory().fit(feature_frame(laps), laps[TARGET])
        X = feature_frame(laps)
        a = model.predict_dist(X, 10, np.random.default_rng(7))
        b = model.predict_dist(X, 10, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_predict_dist_is_centred_on_predict(self, factory) -> None:
        laps = add_reference_pace(_laps())
        model = factory().fit(feature_frame(laps), laps[TARGET])
        X = feature_frame(laps)
        draws = model.predict_dist(X, 800, np.random.default_rng(1))
        assert np.abs(draws.mean(axis=0) - model.predict(X)).max() < 0.15


def test_predict_dist_before_fit_raises() -> None:
    class Bare(ResidualSampler):
        def predict(self, X):  # pragma: no cover - never reached
            return np.zeros(len(X))

    with pytest.raises(RuntimeError, match="must be fitted"):
        Bare().predict_dist(pd.DataFrame({"a": [1]}), 5, np.random.default_rng(0))
