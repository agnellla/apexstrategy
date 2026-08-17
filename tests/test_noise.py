"""Phase 8 unit tests: does the noise actually live at the level it claims to?

The interesting tests here are structural. Any sampler can produce a wide distribution;
what makes this one correct is that a stint effect is *constant within a stint*, a
reference effect is *constant across a race*, and a shared effect is *identical for every
driver*. Those are checkable properties, and they are what stops the variance collapsing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexstrategy.sim.noise import MAX_REFERENCE_SHARED_SD, NoiseModel

DRIVERS = ("AAA", "BBB", "CCC")


def _schedule(from_lap: int = 10, to_lap: int = 40, pit_lap: int = 25) -> pd.DataFrame:
    rows = []
    for d in DRIVERS:
        age, stint = 5, 1
        for lap in range(from_lap + 1, to_lap + 1):
            if lap == pit_lap + 1:
                age, stint = 1, 2
            else:
                age += 1
            rows.append(
                {"driver": d, "lap_number": lap, "tyre_age": float(age), "stint_number": stint}
            )
    return pd.DataFrame(rows)


def _noise(**kw) -> NoiseModel:
    defaults = {
        "lap_residuals": np.array([-0.5, -0.2, 0.0, 0.2, 0.9]),
        "stint_level_sd": 0.4,
        "stint_slope_sd": 0.05,
    }
    return NoiseModel(**{**defaults, **kw})


def _sample(model: NoiseModel, n_sims: int = 600, observed: int = 20, seed: int = 0):
    return model.sample(
        _schedule(),
        DRIVERS,
        dict.fromkeys(DRIVERS, observed),
        from_lap=10,
        n_sims=n_sims,
        rng=np.random.default_rng(seed),
    )


class TestShape:
    def test_returns_one_value_per_sim_driver_lap(self) -> None:
        assert _sample(_noise()).shape == (600, 3, 30)

    def test_is_seed_deterministic(self) -> None:
        model = _noise()
        np.testing.assert_array_equal(_sample(model, seed=4), _sample(model, seed=4))


class TestLevels:
    """Each component must be constant over exactly the span it is defined on."""

    def test_reference_effect_persists_across_the_whole_race(self) -> None:
        """Only the driver-level term survives: with everything else off, a driver's
        noise is one number repeated for every lap of the race."""
        model = _noise(
            lap_residuals=np.array([0.0]),
            stint_level_sd=0.0,
            stint_slope_sd=0.0,
            progress_sd=0.0,
            reference_shared_k=0.0,
        )
        draws = _sample(model)
        assert draws.std(axis=2).max() == pytest.approx(0.0, abs=1e-12)
        assert draws[:, :, 0].std() > 0.05

    def test_shared_reference_is_identical_for_every_driver(self) -> None:
        """A field-wide error moves everyone together, so it cannot change who wins.
        Getting this wrong is what made the first calibration run over-confident about
        comparisons and under-confident about the field as a whole."""
        model = _noise(
            lap_residuals=np.array([0.0]),
            stint_level_sd=0.0,
            stint_slope_sd=0.0,
            progress_sd=0.0,
            reference_k=0.0,
        )
        draws = _sample(model)
        assert draws.std(axis=1).max() == pytest.approx(0.0, abs=1e-12)
        assert draws[:, 0, 0].std() > 0.05

    def test_stint_level_is_constant_within_a_stint_and_resampled_after_a_stop(self) -> None:
        model = _noise(
            lap_residuals=np.array([0.0]),
            stint_slope_sd=0.0,
            progress_sd=0.0,
            reference_k=0.0,
            reference_shared_k=0.0,
            continuing_stint_level_observed=False,
        )
        draws = _sample(model)
        first, second = draws[:, :, :15], draws[:, :, 15:]  # stop is at lap 25 of 11..40
        assert first.std(axis=2).max() == pytest.approx(0.0, abs=1e-12)
        assert second.std(axis=2).max() == pytest.approx(0.0, abs=1e-12)
        # Independent draws either side of the stop: a new set of tyres is a new unknown.
        assert abs(np.corrcoef(first[:, 0, 0], second[:, 0, 0])[0, 1]) < 0.15

    def test_continuing_stint_level_is_treated_as_already_observed(self) -> None:
        """The reference estimate was built from the running stint's laps, so it has
        already absorbed how good that set is. Drawing it again double-counts."""
        common = {
            "lap_residuals": np.array([0.0]),
            "stint_slope_sd": 0.0,
            "progress_sd": 0.0,
            "reference_k": 0.0,
            "reference_shared_k": 0.0,
        }
        known = _sample(_noise(**common, continuing_stint_level_observed=True))
        redrawn = _sample(_noise(**common, continuing_stint_level_observed=False))
        assert known[:, :, 0].std() == pytest.approx(0.0, abs=1e-12)
        assert redrawn[:, :, 0].std() > 0.1
        # The stint after the stop is unknown either way.
        assert known[:, :, -1].std() > 0.1

    def test_progress_effect_grows_with_distance_from_the_decision_lap(self) -> None:
        model = _noise(
            lap_residuals=np.array([0.0]),
            stint_level_sd=0.0,
            stint_slope_sd=0.0,
            reference_k=0.0,
            reference_shared_k=0.0,
            progress_sd=0.01,
        )
        draws = _sample(model)
        sd = draws.std(axis=0).mean(axis=0)
        assert sd[0] == pytest.approx(0.01, abs=0.002)  # one lap past the decision lap
        assert sd[-1] == pytest.approx(0.30, abs=0.05)  # thirty laps past it

    def test_stint_slope_pivots_around_the_middle_of_the_stint(self) -> None:
        """Age is centred within the stint when fitted, so the slope term must add
        nothing on average over the stint and the most at its ends."""
        model = _noise(
            lap_residuals=np.array([0.0]),
            stint_level_sd=0.0,
            progress_sd=0.0,
            reference_k=0.0,
            reference_shared_k=0.0,
            stint_slope_sd=0.05,
        )
        draws = _sample(model)[:, 0, :15]
        assert draws.mean(axis=1).std() == pytest.approx(0.0, abs=1e-12)
        assert draws[:, 0].std() > draws[:, 7].std()


class TestScales:
    def test_reference_uncertainty_narrows_as_clean_laps_accumulate(self) -> None:
        model = _noise()
        assert model.reference_sd(2) > model.reference_sd(10) > model.reference_sd(40)
        assert model.reference_sd(1) == pytest.approx(model.reference_k)

    def test_shared_reference_is_capped(self) -> None:
        """k/sqrt(N) extrapolates past anything observed when the field has barely run."""
        model = _noise()
        assert model.reference_shared_sd(1) == MAX_REFERENCE_SHARED_SD
        assert model.reference_shared_sd(400) < MAX_REFERENCE_SHARED_SD

    def test_zero_observed_laps_does_not_divide_by_zero(self) -> None:
        assert np.isfinite(_noise().reference_sd(0))
        assert np.isfinite(_noise().reference_shared_sd(0))

    def test_lap_noise_keeps_the_right_skew(self) -> None:
        """Gaussian lap noise would erase the slow tail that strategy risk is about."""
        skewed = np.concatenate([np.zeros(90), np.full(10, 3.0)])
        model = _noise(
            lap_residuals=skewed,
            stint_level_sd=0.0,
            stint_slope_sd=0.0,
            progress_sd=0.0,
            reference_k=0.0,
            reference_shared_k=0.0,
        )
        draws = _sample(model).ravel()
        assert pd.Series(draws).skew() > 1.5


class TestFit:
    def test_decomposes_a_synthetic_world_with_known_components(self) -> None:
        """Synthetic-world test (Part 11, test 7): build a race from known variance
        components and require the estimator to recover them."""
        rng = np.random.default_rng(0)
        rows = []
        for race in range(6):
            for driver in ("A", "B", "C", "D"):
                for stint, start in enumerate([1, 21, 41], start=1):
                    level, slope = rng.normal(0, 0.5), rng.normal(0, 0.06)
                    for age in range(1, 19):
                        centred = age - 9.5
                        rows.append(
                            {
                                "race_id": f"R{race}",
                                "circuit": "t",
                                "driver": driver,
                                "compound": "MEDIUM",
                                "stint_number": stint,
                                "tyre_age": float(age),
                                "lap_number": start + age - 1,
                                "track_temp": 40.0,
                                "air_temp": 25.0,
                                "is_fresh_tyre": age == 1,
                                "is_clean": True,
                                "lap_time_s": 90.0 + level + slope * centred + rng.normal(0, 0.30),
                            }
                        )

        class Zero:
            def predict(self, X):
                return np.zeros(len(X))

        model = NoiseModel.fit(pd.DataFrame(rows), Zero())
        assert model.stint_level_sd == pytest.approx(0.5, abs=0.12)
        assert model.stint_slope_sd == pytest.approx(0.06, abs=0.02)
        assert float(np.std(model.lap_residuals)) == pytest.approx(0.30, abs=0.05)

    def test_raises_when_no_stint_is_long_enough(self) -> None:
        class Zero:
            def predict(self, X):
                return np.zeros(len(X))

        rows = [
            {
                "race_id": "R",
                "circuit": "t",
                "driver": "A",
                "compound": "MEDIUM",
                "stint_number": i,
                "tyre_age": 1.0,
                "lap_number": i,
                "track_temp": 40.0,
                "air_temp": 25.0,
                "is_fresh_tyre": True,
                "is_clean": True,
                "lap_time_s": 90.0,
            }
            for i in range(1, 6)
        ]
        with pytest.raises(ValueError, match="no stint long enough"):
            NoiseModel.fit(pd.DataFrame(rows), Zero())
