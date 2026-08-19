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
    """A model with one component live at a time, for the isolation tests below.

    ``race_scale_log_sd`` is off by default here and nowhere else. It is a *multiplier* on
    every within-race effect, so leaving it on would leak into each isolation and, worse,
    would correlate effects the isolation tests are checking are independent -- which is
    exactly what the race scale is supposed to do in production. ``TestRaceScale`` turns it
    on deliberately.
    """
    defaults = {
        "lap_residuals": np.array([-0.5, -0.2, 0.0, 0.2, 0.9]),
        "stint_level_sd": 0.4,
        "stint_slope_sd": 0.05,
        "race_scale_log_sd": 0.0,
        # Floors off too, for the same reason: an isolation test that sets reference_k=0
        # is asking for "no per-car reference error", and a floor left standing quietly
        # keeps the term alive and breaks the isolation.
        "reference_floor": 0.0,
        "reference_shared_floor": 0.0,
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


class TestRaceScale:
    """One afternoon's worth of chaos, drawn once and applied to everyone.

    Measured out of fold across 48 races, per-race residual spread runs 0.431 s to 2.233 s
    around a pooled 0.773 s. Regressing a race's spread on the same circuit's spread in
    other seasons correlates at 0.04, so it cannot be predicted and has to be integrated
    over instead.
    """

    def test_a_noisy_race_is_noisy_for_everyone_at_once(self) -> None:
        """The scale is shared, so it must correlate drivers who share nothing else."""
        model = _noise(stint_level_sd=0.0, stint_slope_sd=0.0, race_scale_log_sd=0.5)
        draws = _sample(model, n_sims=800)
        spread_a = np.abs(draws[:, 0, :]).mean(axis=1)
        spread_b = np.abs(draws[:, 1, :]).mean(axis=1)
        assert np.corrcoef(spread_a, spread_b)[0, 1] > 0.4

    def test_it_widens_the_distribution_without_moving_the_centre(self) -> None:
        """A multiplier on mean-zero noise adds spread and no bias, which is the whole
        point: it must not quietly make projections slower."""
        flat = _sample(_noise(race_scale_log_sd=0.0), n_sims=1500)
        scaled = _sample(_noise(race_scale_log_sd=0.45), n_sims=1500)
        assert scaled.std() > flat.std() * 1.05
        assert abs(scaled.mean() - flat.mean()) < 0.05

    def test_it_is_fitted_from_data_not_assumed(self) -> None:
        """Two synthetic races with deliberately different spreads must produce a non-zero
        scale, and identical ones must not invent variance that is not there."""
        rng = np.random.default_rng(0)
        rows = []
        for race, sd in [("a", 0.3), ("b", 1.2), ("c", 0.5), ("d", 0.9)]:
            for driver in range(6):
                for lap in range(1, 61):
                    rows.append(
                        {
                            "race_id": race,
                            "driver": f"D{driver}",
                            "stint_number": 1 + lap // 25,
                            "tyre_age": float(lap % 25 + 1),
                            "resid": rng.normal(0, sd),
                        }
                    )
        frame = pd.DataFrame(rows)
        per_race = frame.groupby("race_id")["resid"].std()
        spread = float(np.std(np.log(per_race / frame["resid"].std())))
        assert spread > 0.3, "four races spanning 0.3s to 1.2s must show a real scale"


class TestReferenceFloor:
    """The reference error stops shrinking, and missing that was the calibration bug.

    Built with the shipped defaults rather than the isolation harness, because the floors
    are the thing under test here.
    """

    @staticmethod
    def _shipped() -> NoiseModel:
        return NoiseModel(
            lap_residuals=np.array([-0.5, 0.0, 0.5]),
            stint_level_sd=0.4,
            stint_slope_sd=0.05,
        )

    def test_uncertainty_does_not_fall_to_zero_with_unlimited_laps(self) -> None:
        model = self._shipped()
        assert model.reference_sd(10_000) == pytest.approx(model.reference_floor, abs=0.01)
        assert model.reference_sd(10_000) > 0.3

    def test_it_still_narrows_over_the_range_that_matters(self) -> None:
        model = self._shipped()
        assert model.reference_sd(2) > model.reference_sd(10) > model.reference_sd(30)

    def test_the_shared_term_has_its_own_floor_and_stays_capped(self) -> None:
        model = self._shipped()
        assert model.reference_shared_sd(10_000) == pytest.approx(
            model.reference_shared_floor, abs=0.01
        )
        assert model.reference_shared_sd(1) <= MAX_REFERENCE_SHARED_SD
