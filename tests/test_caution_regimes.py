"""Caution regimes are not interchangeable, and the multipliers are fitted.

A single 1.21 stood in for every caution while the store held 27 caution laps, all from one
virtual safety car. That value was close to right for a VSC and badly wrong for a safety
car, which is the regime that actually dominates -- and caution laps were carrying 78% of
held-out race-time bias as a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexstrategy.data import store
from apexstrategy.sim.engine import (
    DEFAULT_CAUTION_LAP_MULTIPLIERS,
    DEFAULT_CAUTION_PIT_MULTIPLIER,
    SimComponents,
    caution_regime,
    fit_caution_lap_multipliers,
    fit_caution_pit_multiplier,
    simulate_field,
)


@pytest.fixture(scope="module")
def pace_model():
    from apexstrategy.models.pace import TARGET, feature_frame
    from apexstrategy.models.train import SELECTED, training_frame

    train = training_frame()
    return SELECTED().fit(feature_frame(train), train[TARGET])


@pytest.fixture(scope="module")
def components(pace_model) -> SimComponents:
    return SimComponents.fit(pace_model)


class TestRegimeOrdering:
    def test_a_safety_car_is_slower_than_a_vsc_is_slower_than_a_vsc_ending(
        self, components: SimComponents
    ) -> None:
        m = components.caution_lap_multipliers
        assert m["sc"] > m["vsc"] > m["vsc_ending"] > 1.0

    def test_every_regime_slows_the_car(self, components: SimComponents) -> None:
        assert all(v > 1.0 for v in components.caution_lap_multipliers.values())

    def test_the_blended_default_sits_between_the_extremes(self) -> None:
        """It stands in for a caution whose kind is unknown, so it must not be either
        extreme -- an assumed caution should cost what an average one costs."""
        m = DEFAULT_CAUTION_LAP_MULTIPLIERS
        assert (
            m["vsc_ending"]
            < SimComponents(pace_model=None, pit_loss_s=np.array([20.0])).caution_lap_multiplier
            < m["sc"]
        )

    def test_yellow_alone_is_not_a_caution(self) -> None:
        """Yellow-flag laps run at 1.006 of green pace. Treating them as cautions would
        slow a fifth of the field's laps for no reason."""
        assert caution_regime(pd.Series(["2", "12", "1"])).tolist() == ["green"] * 3

    def test_concatenated_codes_resolve_to_the_most_severe(self) -> None:
        """FastF1 concatenates every code active during the lap: '1254' is
        AllClear+Yellow+SC+Red. A lap that saw a safety car is a safety-car lap."""
        assert caution_regime(pd.Series(["1254", "16", "17", "1"])).tolist() == [
            "sc",
            "vsc",
            "vsc_ending",
            "green",
        ]


class TestFittedNotHardCoded:
    def test_the_lap_multipliers_come_from_data(self, components: SimComponents) -> None:
        """Equal to the fallbacks would prove nothing; they must be the training set's own
        numbers, which differ from the fallbacks in the fourth decimal or better."""
        fitted = components.caution_lap_multipliers
        assert fitted != DEFAULT_CAUTION_LAP_MULTIPLIERS
        assert fitted["sc"] == pytest.approx(1.45, abs=0.1)

    def test_the_pit_multiplier_comes_from_data(self, components: SimComponents) -> None:
        """It was ``ASSUMED_CAUTION_PIT_MULTIPLIER = 0.45`` because seven races contained
        no green-to-caution stop pairs at all. It is fitted now."""
        assert components.caution_pit_multiplier != DEFAULT_CAUTION_PIT_MULTIPLIER
        assert 0.3 < components.caution_pit_multiplier < 0.9

    def test_a_stop_under_caution_is_cheaper_than_a_green_one(
        self, components: SimComponents
    ) -> None:
        """The field is slowed while the stopping car loses only the pit lane, which is the
        whole reason teams stop under a safety car."""
        assert 0 < components.caution_pit_multiplier < 1.0

    def test_thin_regimes_fall_back_rather_than_fitting_on_a_handful(self) -> None:
        laps = pd.DataFrame(
            {
                "race_id": ["r"] * 60,
                "driver": ["AAA"] * 60,
                "lap_number": range(2, 62),
                "lap_time_s": [90.0] * 55 + [130.0] * 5,
                "track_status": ["1"] * 55 + ["4"] * 5,
                "is_pit_in": False,
                "is_pit_out": False,
            }
        )
        assert fit_caution_lap_multipliers(laps)["sc"] == DEFAULT_CAUTION_LAP_MULTIPLIERS["sc"]

    def test_the_pit_multiplier_falls_back_without_enough_stop_pairs(self, pace_model) -> None:
        empty = store.fittable_laps().head(0)
        got = fit_caution_pit_multiplier(empty, pace_model, DEFAULT_CAUTION_LAP_MULTIPLIERS)
        assert got == DEFAULT_CAUTION_PIT_MULTIPLIER


class TestNoWetContamination:
    def test_the_caution_fit_sees_only_usable_races(self) -> None:
        """Cautions cluster in wet races, so this fit is more exposed to the contamination
        than most. It goes through the same race-level rule as everything else."""
        laps = store.fittable_laps()
        admitted = laps[store.usable_races(laps)]
        dry = store.dry_race_ids(store.read_table("laps"))
        assert set(admitted["race_id"]) <= dry

    def test_fitting_on_wet_races_would_have_changed_the_answer(self, pace_model) -> None:
        """Evidence the exclusion matters here rather than being ceremonial."""
        laps = store.fittable_laps()
        clean_fit = fit_caution_lap_multipliers(laps[store.usable_races(laps)])
        everything = fit_caution_lap_multipliers(laps)
        assert clean_fit["sc"] != everything["sc"]


class TestSimulatorApplication:
    @staticmethod
    def _state():
        from apexstrategy.sim.state import DriverState, RaceState

        driver = DriverState(
            code="AAA",
            team="t",
            gap_to_leader_s=0.0,
            compound="MEDIUM",
            tyre_age=10,
            stint_number=1,
            compounds_used=frozenset({"MEDIUM"}),
            pit_stops=0,
            laps_completed=20,
            status="running",
            reference_pace_s=90.0,
            clean_laps_observed=15,
        )
        return RaceState(
            race_id="r",
            circuit="c",
            season=2024,
            round=1,
            lap=20,
            total_laps=40,
            track_status="green",
            drivers=(driver,),
            air_temp=25.0,
            track_temp=40.0,
            rainfall=False,
        )

    class _Flat:
        def predict(self, X):
            return np.zeros(len(X))

        def predict_dist(self, X, n, rng):
            return np.zeros((n, len(X)))

    def _components(self) -> SimComponents:
        return SimComponents(
            pace_model=self._Flat(),
            pit_loss_s=np.array([20.0]),
            noise=None,
            caution_lap_multipliers={"sc": 1.5, "vsc": 1.2, "vsc_ending": 1.05},
            caution_lap_multiplier=1.35,
        )

    def test_a_safety_car_lap_is_slowed_more_than_a_vsc_lap(self) -> None:
        state, c = self._state(), self._components()
        kw = dict(n_sims=2, seed=0, to_lap=30, caution_laps=frozenset({25}))
        sc = simulate_field(state, {}, c, **kw, caution_regimes={25: "sc"})
        vsc = simulate_field(state, {}, c, **kw, caution_regimes={25: "vsc"})
        green = simulate_field(state, {}, c, n_sims=2, seed=0, to_lap=30)
        i = 25 - 21
        assert sc.lap_times_s[0, 0, i] > vsc.lap_times_s[0, 0, i] > green.lap_times_s[0, 0, i]
        assert sc.lap_times_s[0, 0, i] / green.lap_times_s[0, 0, i] == pytest.approx(1.5)

    def test_an_unlabelled_caution_gets_the_blended_rate(self) -> None:
        """Which is the honest treatment of a caution that is assumed rather than observed."""
        state, c = self._state(), self._components()
        kw = dict(n_sims=2, seed=0, to_lap=30, caution_laps=frozenset({25}))
        unlabelled = simulate_field(state, {}, c, **kw)
        green = simulate_field(state, {}, c, n_sims=2, seed=0, to_lap=30)
        i = 25 - 21
        ratio = unlabelled.lap_times_s[0, 0, i] / green.lap_times_s[0, 0, i]
        assert ratio == pytest.approx(1.35)

    def test_green_laps_are_untouched(self) -> None:
        state, c = self._state(), self._components()
        with_caution = simulate_field(
            state,
            {},
            c,
            n_sims=2,
            seed=0,
            to_lap=30,
            caution_laps=frozenset({25}),
            caution_regimes={25: "sc"},
        )
        green = simulate_field(state, {}, c, n_sims=2, seed=0, to_lap=30)
        np.testing.assert_allclose(with_caution.lap_times_s[:, :, 6], green.lap_times_s[:, :, 6])
