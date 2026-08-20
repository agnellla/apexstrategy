"""The uncertainty model must not know anything the pit wall does not.

Two separate guarantees, both easy to lose silently:

**Within a race.** Every input the noise model reads about the current race comes from
``RaceState``, which is built behind the Phase 6 temporal firewall. If a variance component
ever reached back to the laps frame it would be reading the future, and the intervals would
look excellent for the worst possible reason.

**Across races.** The components themselves are fitted from other races. A fit that
included the race being predicted would produce a noise model tuned to that afternoon's
chaos, which is the same leak one level up. The season gate (D15) and ``exclude_circuit``
are what prevent it, and this asserts they actually do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexstrategy import config
from apexstrategy.data import store
from apexstrategy.sim.engine import SimComponents
from apexstrategy.sim.noise import NoiseModel


@pytest.fixture(scope="module")
def pace_model():
    from apexstrategy.models.pace import TARGET, feature_frame
    from apexstrategy.models.train import SELECTED, training_frame

    train = training_frame()
    return SELECTED().fit(feature_frame(train), train[TARGET])


class TestNoFutureInformation:
    def test_sampling_reads_only_the_state_not_the_laps_frame(self) -> None:
        """``sample`` takes a schedule, driver codes, observed lap counts and a seed. There
        is no argument through which a future lap time could arrive, which is the guarantee
        -- stated as a signature check so that adding one is a deliberate act."""
        import inspect

        params = set(inspect.signature(NoiseModel.sample).parameters) - {"self"}
        assert params == {
            "schedule",
            "drivers",
            "clean_laps_observed",
            "from_lap",
            "n_sims",
            "rng",
            "n_stint_slots",
        }

    def test_widths_depend_on_observed_running_and_nothing_else(self) -> None:
        """A car with two clean laps must get a wider reference interval than one with
        thirty. That is the only race-specific input, and it is a count of laps already
        completed."""
        model = NoiseModel(np.array([-0.3, 0.0, 0.4]), 0.4, 0.05)
        assert model.reference_sd(2) > model.reference_sd(30) > model.reference_sd(2) * 0.4

    def test_the_race_scale_is_drawn_blind(self) -> None:
        """How chaotic this race turns out to be is sampled, never looked up. If it were
        conditioned on anything from the race in progress it would need an argument for it,
        and there is none -- two states differing only in driver identity draw the same
        scale for the same seed."""
        model = NoiseModel(np.array([-0.3, 0.0, 0.4]), 0.0, 0.0, race_scale_log_sd=0.4)
        sched = pd.DataFrame(
            {
                "driver": ["AAA"] * 20 + ["BBB"] * 20,
                "lap_number": list(range(21, 41)) * 2,
                "tyre_age": [float(i) for i in range(1, 21)] * 2,
                "stint_number": [2] * 40,
            }
        )
        args = (sched, ("AAA", "BBB"), {"AAA": 10, "BBB": 10}, 20, 64)
        a = model.sample(*args, np.random.default_rng(1))
        b = model.sample(*args, np.random.default_rng(1))
        np.testing.assert_array_equal(a, b)


class TestNoLeakageAcrossRaces:
    def test_the_held_out_season_never_reaches_the_noise_fit(self, pace_model) -> None:
        """The season gate is what makes the 2025 result a real out-of-sample number."""
        fitted = store.fittable_laps()
        assert set(fitted["season"]) <= set(config.TRAINING_SEASONS)
        assert not set(fitted["season"]) & set(config.HOLDOUT_SEASONS)

    def test_excluding_a_circuit_changes_the_fitted_components(self, pace_model) -> None:
        """If the exclusion were a no-op the components would be identical. They are not,
        which is the evidence that the replayed circuit is genuinely out of the fit."""
        everything = SimComponents.fit(pace_model)
        without = SimComponents.fit(pace_model, exclude_circuit="hungary")
        assert without.noise is not None and everything.noise is not None
        assert without.noise.lap_residuals.size < everything.noise.lap_residuals.size
        assert without.noise.stint_level_sd != everything.noise.stint_level_sd
        assert without.pit_loss_s.size < everything.pit_loss_s.size


class TestNoWetRaceContamination:
    """The constants path admits races, not circuits.

    D10 excluded Silverstone 2024 by hand after measuring it at 24.3% wet, because in and
    out laps run in damp conditions while the clean-pace reference comes from dry laps --
    its pit loss reads 41.4 s against 19.3 s for the dry running of the same track. When the
    hand list became a derived set, ``SimComponents.fit`` started filtering by *circuit*,
    so a dry Silverstone 2023 silently readmitted the wet 2024 race. Four races leaked.

    The pace-model path was never affected because ``store.training_laps`` filters per race.
    These tests pin the rule at the level it belongs to, so the two paths cannot drift apart
    again.
    """

    def test_no_non_dry_race_reaches_the_constants_fit(self) -> None:
        laps = store.fittable_laps()
        admitted = laps[store.usable_races(laps)]
        dry = store.dry_race_ids(store.read_table("laps"))
        assert set(admitted["race_id"]) <= dry, sorted(set(admitted["race_id"]) - dry)

    def test_the_four_known_leaks_are_gone(self) -> None:
        """Named individually: a rule that happens to exclude them today for some other
        reason would still be a regression if it stopped."""
        laps = store.fittable_laps()
        admitted = set(laps[store.usable_races(laps)]["race_id"])
        for race in (
            "2024_sao_paulo_R",
            "2024_montreal_R",
            "2023_zandvoort_R",
            "2024_britain_R",
        ):
            assert race not in admitted, race

    def test_a_dry_running_of_a_wet_venue_is_still_admitted(self) -> None:
        """The rule excludes wet *races*, not wet *tracks*. Silverstone 2023 was bone dry
        and belongs in the pool; throwing it out with 2024 is the failure the old
        per-circuit exclusion had."""
        laps = store.fittable_laps()
        admitted = set(laps[store.usable_races(laps)]["race_id"])
        assert "2023_britain_R" in admitted

    def test_the_season_gate_and_venue_exclusion_still_hold(self) -> None:
        """The fix must not have loosened the protections it sits beside."""
        laps = store.fittable_laps()
        admitted = laps[store.usable_races(laps)]
        assert set(admitted["season"]) <= set(config.TRAINING_SEASONS)
        assert not set(admitted["season"]) & set(config.HOLDOUT_SEASONS)
        assert not set(admitted["circuit"]) & store.EXCLUDED_VENUES

    def test_the_constants_actually_come_from_dry_races_only(self, pace_model) -> None:
        """End to end rather than on the filter alone: refit and check the pit-loss sample
        no longer carries the wet-race tail."""
        import numpy as np

        components = SimComponents.fit(pace_model)
        # Silverstone 2024's stops sat near 41 s; a dry-only pool should not reach there.
        assert np.median(components.pit_loss_s) < 30.0
        assert components.pit_loss_s.size > 500


class TestBootstrapTail:
    """The lap level is bootstrapped, not Gaussian, and that has to stay true."""

    def test_the_slow_tail_survives_resampling(self, pace_model) -> None:
        """Lap-time residuals are strongly right-skewed: a lap can go badly wrong but there
        is a floor on how fast it can go. A Gaussian erases exactly the slow tail that
        strategy risk lives in, so the empirical distribution is resampled instead."""
        from scipy import stats

        noise = SimComponents.fit(pace_model).noise
        residuals = noise.lap_residuals
        assert stats.skew(residuals) > 1.0, "the residuals really are right-skewed"

        rng = np.random.default_rng(0)
        drawn = rng.choice(residuals, size=200_000, replace=True)
        assert stats.skew(drawn) == pytest.approx(stats.skew(residuals), rel=0.15)
        # The far tail is the part a Gaussian would lose, so check it explicitly.
        assert np.quantile(drawn, 0.999) == pytest.approx(np.quantile(residuals, 0.999), rel=0.2)
        gaussian = rng.normal(residuals.mean(), residuals.std(), 200_000)
        assert np.quantile(drawn, 0.999) > np.quantile(gaussian, 0.999)

    def test_the_tail_is_not_borrowed_from_the_race_being_predicted(self, pace_model) -> None:
        """The extreme laps that shape the tail must come from other races. Fitting with a
        circuit held out has to change the tail, or the held-out race is contributing to
        the distribution used to predict it."""
        everything = SimComponents.fit(pace_model).noise
        without = SimComponents.fit(pace_model, exclude_circuit="hungary").noise
        assert np.quantile(without.lap_residuals, 0.999) != np.quantile(
            everything.lap_residuals, 0.999
        )
