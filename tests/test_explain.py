"""Phase 10 unit tests: the explanation decomposition.

The exit criterion is that the components sum to the total, so that is what most of these
assert -- but summing to the total is cheap to fake with a residual bucket. The tests that
earn their place check the *split* against a closed form, and check the two claims the
decomposition makes about what it leaves out: that reference pace and race progress cancel
between two plans, and that the sampling row is Monte Carlo residue rather than a cause.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from apexstrategy.sim.engine import SimComponents, build_schedule, simulate_field
from apexstrategy.sim.noise import NoiseModel
from apexstrategy.sim.scenario import PitInstruction, Scenario
from apexstrategy.sim.state import DriverState, RaceState
from apexstrategy.strategy.explain import (
    RECONCILIATION_TOLERANCE_S,
    caution_option_value,
    explain,
)

TOTAL_LAPS = 70
EGO = "AAA"

#: Two plans differing in both stop lap and stint structure, so every row does work.
EARLY = Scenario(EGO, (PitInstruction(25, "HARD"),), label="early")
LATE = Scenario(EGO, (PitInstruction(40, "HARD"),), label="late")


class _DegradingModel:
    """``delta = offset[compound] + rate[compound] * tyre_age``.

    Deliberately separable in closed form, so the Shapley split can be checked against the
    right answer rather than against itself. Nothing else in the feature frame is read,
    which is also the point: the split must work through ``predict`` alone.
    """

    OFFSET = {"SOFT": -0.80, "MEDIUM": 0.00, "HARD": 0.60}
    RATE = {"SOFT": 0.090, "MEDIUM": 0.050, "HARD": 0.030}

    def predict(self, X):
        compound = X["compound"].astype(str).to_numpy()
        age = X["tyre_age"].to_numpy(dtype=float)
        offset = np.array([self.OFFSET[c] for c in compound])
        rate = np.array([self.RATE[c] for c in compound])
        return offset + rate * age

    def predict_dist(self, X, n, rng):
        return np.tile(self.predict(X), (n, 1))


def _silent_noise() -> NoiseModel:
    """A noise model that draws exactly zero, so the rest of the arithmetic is exact.

    Every scale has to be zeroed, the reference terms included -- they default to non-zero
    and would otherwise leave a random level on every lap.
    """
    return NoiseModel(
        lap_residuals=np.zeros(8),
        stint_level_sd=0.0,
        stint_slope_sd=0.0,
        reference_k=0.0,
        reference_shared_k=0.0,
        progress_sd=0.0,
    )


def _components(noise: NoiseModel | None = None, pit_loss: float = 22.0) -> SimComponents:
    return SimComponents(
        pace_model=_DegradingModel(),
        pit_loss_s=np.array([pit_loss]),
        pit_in_lap_share=0.2,
        noise=_silent_noise() if noise is None else noise,
    )


def _state(reference: float = 90.0, progress: float = 0.0, tyre_age: int = 20) -> RaceState:
    driver = DriverState(
        code=EGO,
        team="team",
        gap_to_leader_s=0.0,
        compound="MEDIUM",
        tyre_age=tyre_age,
        stint_number=1,
        compounds_used=frozenset({"MEDIUM"}),
        pit_stops=0,
        laps_completed=20,
        status="running",
        reference_pace_s=reference,
        clean_laps_observed=15,
    )
    return RaceState(
        race_id="2024_test_R",
        circuit="testtrack",
        season=2024,
        round=1,
        lap=20,
        total_laps=TOTAL_LAPS,
        track_status="green",
        drivers=(driver,),
        air_temp=25.0,
        track_temp=40.0,
        rainfall=False,
        race_progress_adjustment_s_per_lap=progress,
    )


class TestCapture:
    def test_capturing_does_not_change_the_simulation(self) -> None:
        state, c = _state(), _components()
        plans = {EGO: EARLY.plan}
        plain = simulate_field(state, plans, c, n_sims=32, seed=4, to_lap=TOTAL_LAPS)
        caught = simulate_field(state, plans, c, n_sims=32, seed=4, to_lap=TOTAL_LAPS, capture=True)
        np.testing.assert_array_equal(plain.lap_times_s, caught.lap_times_s)
        assert plain.breakdown is None

    def test_the_parts_reconstruct_every_lap(self) -> None:
        c = _components(noise=NoiseModel(np.linspace(-1, 2, 32), 0.5, 0.05))
        sim = simulate_field(
            _state(),
            {EGO: EARLY.plan},
            c,
            n_sims=32,
            seed=1,
            to_lap=TOTAL_LAPS,
            caution_laps=frozenset({30, 31}),
            capture=True,
        )
        b = sim.breakdown
        np.testing.assert_allclose(b.pace_s + b.noise_s + b.pit_s, sim.lap_times_s, atol=1e-12)

    def test_capture_refuses_the_phase_7_fallback(self) -> None:
        """``predict_dist`` draws the mean and the spread together; they cannot be split
        afterwards, and returning a breakdown that quietly called all of it 'pace' would
        be worse than refusing."""
        c = SimComponents(pace_model=_DegradingModel(), pit_loss_s=np.array([22.0]), noise=None)
        with pytest.raises(ValueError, match="needs a noise model"):
            simulate_field(_state(), {EGO: EARLY.plan}, c, n_sims=8, seed=0, capture=True)


class TestReconciliation:
    def test_components_sum_to_the_total(self) -> None:
        """The Phase 10 exit criterion, with real hierarchical noise in play."""
        c = _components(noise=NoiseModel(np.linspace(-1, 2, 64), 0.467, 0.049))
        why = explain(_state(), EGO, EARLY, LATE, c, n_sims=512)
        assert abs(why.residual_s) < RECONCILIATION_TOLERANCE_S
        parts = why.tyre_age_s + why.compound_s + why.pit_lane_s + why.sampling_s
        assert parts == pytest.approx(why.total_s, abs=1e-9)

    def test_a_plan_against_itself_explains_nothing(self) -> None:
        why = explain(_state(), EGO, EARLY, EARLY, _components(), n_sims=64)
        for value in (why.total_s, why.tyre_age_s, why.compound_s, why.pit_lane_s):
            assert value == pytest.approx(0.0, abs=1e-9)


class TestTheSplit:
    """The Shapley halves must match the closed form, not merely add up."""

    @staticmethod
    def _ages_and_compounds(plan):
        sched = build_schedule(_state(), {EGO: plan.plan}, TOTAL_LAPS)
        return sched["tyre_age"].to_numpy(float), sched["compound"].astype(str).to_numpy()

    def test_matches_the_closed_form_for_a_separable_model(self) -> None:
        age_a, comp_a = self._ages_and_compounds(EARLY)
        age_b, comp_b = self._ages_and_compounds(LATE)
        rate_a = np.array([_DegradingModel.RATE[c] for c in comp_a])
        rate_b = np.array([_DegradingModel.RATE[c] for c in comp_b])
        off_a = np.array([_DegradingModel.OFFSET[c] for c in comp_a])
        off_b = np.array([_DegradingModel.OFFSET[c] for c in comp_b])

        # age  = 1/2 (rate_a + rate_b)(age_a - age_b)
        # comp = (off_a - off_b) + 1/2 (rate_a - rate_b)(age_a + age_b)
        expected_age = (0.5 * (rate_a + rate_b) * (age_a - age_b)).sum()
        expected_compound = ((off_a - off_b) + 0.5 * (rate_a - rate_b) * (age_a + age_b)).sum()

        why = explain(_state(), EGO, EARLY, LATE, _components(), n_sims=32)
        assert why.tyre_age_s == pytest.approx(expected_age, abs=1e-9)
        assert why.compound_s == pytest.approx(expected_compound, abs=1e-9)

    def test_stopping_earlier_buys_fresher_rubber_and_pays_no_extra_pit_time(self) -> None:
        """Same stop count either side, so the whole difference is tyre state."""
        why = explain(_state(), EGO, EARLY, LATE, _components(), n_sims=64)
        assert why.tyre_age_s < 0  # 15 laps earlier onto a new set
        assert why.pit_lane_s == pytest.approx(0.0, abs=1e-9)

    def test_an_extra_stop_shows_up_as_pit_lane_time(self) -> None:
        one = Scenario(EGO, (PitInstruction(40, "HARD"),))
        two = Scenario(EGO, (PitInstruction(30, "HARD"), PitInstruction(52, "HARD")))
        why = explain(_state(), EGO, two, one, _components(pit_loss=22.0), n_sims=64)
        assert why.pit_lane_s == pytest.approx(22.0, abs=1e-9)


class TestWhatCancels:
    """The decomposition claims two whole effects vanish. Both are asserted, not asserted
    in prose."""

    def test_reference_pace_does_not_enter_a_comparison(self) -> None:
        slow = explain(_state(reference=95.0), EGO, EARLY, LATE, _components(), n_sims=64)
        fast = explain(_state(reference=85.0), EGO, EARLY, LATE, _components(), n_sims=64)
        assert slow.total_s == pytest.approx(fast.total_s, abs=1e-9)
        assert slow.tyre_age_s == pytest.approx(fast.tyre_age_s, abs=1e-9)

    def test_the_race_progress_trend_does_not_either(self) -> None:
        flat = explain(_state(progress=0.0), EGO, EARLY, LATE, _components(), n_sims=64)
        tilted = explain(_state(progress=-0.04), EGO, EARLY, LATE, _components(), n_sims=64)
        assert flat.total_s == pytest.approx(tilted.total_s, abs=1e-9)


class TestSamplingRow:
    def test_is_exactly_zero_when_nothing_is_drawn(self) -> None:
        why = explain(_state(), EGO, EARLY, LATE, _components(), n_sims=64)
        assert why.sampling_s == 0.0
        assert why.is_settled

    def test_shrinks_as_draws_are_added(self) -> None:
        """It is Monte Carlo residue, so it must fall towards zero rather than converge on
        some contribution of its own."""
        c = _components(noise=NoiseModel(np.linspace(-1, 2, 64), 0.467, 0.049))
        few = abs(explain(_state(), EGO, EARLY, LATE, c, n_sims=64).sampling_s)
        many = abs(explain(_state(), EGO, EARLY, LATE, c, n_sims=4096).sampling_s)
        assert many < few


class TestCautionContrast:
    def test_no_assumed_caution_means_no_option_value(self) -> None:
        value = caution_option_value(
            _state(), EGO, EARLY, LATE, _components(), frozenset(), n_sims=64
        )
        assert value == pytest.approx(0.0, abs=1e-9)

    def test_a_caution_moves_the_comparison(self) -> None:
        value = caution_option_value(
            _state(), EGO, EARLY, LATE, _components(), frozenset({26, 27, 28}), n_sims=64
        )
        assert value != 0.0


def test_explanation_is_json_serialisable_with_its_omissions() -> None:
    why = explain(_state(), EGO, EARLY, LATE, _components(), n_sims=64)
    payload = json.loads(json.dumps(why.to_dict()))  # D9
    assert payload["residual_s"] == pytest.approx(0.0, abs=RECONCILIATION_TOLERANCE_S)
    # The things the number does not contain travel with the number.
    terms = {row["term"] for row in payload["not_included"]}
    assert terms == {"traffic on rejoin", "safety-car option value"}
