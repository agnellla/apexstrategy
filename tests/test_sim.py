"""Phase 6/7 unit tests: race state, scenarios, and the simulator.

The tests that earn their place here are the structural ones. ``TestTemporalFirewall`` is
the important one -- it turns "does anything read the future?" from a code-review question
into a CI failure, which is the only form of that guarantee worth having.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexstrategy.sim.engine import SimComponents, build_schedule, simulate_field
from apexstrategy.sim.scenario import PitInstruction, Scenario, observed_plans
from apexstrategy.sim.state import build_state, estimate_reference_pace, resolve_track_status

TOTAL_LAPS = 40


def _race(n_laps: int = 30, drivers: tuple[str, ...] = ("AAA", "BBB", "CCC")) -> pd.DataFrame:
    """A synthetic race: three drivers, one stop each on lap 12, known pace ordering."""
    rows = []
    for i, driver in enumerate(drivers):
        base, clock = 90.0 + i, 0.0
        compound, age, stint = "MEDIUM", 0, 1
        for lap in range(1, n_laps + 1):
            pit_in, pit_out = lap == 12, lap == 13
            if pit_out:
                compound, age, stint = "HARD", 0, 2
            age += 1
            t = base + 0.05 * age + (20.0 if pit_in else 0.0) + (5.0 if lap == 1 else 0.0)
            clock += t
            rows.append(
                {
                    "race_id": "2024_test_R",
                    "season": 2024,
                    "round": 1,
                    "circuit": "testtrack",
                    "driver": driver,
                    "team": f"team{i}",
                    "lap_number": lap,
                    "lap_time_s": t,
                    "cumulative_time_s": clock,
                    "compound": compound,
                    "tyre_age": float(age),
                    "is_fresh_tyre": age == 1,
                    "stint_number": stint,
                    "position": float(i + 1),
                    "derived_position": i + 1,
                    "gap_to_leader_s": clock,  # overwritten below
                    "gap_ahead_s": np.nan,
                    "entering_gap_s": 9.0,
                    "in_dirty_air": False,
                    "is_pit_in": pit_in,
                    "is_pit_out": pit_out,
                    "track_status": "1",
                    "is_clean": not (pit_in or pit_out or lap == 1),
                    "exclusion_reason": "clean",
                    "track_temp": 40.0,
                    "air_temp": 25.0,
                    "humidity": 40.0,
                    "wind_speed": 2.0,
                    "rainfall": False,
                }
            )
    laps = pd.DataFrame(rows)
    leader = laps.groupby("lap_number")["cumulative_time_s"].transform("min")
    laps["gap_to_leader_s"] = laps["cumulative_time_s"] - leader
    return laps


class TestTemporalFirewall:
    """``build_state`` must be a function of laps <= L and nothing else."""

    @pytest.mark.parametrize("lap", [3, 10, 15, 25])
    def test_future_laps_cannot_influence_the_state(self, lap: int) -> None:
        """The strong form: corrupt every lap after L and demand an identical state.

        A weaker truncation test only proves the two code paths agree. This proves no
        future row is read at all, which is the actual guarantee (Part 13).
        """
        laps = _race()
        poisoned = laps.copy()
        future = poisoned["lap_number"] > lap
        poisoned.loc[future, "lap_time_s"] = 999.0
        poisoned.loc[future, "cumulative_time_s"] = 1e9
        poisoned.loc[future, "compound"] = "SOFT"
        poisoned.loc[future, "is_pit_in"] = True
        poisoned.loc[future, "is_clean"] = False

        assert build_state(laps, lap, TOTAL_LAPS) == build_state(poisoned, lap, TOTAL_LAPS)

    @pytest.mark.parametrize("lap", [5, 12, 20])
    def test_truncating_the_input_changes_nothing(self, lap: int) -> None:
        laps = _race()
        truncated = laps[laps["lap_number"] <= lap]
        assert build_state(laps, lap, TOTAL_LAPS) == build_state(truncated, lap, TOTAL_LAPS)

    def test_rejects_a_multi_race_frame(self) -> None:
        laps = pd.concat([_race(), _race().assign(race_id="2024_other_R")])
        with pytest.raises(ValueError, match="one race"):
            build_state(laps, 10, TOTAL_LAPS)


class TestRaceState:
    def test_position_is_derived_from_gap_not_stored(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        assert not hasattr(state.drivers[0], "position")
        assert [d.code for d in state.order()] == ["AAA", "BBB", "CCC"]
        assert state.position_of("BBB") == 2

    def test_tyre_and_stint_state_reflect_the_pit_stop(self) -> None:
        before = build_state(_race(), 11, TOTAL_LAPS).by_code["AAA"]
        after = build_state(_race(), 15, TOTAL_LAPS).by_code["AAA"]
        assert (before.compound, before.stint_number, before.pit_stops) == ("MEDIUM", 1, 0)
        assert (after.compound, after.stint_number, after.pit_stops) == ("HARD", 2, 1)
        assert after.tyre_age == 3  # laps 13, 14, 15 on the new set
        assert after.compounds_used == frozenset({"MEDIUM", "HARD"})

    def test_a_stopped_car_reads_as_retired_and_a_lapped_one_does_not(self) -> None:
        laps = _race()
        stopped = laps[~((laps["driver"] == "CCC") & (laps["lap_number"] > 8))]
        state = build_state(stopped, 15, TOTAL_LAPS)
        assert state.by_code["CCC"].status == "retired"
        assert "CCC" not in [d.code for d in state.order()]

        # A car a lap down is still crossing the line and must stay in the projection.
        lapped = laps[~((laps["driver"] == "CCC") & (laps["lap_number"] > 14))]
        assert build_state(lapped, 15, TOTAL_LAPS).by_code["CCC"].status == "running"

    def test_laps_remaining_and_json_round_trip(self) -> None:
        import json

        state = build_state(_race(), 10, TOTAL_LAPS)
        assert state.laps_remaining == 30
        json.dumps(state.to_dict())  # D9: results must be JSON-serialisable

    def test_reference_pace_recovers_each_driver_own_level(self) -> None:
        """Without a model this is the raw clean-lap median, which must still separate
        cars that are a second apart."""
        state = build_state(_race(), 25, TOTAL_LAPS)
        refs = {d.code: d.reference_pace_s for d in state.drivers}
        assert refs["BBB"] - refs["AAA"] == pytest.approx(1.0, abs=0.2)

    def test_reference_falls_back_to_green_laps_when_no_clean_ones_exist(self) -> None:
        """A car stuck in traffic all race generates no clean laps. Falling back to the
        field's pace hands a backmarker the field's speed -- the bug this replaced."""
        laps = _race()
        laps.loc[laps["driver"] == "CCC", ["is_clean", "in_dirty_air"]] = [False, True]
        state = build_state(laps, 25, TOTAL_LAPS)
        assert state.by_code["CCC"].reference_pace_s > state.by_code["AAA"].reference_pace_s

    def test_reference_pace_is_nan_free_with_no_usable_laps(self) -> None:
        empty = _race().head(0)
        assert np.isnan(estimate_reference_pace(empty))


class TestTrackStatus:
    def test_concatenated_codes_resolve_to_the_most_severe(self) -> None:
        """FastF1 concatenates every code active during the lap: '1254' is
        AllClear+Yellow+SC+Red. Equality matching would miss all of it."""
        assert resolve_track_status(pd.Series(["1254"])) == "red"
        assert resolve_track_status(pd.Series(["14"])) == "sc"
        assert resolve_track_status(pd.Series(["671"])) == "vsc"
        assert resolve_track_status(pd.Series(["1"])) == "green"


class TestScenario:
    def test_two_compound_rule_is_enforced(self) -> None:
        one_stop = Scenario("AAA", (PitInstruction(20, "HARD"),))
        assert one_stop.is_legal(frozenset({"MEDIUM"}), TOTAL_LAPS, 10)
        # Already on HARD, stopping for HARD again: never ran a second compound.
        assert not one_stop.is_legal(frozenset({"HARD"}), TOTAL_LAPS, 10)

    def test_implausibly_short_stints_are_rejected(self) -> None:
        plan = Scenario("AAA", (PitInstruction(38, "HARD"),))
        assert not plan.is_legal(frozenset({"MEDIUM"}), TOTAL_LAPS, 10)

    def test_wet_compounds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="dry-only"):
            PitInstruction(20, "INTERMEDIATE")

    def test_plan_must_be_sorted_and_unique(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            Scenario("AAA", (PitInstruction(30, "HARD"), PitInstruction(15, "SOFT")))
        with pytest.raises(ValueError, match="same lap"):
            Scenario("AAA", (PitInstruction(15, "HARD"), PitInstruction(15, "SOFT")))

    def test_observed_plans_recover_the_executed_stop(self) -> None:
        plans = observed_plans(_race())
        assert plans["AAA"] == (PitInstruction(12, "HARD"),)
        assert observed_plans(_race(), after_lap=20)["AAA"] == ()


class TestSchedule:
    def test_pit_lap_switches_compound_on_the_following_lap(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        sched = build_schedule(state, {"AAA": (PitInstruction(20, "SOFT"),)}, to_lap=25)
        aaa = sched[sched["driver"] == "AAA"].set_index("lap_number")

        assert aaa.loc[20, "compound"] == "MEDIUM"  # in-lap runs the old tyre
        assert aaa.loc[20, "is_pit_in"]
        assert (aaa.loc[21, "compound"], aaa.loc[21, "tyre_age"]) == ("SOFT", 1.0)
        assert aaa.loc[21, "is_pit_out"] and aaa.loc[21, "stint_number"] == 2
        assert aaa.loc[22, "tyre_age"] == 2.0

    def test_tyre_age_continues_from_the_state(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        sched = build_schedule(state, {}, to_lap=12)
        first = sched[sched["driver"] == "AAA"].iloc[0]
        assert first["lap_number"] == 11
        assert first["tyre_age"] == state.by_code["AAA"].tyre_age + 1

    def test_stops_already_taken_are_ignored(self) -> None:
        state = build_state(_race(), 20, TOTAL_LAPS)
        sched = build_schedule(state, {"AAA": (PitInstruction(12, "HARD"),)}, to_lap=25)
        assert not sched["is_pit_in"].any()


class _FlatModel:
    """A pace model that predicts a constant delta with no residual spread.

    Lets the simulator's arithmetic be asserted exactly, which a bootstrapped model
    cannot. Implements the same ``predict``/``predict_dist`` contract as the real ones.
    """

    def __init__(self, delta: float = 0.0) -> None:
        self.delta = delta

    def predict(self, X):
        return np.full(len(X), self.delta)

    def predict_dist(self, X, n, rng):
        return np.full((n, len(X)), self.delta)


def _components(**kw) -> SimComponents:
    return SimComponents(
        pace_model=kw.pop("pace_model", _FlatModel()),
        pit_loss_s=kw.pop("pit_loss_s", np.array([20.0])),
        **kw,
    )


class TestSimulator:
    def test_is_bitwise_deterministic_for_a_seed(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        plans = {"AAA": (PitInstruction(20, "SOFT"),)}
        components = _components(pace_model=_FlatModel())
        a = simulate_field(state, plans, components, n_sims=16, seed=3, to_lap=30)
        b = simulate_field(state, plans, components, n_sims=16, seed=3, to_lap=30)
        np.testing.assert_array_equal(a.lap_times_s, b.lap_times_s)

    def test_shape_and_cumulative_time_is_monotonic(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        sim = simulate_field(state, {}, _components(), n_sims=8, seed=0, to_lap=30)
        assert sim.lap_times_s.shape == (8, 3, 20)
        assert (sim.lap_times_s > 0).all()
        for lap in range(11, 31):
            assert (sim.elapsed_s("AAA", lap) >= sim.elapsed_s("AAA", lap - 1)).all()

    def test_pit_loss_is_conserved_across_the_in_and_out_lap(self) -> None:
        """The stop's cost is split for attribution, never created or destroyed."""
        state = build_state(_race(), 10, TOTAL_LAPS)
        components = _components(pit_loss_s=np.array([20.0]), pit_in_lap_share=0.2)
        no_stop = simulate_field(state, {}, components, n_sims=4, seed=1, to_lap=30)
        stop = simulate_field(
            state, {"AAA": (PitInstruction(20, "SOFT"),)}, components, n_sims=4, seed=1, to_lap=30
        )
        extra = stop.elapsed_s("AAA") - no_stop.elapsed_s("AAA")
        assert extra == pytest.approx(20.0, abs=1e-9)

    def test_reference_pace_sets_the_level(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        sim = simulate_field(state, {}, _components(pace_model=_FlatModel(0.5)), 4, 0, to_lap=11)
        assert sim.lap_times_s[0, 0, 0] == pytest.approx(
            state.by_code["AAA"].reference_pace_s + 0.5
        )

    def test_race_progress_adjustment_tilts_later_laps(self) -> None:
        laps = _race()
        base = build_state(laps, 10, TOTAL_LAPS, race_progress_adjustment=0.0)
        tilted = build_state(laps, 10, TOTAL_LAPS, race_progress_adjustment=-0.02)
        c = _components()
        flat = simulate_field(base, {}, c, 2, 0, to_lap=30).lap_times_s[0, 0]
        slope = simulate_field(tilted, {}, c, 2, 0, to_lap=30).lap_times_s[0, 0]
        assert slope[0] == pytest.approx(flat[0] - 0.02)
        assert slope[-1] == pytest.approx(flat[-1] - 0.02 * 20)

    def test_caution_laps_slow_the_whole_field(self) -> None:
        state = build_state(_race(), 10, TOTAL_LAPS)
        c = _components(caution_lap_multiplier=1.5)
        green = simulate_field(state, {}, c, 2, 0, to_lap=20)
        under = simulate_field(state, {}, c, 2, 0, to_lap=20, caution_laps=frozenset({15}))
        ratio = under.lap_times_s[:, :, 4] / green.lap_times_s[:, :, 4]
        assert ratio == pytest.approx(1.5)
        np.testing.assert_allclose(under.lap_times_s[:, :, 5], green.lap_times_s[:, :, 5])

    def test_retired_drivers_are_not_projected(self) -> None:
        laps = _race()
        laps = laps[~((laps["driver"] == "CCC") & (laps["lap_number"] > 5))]
        state = build_state(laps, 15, TOTAL_LAPS)
        sim = simulate_field(state, {}, _components(), 2, 0, to_lap=20)
        assert "CCC" not in sim.drivers

    def test_refuses_to_simulate_backwards(self) -> None:
        state = build_state(_race(), 20, TOTAL_LAPS)
        with pytest.raises(ValueError, match="nothing to simulate"):
            simulate_field(state, {}, _components(), 2, 0, to_lap=20)

    def test_result_is_json_serialisable(self) -> None:
        import json

        state = build_state(_race(), 10, TOTAL_LAPS)
        sim = simulate_field(state, {"AAA": (PitInstruction(20, "SOFT"),)}, _components(), 8, 0)
        payload = json.dumps(sim.to_dict())  # D9
        assert "rival_policy" in payload  # the tier must travel with the number
        assert len(sim.summary()) == 3
