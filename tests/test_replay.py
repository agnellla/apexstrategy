"""Phase 7 integration tests: the replay gate, run against the built store.

These are regression guards on the *result*, not the logic. The numbers they assert are
the ones recorded in the Phase 7 build log, so a change that quietly degrades the
simulator fails here rather than being discovered at Phase 9.
"""

from __future__ import annotations

import numpy as np
import pytest

from apexstrategy import config
from apexstrategy.data import store
from apexstrategy.evaluation.replay import (
    GATE_RACE_TIME_PCT,
    loro_pace_model,
    replay_race,
)
from apexstrategy.sim.engine import SimComponents, fit_pit_loss
from apexstrategy.sim.scenario import observed_plans
from apexstrategy.sim.state import build_state

pytestmark = pytest.mark.skipif(not store.table_path("laps").exists(), reason="store not built")

GATE_CIRCUIT = "hungary"  # the demo race (D4)


@pytest.fixture(scope="module")
def held_out():
    model = loro_pace_model(GATE_CIRCUIT)
    return model, SimComponents.fit(model, exclude_circuit=GATE_CIRCUIT)


@pytest.fixture(scope="module")
def result(held_out):
    model, components = held_out
    return replay_race(
        GATE_CIRCUIT, from_lap=25, n_sims=500, pace_model=model, components=components
    )


class TestGate:
    def test_replay_reproduces_the_race_within_tolerance(self, result) -> None:
        """The Phase 7 hard gate. If this fails, no counterfactual downstream is worth
        anything and the strategy engine must not be built on it."""
        pct = result.metrics()["race_time_median_pct"]
        assert pct <= GATE_RACE_TIME_PCT, f"race-time error {pct:.2f}%"

    def test_every_driver_is_projected_and_compared(self, result) -> None:
        assert len(result.races) >= 18
        assert (result.races["laps_compared"] > 0).all()

    def test_clean_lap_error_is_in_the_recorded_range(self, result) -> None:
        """Guards against a regression that leaves race time right by cancelling errors."""
        assert result.metrics()["clean_lap_mae_s"] < 1.0

    def test_error_decomposition_sums_to_the_total(self, result) -> None:
        """The explanation is arithmetic, so it has to add up (Challenge 6)."""
        decomp = result.decomposition()
        assert decomp["total_error_s"].sum() == pytest.approx(result.laps["error_s"].sum())
        assert decomp["share_of_total"].sum() == pytest.approx(1.0)


class TestCalibration:
    """Phase 8: the intervals have to mean what they say."""

    def test_hierarchical_noise_is_on_by_default(self, held_out) -> None:
        """i.i.d. per-lap sampling covered 11-28% at race level. Falling back to it
        silently would look fine on every point-estimate metric."""
        _, components = held_out
        assert components.noise is not None

    def test_race_time_intervals_are_not_collapsed(self, result) -> None:
        m = result.metrics()
        assert m["cover_80"] > 0.5, "variance has collapsed back to the Phase 7 behaviour"

    def test_pit_values_span_the_unit_interval(self, result) -> None:
        """Under-dispersed intervals pile the PIT up at 0 and 1 and leave the middle
        empty; this is the cheap version of that check."""
        pit = result.races["pit"]
        assert pit.between(0.0, 1.0).all()
        assert 0.1 < pit.std() < 0.45

    def test_hierarchical_noise_widens_without_moving_the_point_estimate(self, held_out) -> None:
        """The decomposition is about spread. If it shifts the median it is a bug."""
        import dataclasses

        model, components = held_out
        flat = dataclasses.replace(components, noise=None)
        args = {"from_lap": 25, "n_sims": 400, "pace_model": model}
        wide = replay_race(GATE_CIRCUIT, components=components, **args)
        narrow = replay_race(GATE_CIRCUIT, components=flat, **args)

        assert wide.metrics()["cover_80"] > narrow.metrics()["cover_80"] + 0.2
        assert wide.metrics()["race_time_median_pct"] == pytest.approx(
            narrow.metrics()["race_time_median_pct"], abs=0.15
        )


class TestLeaveOneRaceOut:
    """The replay must not be fed anything estimated from the race it is replaying."""

    def test_pit_loss_excludes_the_replayed_circuit(self) -> None:
        laps = store.read_table("laps")
        allowed = {c.key for c in config.TRAINING_CIRCUITS}
        model = loro_pace_model(GATE_CIRCUIT)

        with_it, _ = fit_pit_loss(laps[laps["circuit"].isin(allowed)], model)
        without = SimComponents.fit(model, exclude_circuit=GATE_CIRCUIT).pit_loss_s
        assert len(without) < len(with_it)

    def test_pace_model_never_sees_the_replayed_circuit(self) -> None:
        """Fitted on four circuits, not five -- circuit constants leak like weights do
        (Part 13, risk 5)."""
        held = loro_pace_model(GATE_CIRCUIT)
        everything = loro_pace_model("__none__")
        assert held.model_.nobs < everything.model_.nobs

    def test_pit_loss_is_measured_in_the_simulator_own_currency(self) -> None:
        """Fitted against reference + model prediction, not Phase 4's local clean pace.
        Reusing the Phase 4 number credited every out-lap with a fresh-tyre bonus it never
        enjoyed and under-charged each stop by ~1.7s."""
        model = loro_pace_model(GATE_CIRCUIT)
        components = SimComponents.fit(model, exclude_circuit=GATE_CIRCUIT)
        assert 15.0 < np.median(components.pit_loss_s) < 30.0
        assert 0.05 < components.pit_in_lap_share < 0.35


class TestStateAgainstRealData:
    def test_state_matches_the_official_order_at_the_top(self) -> None:
        laps = store.race_laps(GATE_CIRCUIT)
        state = build_state(laps, 25, store.race_total_laps(GATE_CIRCUIT))
        leaders = [d.code for d in state.order()[:5]]
        official = (
            laps[laps["lap_number"] == 25].nsmallest(5, "cumulative_time_s")["driver"].tolist()
        )
        assert leaders == official

    def test_firewall_holds_on_real_laps(self) -> None:
        laps = store.race_laps(GATE_CIRCUIT)
        total = store.race_total_laps(GATE_CIRCUIT)
        poisoned = laps.copy()
        future = poisoned["lap_number"] > 25
        poisoned.loc[future, ["lap_time_s", "cumulative_time_s"]] = [999.0, 1e9]
        assert build_state(laps, 25, total) == build_state(poisoned, 25, total)

    def test_observed_plans_match_the_pits_table(self) -> None:
        """Plans are recovered from lap flags; the pits table is the independent record.

        The one legitimate disagreement is a car that pitted and never rejoined -- Hungary
        2024 GAS stopped on lap 33 and retired there. With no out-lap the compound fitted
        is unknowable, and a stop the car never took the restart on is not a strategy the
        simulator can replay, so it is dropped rather than guessed at.
        """
        laps = store.race_laps(GATE_CIRCUIT)
        pits = store.read_table("pits")
        # Season-scoped: ``circuit`` alone stopped identifying a race once a held-out
        # season entered the store, and an unscoped filter here silently compared 2024 lap
        # flags against two seasons of stops.
        pits = pits[(pits["circuit"] == GATE_CIRCUIT) & (pits["season"] == config.SEASON)]
        plans = observed_plans(laps)

        for driver, expected in pits.groupby("driver")["lap"].apply(sorted).items():
            recovered = [p.lap for p in plans.get(driver, ())]
            dropped = [lap for lap in expected if lap not in recovered]
            assert recovered == [lap for lap in expected if lap not in dropped], driver
            last_lap = laps.loc[laps["driver"] == driver, "lap_number"].max()
            assert all(lap >= last_lap for lap in dropped), (driver, dropped)
