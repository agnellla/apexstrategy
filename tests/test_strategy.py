"""Phase 9 unit tests: candidate pruning and the ranking engine.

Two families earn their place here. The pruning tests pin the filters that decide what the
engine is even allowed to recommend -- an illegal or physically impossible plan that slips
through is not a bad recommendation, it is a wrong one. ``TestPairing`` is the structural
one: it asserts that two candidates were simulated against the same luck, which is what
every delta, probability and risk number in the output silently assumes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from apexstrategy.sim.engine import RandomAlignment, SimComponents, alignment_for, simulate_field
from apexstrategy.sim.noise import NoiseModel
from apexstrategy.sim.scenario import MIN_STINT_LAPS, PitInstruction, Scenario
from apexstrategy.sim.state import DriverState, RaceState
from apexstrategy.strategy.candidates import (
    fit_max_stint_share,
    generate_candidates,
    is_physically_plausible,
    refine_candidates,
    stint_lengths,
    uses_two_dry_compounds,
)
from apexstrategy.strategy.engine import evaluate, recommend

TOTAL_LAPS = 70

#: Caps that make the arithmetic checkable by hand: on a 70-lap race, 41 / 34 / 26 laps.
CAPS = {"HARD": 0.586, "MEDIUM": 0.485, "SOFT": 0.379}


def _driver(code: str, compound: str = "MEDIUM", tyre_age: int = 20, used=("MEDIUM",), gap=0.0):
    return DriverState(
        code=code,
        team="team",
        gap_to_leader_s=gap,
        compound=compound,
        tyre_age=tyre_age,
        stint_number=1,
        compounds_used=frozenset(used),
        pit_stops=len(used) - 1,
        laps_completed=20,
        status="running",
        reference_pace_s=90.0,
        clean_laps_observed=15,
    )


def _state(lap: int = 20, drivers=None, **kw) -> RaceState:
    return RaceState(
        race_id="2024_test_R",
        circuit="testtrack",
        season=2024,
        round=1,
        lap=lap,
        total_laps=TOTAL_LAPS,
        track_status="green",
        drivers=drivers or (_driver("AAA"), _driver("BBB", gap=10.0)),
        air_temp=25.0,
        track_temp=40.0,
        rainfall=False,
        **kw,
    )


class _FlatModel:
    """Constant predicted delta, so the simulator's arithmetic stays checkable."""

    def predict(self, X):
        return np.zeros(len(X))

    def predict_dist(self, X, n, rng):
        return np.zeros((n, len(X)))


def _components(noise: bool = True) -> SimComponents:
    return SimComponents(
        pace_model=_FlatModel(),
        pit_loss_s=np.array([20.0, 22.0, 25.0]),
        pit_in_lap_share=0.2,
        # Hand-set rather than fitted: these tests are about the strategy layer, and a
        # fitted noise model would make them depend on the store.
        noise=NoiseModel(
            lap_residuals=np.linspace(-1.0, 2.0, 64),
            stint_level_sd=0.5,
            stint_slope_sd=0.05,
        )
        if noise
        else None,
    )


class TestMaxStintShare:
    def test_caps_are_ordered_by_compound_durability(self) -> None:
        share = fit_max_stint_share()
        assert set(share) == {"SOFT", "MEDIUM", "HARD"}
        assert share["SOFT"] < share["MEDIUM"] < share["HARD"] < 1.0

    def test_excluding_a_circuit_still_covers_every_compound(self) -> None:
        """LORO must not silently forbid a compound by leaving it with no observations."""
        share = fit_max_stint_share(exclude_circuit="hungary")
        assert set(share) == {"SOFT", "MEDIUM", "HARD"}
        assert all(v > 0 for v in share.values())


class TestPhysicalFilter:
    def test_the_set_already_on_the_car_counts_toward_its_life(self) -> None:
        """20 laps done on a MEDIUM capped at 34 leaves 14, so lap 45 is out of reach."""
        state = _state(lap=20)
        assert is_physically_plausible(state, "AAA", (PitInstruction(30, "HARD"),), CAPS)
        assert not is_physically_plausible(state, "AAA", (PitInstruction(45, "HARD"),), CAPS)

    def test_stint_lengths_start_from_where_the_tyre_was_fitted(self) -> None:
        lengths = stint_lengths(_state(lap=20), "AAA", (PitInstruction(30, "HARD"),))
        assert lengths == [30, 40]  # 20 already run + 10 more, then 30 -> 70

    def test_a_fresh_tyre_cannot_be_thrown_away_next_lap(self) -> None:
        fresh = _driver("AAA", "HARD", tyre_age=1, used=("MEDIUM", "HARD"))
        state = _state(lap=20, drivers=(fresh,))
        assert not is_physically_plausible(state, "AAA", (PitInstruction(21, "HARD"),), CAPS)
        assert is_physically_plausible(state, "AAA", (PitInstruction(30, "HARD"),), CAPS)

    def test_staying_out_is_capped_like_any_other_stint(self) -> None:
        """The plan with no stop is the one most likely to break the pace model.

        A 45-lap MEDIUM against a 34-lap cap makes the linear degradation term extrapolate
        past every stint in the store, and the phantom degradation then argues for a stop.
        Exempting it -- which an earlier version did, because there is no stop to filter --
        was worth a fictitious 28.6 s on Hamilton at Hungary lap 40.
        """
        state = _state(lap=45, drivers=(_driver("AAA", "MEDIUM", 20, ("MEDIUM", "HARD")),))
        assert stint_lengths(state, "AAA", ())[0] == 45 > CAPS["MEDIUM"] * TOTAL_LAPS
        assert not is_physically_plausible(state, "AAA", (), CAPS)
        assert () not in [s.plan for s in generate_candidates(state, "AAA", max_stint_share=CAPS)]

    def test_staying_out_survives_when_it_is_the_only_thing_left(self) -> None:
        """Five laps from the flag no stop leaves a stint worth running, so running to the
        end is not a strategy but the only remaining option. It has to be returned."""
        state = _state(lap=67, drivers=(_driver("AAA", "MEDIUM", 40, ("MEDIUM", "HARD")),))
        assert [s.plan for s in generate_candidates(state, "AAA", max_stint_share=CAPS)] == [()]

    def test_a_car_already_past_the_cap_still_gets_candidates(self) -> None:
        """A driver 40 laps into a MEDIUM is past anything observed, and pruning on that
        would empty the candidate set for exactly the car that most needs advice."""
        state = _state(lap=45, drivers=(_driver("AAA", "MEDIUM", tyre_age=45),))
        assert is_physically_plausible(state, "AAA", (PitInstruction(46, "HARD"),), CAPS)


class TestCandidateGeneration:
    def test_every_candidate_satisfies_the_two_compound_rule(self) -> None:
        state = _state(lap=20)
        candidates = generate_candidates(state, "AAA", max_stint_share=CAPS)
        assert candidates
        used = state.by_code["AAA"].compounds_used
        assert all(uses_two_dry_compounds(s, used) for s in candidates)

    def test_running_no_further_stops_is_offered_once_both_compounds_are_used(self) -> None:
        state = _state(lap=45, drivers=(_driver("AAA", "HARD", 15, ("MEDIUM", "HARD")),))
        plans = [s.plan for s in generate_candidates(state, "AAA", max_stint_share=CAPS)]
        assert () in plans

    def test_no_further_stops_is_withheld_when_it_would_be_illegal(self) -> None:
        """One compound used and no stop planned is not a strategy, it is a penalty."""
        state = _state(lap=45, drivers=(_driver("AAA", "MEDIUM", 15, ("MEDIUM",)),))
        plans = [s.plan for s in generate_candidates(state, "AAA", max_stint_share=CAPS)]
        assert () not in plans

    def test_every_candidate_survives_its_own_physical_filter(self) -> None:
        state = _state(lap=20)
        for scenario in generate_candidates(state, "AAA", max_stint_share=CAPS):
            lengths = stint_lengths(state, "AAA", scenario.plan)
            assert min(lengths) >= MIN_STINT_LAPS
            assert is_physically_plausible(state, "AAA", scenario.plan, CAPS)

    def test_an_immediate_stop_is_representable(self) -> None:
        """The undercut is stopping *now*, and it is the Hungary 2024 demo (D4).

        ``Scenario.is_legal`` rejects it because it measures the first stint from the
        decision lap rather than from where the tyre was fitted. That divergence is
        deliberate and documented; this pins it so it cannot be "tidied up" later.
        """
        state = _state(lap=20)
        candidates = generate_candidates(state, "AAA", max_stint_share=CAPS)
        immediate = [s for s in candidates if s.plan and s.plan[0].lap == 21]
        assert immediate
        assert not immediate[0].is_legal(frozenset({"MEDIUM"}), TOTAL_LAPS, from_lap=20)

    def test_a_retired_car_gets_no_recommendation(self) -> None:
        out = DriverState(**{**_driver("AAA").__dict__, "status": "retired"})
        state = _state(lap=20, drivers=(out,))
        with pytest.raises(ValueError, match="not running"):
            generate_candidates(state, "AAA", max_stint_share=CAPS)

    def test_the_surviving_set_stays_small_enough_to_read(self) -> None:
        candidates = generate_candidates(_state(lap=20), "AAA", max_stint_share=CAPS)
        assert 10 <= len(candidates) <= 150  # pruned, not exhaustive; see the module docstring


class TestRefinement:
    def test_only_the_next_stop_moves(self) -> None:
        state = _state(lap=20)
        around = [Scenario("AAA", (PitInstruction(30, "HARD"), PitInstruction(50, "MEDIUM")))]
        refined = refine_candidates(state, "AAA", around, spread=2, max_stint_share=CAPS)
        assert {s.plan[0].lap for s in refined} == {28, 29, 30, 31, 32}
        # The stop thirty laps away stays where it is: it will be re-decided at its own
        # decision lap, with evidence this state does not have.
        assert all(s.plan[1] == PitInstruction(50, "MEDIUM") for s in refined)

    def test_the_coarse_set_is_not_re_evaluated(self) -> None:
        state = _state(lap=20)
        coarse = generate_candidates(state, "AAA", max_stint_share=CAPS)
        refined = refine_candidates(state, "AAA", coarse[:5], max_stint_share=CAPS, exclude=coarse)
        seen = {tuple((p.lap, p.compound) for p in s.plan) for s in coarse}
        assert all(tuple((p.lap, p.compound) for p in s.plan) not in seen for s in refined)

    def test_refined_plans_remain_plausible(self) -> None:
        state = _state(lap=20)
        coarse = generate_candidates(state, "AAA", max_stint_share=CAPS)
        for scenario in refine_candidates(state, "AAA", coarse[:6], max_stint_share=CAPS):
            assert is_physically_plausible(state, "AAA", scenario.plan, CAPS)


class TestPairing:
    """The comparison is only meaningful if both plans met the same luck."""

    @staticmethod
    def _run(plan, align, seed=1, n_sims=64):
        state = _state(lap=20)
        return simulate_field(
            state,
            {"AAA": plan},
            _components(),
            n_sims=n_sims,
            seed=seed,
            to_lap=TOTAL_LAPS,
            align=align,
        )

    def test_alignment_survives_a_change_in_stop_count(self) -> None:
        """Two plans sharing a first stint must produce a bitwise-identical first lap.

        Everything about lap 21 is settled before either plan diverges, so any difference
        is the random stream having moved -- which is exactly what an extra stop does to
        it. This is the assertion that fails if the aligned draw sizes are removed.
        """
        one = (PitInstruction(30, "HARD"),)
        two = (PitInstruction(30, "HARD"), PitInstruction(50, "MEDIUM"))
        align = RandomAlignment(stint_slots=3, stop_slots=2)

        aligned = [self._run(p, align).lap_times_s[:, 0, 0] for p in (one, two)]
        np.testing.assert_array_equal(*aligned)

        drifted = [self._run(p, None).lap_times_s[:, 0, 0] for p in (one, two)]
        assert not np.array_equal(*drifted)

    def test_pairing_shrinks_the_spread_of_a_comparison(self) -> None:
        align = RandomAlignment(stint_slots=3, stop_slots=2)
        a = (PitInstruction(30, "HARD"),)
        b = (PitInstruction(36, "HARD"),)
        paired = self._run(a, align, seed=1).elapsed_s("AAA") - self._run(
            b, align, seed=1
        ).elapsed_s("AAA")
        independent = self._run(a, align, seed=1).elapsed_s("AAA") - self._run(
            b, align, seed=2
        ).elapsed_s("AAA")
        assert paired.std() < independent.std() / 3

    def test_alignment_refuses_to_silently_lose_the_pairing(self) -> None:
        """Too-narrow slots must raise. Quietly falling back would produce a delta and a
        win probability that look fine and are not paired, which is the worst outcome."""
        two_stops = (PitInstruction(30, "HARD"), PitInstruction(50, "MEDIUM"))
        with pytest.raises(ValueError, match="stints scheduled"):
            self._run(two_stops, RandomAlignment(stint_slots=2, stop_slots=2))
        with pytest.raises(ValueError, match="stop_slots"):
            self._run(two_stops, RandomAlignment(stint_slots=3, stop_slots=1))

    def test_alignment_for_sizes_over_the_whole_field(self) -> None:
        """A rival with more stops than the ego still sets the floor."""
        plans = [
            {"AAA": (PitInstruction(30, "HARD"),), "BBB": ()},
            {"AAA": (), "BBB": (PitInstruction(30, "HARD"), PitInstruction(50, "SOFT"))},
        ]
        assert alignment_for(plans, from_lap=20) == RandomAlignment(stint_slots=3, stop_slots=2)

    def test_stops_already_taken_do_not_widen_the_draw(self) -> None:
        plans = [{"AAA": (PitInstruction(10, "HARD"), PitInstruction(30, "SOFT"))}]
        assert alignment_for(plans, from_lap=20) == RandomAlignment(stint_slots=2, stop_slots=1)


class TestRanking:
    @staticmethod
    def _ranking(**kw):
        state = _state(lap=20)
        candidates = generate_candidates(state, "AAA", max_stint_share=CAPS)[:8]
        return state, evaluate(state, "AAA", candidates, _components(), n_sims=128, **kw)

    def test_the_baseline_measures_zero_against_itself(self) -> None:
        _, ranking = self._ranking()
        row = ranking.table().query("is_baseline").iloc[0]
        assert row["delta_s"] == 0.0
        assert row["p_faster"] == 0.0
        assert row["cvar10_delta_s"] == 0.0

    def test_rank_one_is_the_recommendation(self) -> None:
        _, ranking = self._ranking()
        table = ranking.table()
        assert table["rank"].tolist() == sorted(table["rank"].tolist())
        assert table["mean_s"].is_monotonic_increasing
        assert table.iloc[0]["label"] == ranking.best.label

    def test_regret_is_non_negative_and_probabilities_are_probabilities(self) -> None:
        _, ranking = self._ranking()
        table = ranking.table()
        assert (table["regret_s"] >= 0).all()
        assert (table["cvar10_regret_s"] >= table["regret_s"] - 1e-9).all()
        assert table["p_faster"].between(0.0, 1.0).all()

    def test_an_explicit_baseline_is_used_and_must_exist(self) -> None:
        state = _state(lap=20)
        candidates = generate_candidates(state, "AAA", max_stint_share=CAPS)[:8]
        ranking = evaluate(
            state, "AAA", candidates, _components(), n_sims=64, baseline=candidates[3]
        )
        assert ranking.baseline.scenario.plan == candidates[3].plan

        absent = Scenario("AAA", (PitInstruction(60, "SOFT"),))
        with pytest.raises(ValueError, match="not among the evaluated"):
            evaluate(state, "AAA", candidates, _components(), n_sims=64, baseline=absent)

    def test_is_deterministic_for_a_seed(self) -> None:
        a = self._ranking(seed=5)[1].table()
        b = self._ranking(seed=5)[1].table()
        assert a.equals(b)

    def test_result_is_json_serialisable(self) -> None:
        _, ranking = self._ranking()
        payload = json.loads(json.dumps(ranking.to_dict()))  # D9
        assert payload["objective"] == "expected_race_time_s"
        assert payload["rival_policy"] == "sim_pace"  # the tier travels with the number
        assert len(payload["candidates"]) == len(ranking.evaluations)

    def test_refuses_an_empty_candidate_set(self) -> None:
        with pytest.raises(ValueError, match="no candidates"):
            evaluate(_state(lap=20), "AAA", [], _components())


class TestRecommend:
    def test_refined_plans_join_the_same_ranking(self) -> None:
        state = _state(lap=20)
        ranking = recommend(state, "AAA", _components(), n_sims=64, max_stint_share=CAPS)
        coarse = generate_candidates(state, "AAA", max_stint_share=CAPS)
        assert len(ranking.evaluations) > len(coarse)
        assert ranking.wall_time_s > 0

    def test_the_baseline_is_added_when_the_grid_missed_it(self) -> None:
        state = _state(lap=20)
        odd = Scenario("AAA", (PitInstruction(27, "HARD"),), label="as executed")
        ranking = recommend(
            state, "AAA", _components(), n_sims=64, baseline=odd, max_stint_share=CAPS
        )
        assert ranking.baseline.scenario.plan == odd.plan
        assert ranking.table().query("is_baseline")["delta_s"].iloc[0] == 0.0

    def test_rivals_do_not_move_the_ego_race_time(self) -> None:
        """D5 defers overtaking and traffic, so no driver's lap time depends on another's.

        Asserting it keeps the claim honest: if product-Phase 2 couples them, this test
        fails and the ``rival_policy`` label on every result has to start meaning something.
        """
        state = _state(lap=20)
        candidates = generate_candidates(state, "AAA", max_stint_share=CAPS)[:4]
        kw = {"n_sims": 64, "seed": 3}
        alone = evaluate(state, "AAA", candidates, _components(), **kw)
        with_rival = evaluate(
            state,
            "AAA",
            candidates,
            _components(),
            rival_plans={"BBB": (PitInstruction(35, "SOFT"),)},
            **kw,
        )
        np.testing.assert_allclose(
            alone.evaluations[0].race_time_s, with_rival.evaluations[0].race_time_s
        )
