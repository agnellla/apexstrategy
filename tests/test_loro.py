"""Tests for leave-one-race-out validation.

The critical property is that a fold never sees its own held-out circuit. A silent
leak here would make every downstream metric optimistic and is exactly the failure the
architecture doc ranks third among leakage risks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexstrategy.evaluation.loro import run_loro, summarise
from apexstrategy.models.pace import TARGET, LinearPaceBaseline, add_reference_pace


def _multi_circuit(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for circuit, base in [("a", 80.0), ("b", 92.0), ("c", 71.0)]:
        for driver in ("X", "Y"):
            for stint, (compound, start) in enumerate([("MEDIUM", 1), ("HARD", 26)], start=1):
                for age in range(1, 26):
                    rows.append(
                        {
                            "race_id": f"2024_{circuit}_R",
                            "circuit": circuit,
                            "driver": driver,
                            "compound": compound,
                            "stint_number": stint,
                            "tyre_age": float(age),
                            "lap_number": start + age - 1,
                            "track_temp": 40.0,
                            "air_temp": 25.0,
                            "is_fresh_tyre": age == 1,
                            "entering_gap_s": 5.0,
                            "is_clean": True,
                            "lap_time_s": base + 0.05 * age + rng.normal(0, 0.05),
                        }
                    )
    return add_reference_pace(pd.DataFrame(rows))


def test_one_fold_per_circuit() -> None:
    laps = _multi_circuit()
    results = run_loro(laps, LinearPaceBaseline)
    assert set(results["held_out"]) == {"a", "b", "c"}


def test_folds_partition_the_data() -> None:
    """Train and test sizes must sum to the whole set in every fold -- no overlap,
    nothing silently dropped."""
    laps = _multi_circuit()
    results = run_loro(laps, LinearPaceBaseline)
    assert (results["n_train"] + results["n_test"] == len(laps)).all()


def test_held_out_circuit_is_absent_from_training(monkeypatch) -> None:
    seen = []

    class Spy(LinearPaceBaseline):
        def fit(self, X, y):
            seen.append(set(X["circuit"].astype(str).unique()))
            return super().fit(X, y)

    laps = _multi_circuit()
    results = run_loro(laps, Spy)
    for held_out, train_circuits in zip(results["held_out"], seen, strict=True):
        assert held_out not in train_circuits


def test_target_normalisation_is_per_race_so_folds_stay_comparable() -> None:
    """Circuits differ by 20s in baseline pace. The target must be centred within each
    race, otherwise LORO error would be dominated by circuit level rather than by how
    well pace evolution is predicted."""
    laps = _multi_circuit()
    per_circuit_median = laps.groupby("circuit")[TARGET].median()
    assert per_circuit_median.abs().max() < 0.05


def test_summarise_reports_spread_across_folds() -> None:
    results = run_loro(_multi_circuit(), LinearPaceBaseline)
    summary = summarise(results, "linear")
    assert summary["mae_worst"] >= summary["mae_mean"]
    assert summary["model"] == "linear"
