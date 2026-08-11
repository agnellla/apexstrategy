"""Phase 0: the circuit registry is configuration, so its integrity is asserted here.

Round numbers in ``config.CIRCUITS`` are encoded beliefs. ``test_rounds_match_real_schedule``
verifies them against the actual FastF1 schedule; it is marked ``slow`` because it may
hit the network on a cold cache.
"""

from __future__ import annotations

import pytest

from apexstrategy import config


def test_circuit_keys_are_unique() -> None:
    keys = [c.key for c in config.CIRCUITS]
    assert len(keys) == len(set(keys))


def test_circuit_set_is_frozen_at_seven() -> None:
    """D3 freezes the circuit set until the pipeline runs end-to-end."""
    assert len(config.CIRCUITS) == 7


def test_monaco_is_stored_but_excluded_from_training() -> None:
    """D3: Monaco is a pathological outlier -- present in the store, absent from fitting."""
    monaco = config.CIRCUITS_BY_KEY["monaco"]
    assert monaco.use_for_training is False
    assert monaco.note, "an excluded circuit must record why"
    assert monaco not in config.TRAINING_CIRCUITS


def test_training_pool_is_six_circuits() -> None:
    assert len(config.TRAINING_CIRCUITS) == 6
    assert all(c.use_for_training for c in config.TRAINING_CIRCUITS)


def test_demo_circuit_is_in_the_training_pool() -> None:
    """D4: Hungary is the narrative race but must not be a special case in the data."""
    demo = config.CIRCUITS_BY_KEY[config.DEMO_CIRCUIT_KEY]
    assert demo in config.TRAINING_CIRCUITS


def test_telemetry_is_disabled() -> None:
    """D7. Flipping this silently would multiply the cache footprint ~90x."""
    assert config.LOAD_TELEMETRY is False


def test_ensure_dirs_is_idempotent() -> None:
    config.ensure_dirs()
    config.ensure_dirs()
    assert config.PROCESSED_DIR.is_dir()
    assert config.CACHE_DIR.is_dir()


@pytest.mark.slow
def test_rounds_match_real_schedule() -> None:
    """Encode-then-verify: config's round numbers must match the real 2024 schedule."""
    import fastf1

    config.setup_cache()
    schedule = fastf1.get_event_schedule(config.SEASON, include_testing=False)

    for circuit in config.CIRCUITS:
        matches = schedule[schedule["EventName"].str.contains(circuit.event_name, case=False)]
        assert len(matches) == 1, (
            f"{circuit.key!r}: event_name {circuit.event_name!r} matched "
            f"{len(matches)} events, expected exactly 1"
        )
        actual = int(matches.iloc[0]["RoundNumber"])
        assert actual == circuit.round_2024, (
            f"{circuit.key!r}: config says round {circuit.round_2024}, schedule says {actual}"
        )
