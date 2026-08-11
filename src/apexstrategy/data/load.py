"""FastF1 access. The only module in the project that imports ``fastf1``.

Adapted from the loader in the earlier ``pitwindow`` project, whose separation of
"dumb fetch" from "cleaning and feature work" was sound and is kept: if the FastF1 API
changes, this file should be the only casualty.

Deliberately does no cleaning, renaming, or derivation -- that is ``normalize.py``'s job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .. import config

log = logging.getLogger(__name__)


def race_id(season: int, circuit_key: str, session: str = "R") -> str:
    """Stable identifier used as the partition key across every table."""
    return f"{season}_{circuit_key}_{session}"


@dataclass(frozen=True)
class RawSession:
    """A loaded FastF1 session plus the identity we resolved it from.

    Carrying identity alongside the session keeps ``normalize`` free of any need to
    re-derive round/circuit from FastF1 objects.
    """

    race_id: str
    season: int
    round: int
    circuit: str
    session: Any  # fastf1.core.Session
    total_laps: int


def load_session(circuit: config.Circuit, season: int = config.SEASON) -> RawSession:
    """Load one race session from cache, fetching only what strategy modelling needs.

    ``telemetry=False`` implements D7: car/position telemetry was 337 MB of a 432 MB
    four-race cache and contributes nothing at lap granularity.
    """
    import fastf1

    config.setup_cache()
    session = fastf1.get_session(season, circuit.event_name, config.SESSION)
    session.load(laps=True, telemetry=config.LOAD_TELEMETRY, weather=True, messages=True)

    resolved_round = int(session.event["RoundNumber"])
    if resolved_round != circuit.round_2024:
        raise ValueError(
            f"{circuit.key}: config says round {circuit.round_2024}, "
            f"FastF1 resolved round {resolved_round} ({session.event['EventName']})"
        )

    return RawSession(
        race_id=race_id(season, circuit.key),
        season=season,
        round=resolved_round,
        circuit=circuit.key,
        session=session,
        total_laps=int(session.total_laps),
    )


def load_all(circuits: tuple[config.Circuit, ...] = config.CIRCUITS) -> list[RawSession]:
    """Load every configured circuit. First fetch per race hits the network."""
    out = []
    for circuit in circuits:
        log.info("loading %s", circuit.key)
        out.append(load_session(circuit))
    return out
