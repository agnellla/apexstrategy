"""FastF1 access. The only module in the project that imports ``fastf1``.

Adapted from the loader in the earlier ``pitwindow`` project, whose separation of
"dumb fetch" from "cleaning and feature work" was sound and is kept: if the FastF1 API
changes, this file should be the only casualty.

Deliberately does no cleaning, renaming, or derivation -- that is ``normalize.py``'s job.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .. import config

log = logging.getLogger(__name__)


#: Venue names FastF1 reports that the project already had a shorter key for. The seven
#: hand-encoded circuits keep their original keys so that stored data, tests and the whole
#: build log stay readable; every other venue is keyed by its slugified location.
LEGACY_KEYS: dict[str, str] = {
    "Sakhir": "bahrain",
    "Suzuka": "suzuka",
    "Monaco": "monaco",
    "Barcelona": "barcelona",
    "Spielberg": "austria",
    "Silverstone": "britain",
    "Budapest": "hungary",
    # Same circuit, different Location string between seasons. Keying on the venue is more
    # stable than keying on the event name, but "more stable" is not "stable", and an
    # unnoticed drift here silently splits one track's history into two circuits that each
    # look new to the model.
    "Miami": "miami",
    "Miami Gardens": "miami",
}


def circuit_key(location: str) -> str:
    """A stable key for a venue.

    Keyed on ``Location`` rather than ``EventName`` because the venue is the thing the
    physics belongs to and the event name is marketing: Silverstone has been the British
    Grand Prix throughout, but Imola is the "Emilia Romagna" Grand Prix only some years,
    and a sponsor can rename a race without moving it.
    """
    if location in LEGACY_KEYS:
        return LEGACY_KEYS[location]
    slug = unicodedata.normalize("NFKD", location).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")


@dataclass(frozen=True)
class RaceRef:
    """One race on one calendar, resolved before anything is downloaded."""

    key: str
    season: int
    round: int
    event_name: str
    location: str


def discover_races(season: int) -> list[RaceRef]:
    """Every race on a season's calendar, keyed by venue.

    Replaces hand-encoding a circuit registry, which was right for seven races and does not
    scale to a decade of them. The encode-then-verify check it replaces still exists for the
    original seven (``load_session``); here the schedule *is* the source, so there is
    nothing to verify it against.
    """
    import fastf1

    config.setup_cache()
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    return [
        RaceRef(
            key=circuit_key(str(row.Location)),
            season=season,
            round=int(row.RoundNumber),
            event_name=str(row.EventName),
            location=str(row.Location),
        )
        for row in schedule.itertuples()
        if int(row.RoundNumber) > 0
    ]


def load_race(ref: RaceRef) -> RawSession:
    """Load a discovered race by round number.

    By round rather than by name because the name is what changes between seasons, and a
    round number resolved from that season's own schedule cannot drift.
    """
    import fastf1

    config.setup_cache()
    session = fastf1.get_session(ref.season, ref.round, config.SESSION)
    session.load(laps=True, telemetry=config.LOAD_TELEMETRY, weather=True, messages=True)
    return RawSession(
        race_id=race_id(ref.season, ref.key),
        season=ref.season,
        round=ref.round,
        circuit=ref.key,
        session=session,
        total_laps=int(session.total_laps),
    )


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

    # Encode-then-verify: the round number in config is a belief about the calendar and
    # this is what checks it against the real one. It caught a wrong event matcher on day
    # one ('Austria' resolving to the Australian Grand Prix), so it stays.
    #
    # It is scoped to the season it was encoded for. Calendars are reordered between years
    # -- Bahrain moved from round 1 to round 4 and Japan from 4 to 3 for 2025 -- so
    # applying a 2024 belief to a 2025 session rejects a correctly matched race. The
    # *name* match is what identifies the circuit in any season; the round number is a
    # cross-check on one particular calendar.
    resolved_round = int(session.event["RoundNumber"])
    if season == config.SEASON and resolved_round != circuit.round_2024:
        raise ValueError(
            f"{circuit.key}: config says round {circuit.round_2024} in {config.SEASON}, "
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
