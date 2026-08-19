"""Project-wide paths, constants, and the circuit registry.

Everything here is configuration or belief. Nothing here is derived from data --
derived quantities (fuel coefficients, pit-lane loss, lap counts) are fitted in
Phase 4 and written to ``data/processed/circuits.parquet`` with their standard
errors. Hardcoding them would be a leakage risk (see architecture-proposal.md,
Part 13, risk 5) and a credibility risk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"  # FastF1 raw cache (.ff1pkl)
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"  # normalised Parquet
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED = 20240804  # Hungarian GP 2024, the demo race (D4)

# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

SEASON = 2024
SESSION = "R"

#: The seasons models are fitted on. Everything model-facing filters to these, structurally
#: rather than by convention, so that adding a season to the store cannot silently turn an
#: out-of-sample test into an in-sample one. That is the same leakage discipline the
#: temporal firewall applies within a race, applied across seasons.
TRAINING_SEASONS: tuple[int, ...] = (2023, 2024)

#: Kept as the single "current" season for the demo grid and for race_id defaults.
TRAINING_SEASON = 2024

#: Seasons held out entirely. Never fitted on, only predicted. A model validated by
#: leave-one-race-out inside one season has still only been asked to generalise across
#: circuits; a held-out season asks it to generalise across cars, tyres and regulations,
#: which is the question anyone deploying it would actually ask.
HOLDOUT_SEASONS: tuple[int, ...] = (2025,)

#: Telemetry is not ingested (D7). ``car_data`` and ``position_data`` were 337 MB
#: of a 432 MB four-race cache; lap-level essentials are ~1.2 MB/race. Strategy
#: modelling operates at lap granularity, so telemetry buys nothing for the MVP.
LOAD_TELEMETRY = False


@dataclass(frozen=True)
class Circuit:
    """A circuit in scope.

    Attributes:
        key: Short internal identifier, used in filenames and as a model feature.
        event_name: Substring FastF1 resolves via ``get_event(season, name)``.
        round_2024: Championship round. Encoded as a belief and asserted against
            the real schedule by ``tests/test_config.py`` -- see module docstring
            on the encode-then-verify convention.
        use_for_training: Whether the circuit enters model training/evaluation.
        note: Why a circuit is excluded, where applicable.
    """

    key: str
    event_name: str
    round_2024: int
    use_for_training: bool = True
    note: str = ""


#: Circuit set frozen at seven until the pipeline runs end-to-end (D3).
#: Do not add circuits before Phase 12.
CIRCUITS: tuple[Circuit, ...] = (
    Circuit("bahrain", "Bahrain", 1),
    Circuit("suzuka", "Japan", 4),
    Circuit(
        "monaco",
        "Monaco",
        8,
        use_for_training=False,
        note=(
            "Pathological outlier: near-zero overtaking and atypical pit-lane "
            "geometry. Stored for completeness and for later qualitative use, "
            "but excluded from model fitting and evaluation so it cannot distort "
            "pooled degradation, pit-loss, or pace estimates (D3)."
        ),
    ),
    Circuit("barcelona", "Spanish", 10),
    Circuit("austria", "Austrian", 11),  # 'Austrian' not 'Austria', to exclude 'Australian'
    Circuit(
        "britain",
        "British",
        12,
        use_for_training=False,
        note=(
            "Silverstone 2024 was a mixed-conditions race: 24.3% of its laps were run "
            "on intermediates, against a 5% dry-race threshold. The MVP is dry-only "
            "(Part 14, assumption 8). Contamination is measurable rather than "
            "theoretical -- its pit-loss estimate came out at 35.3s against 20-25s at "
            "every other circuit, because in/out laps ran in damp conditions while the "
            "clean-pace reference came from dry laps. Stored for completeness and for "
            "later wet-weather work; excluded from fitting."
        ),
    ),
    Circuit("hungary", "Hungar", 13),  # 'Hungar' matches Hungarian/Hungaroring
)

CIRCUITS_BY_KEY: dict[str, Circuit] = {c.key: c for c in CIRCUITS}

#: The six circuits models are fitted and evaluated on. Monaco is excluded.
TRAINING_CIRCUITS: tuple[Circuit, ...] = tuple(c for c in CIRCUITS if c.use_for_training)

#: The demo / narrative race for the post-race strategy audit (D4).
DEMO_CIRCUIT_KEY = "hungary"

# --------------------------------------------------------------------------
# FastF1 cache
# --------------------------------------------------------------------------


def setup_cache(cache_dir: Path | None = None) -> Path:
    """Enable the FastF1 on-disk cache, creating it if absent.

    Idempotent and safe to call from any entry point (scripts, tests, Streamlit).

    Args:
        cache_dir: Override the default cache location. Mainly for tests.

    Returns:
        The directory the cache was enabled on.
    """
    import fastf1

    target = cache_dir or Path(os.environ.get("APEXSTRATEGY_CACHE", CACHE_DIR))
    target.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(target))
    return target


def ensure_dirs() -> None:
    """Create the writable data/model directories. Safe to call repeatedly."""
    for d in (CACHE_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
