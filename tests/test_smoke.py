"""Phase 0 exit criterion: the package imports and the environment is wired correctly."""

from __future__ import annotations

import pytest

from apexstrategy import config


def test_package_imports() -> None:
    import apexstrategy

    assert apexstrategy.__version__


def test_core_dependencies_import() -> None:
    """A fast, explicit failure here beats a confusing one deep in the pipeline."""
    import lightgbm  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import pydantic  # noqa: F401
    import sklearn  # noqa: F401
    import statsmodels  # noqa: F401


def test_python_is_311() -> None:
    """D6: the scientific stack is pinned to 3.11, not the system 3.13."""
    import sys

    assert sys.version_info[:2] == (3, 11)


def test_seeded_cache_is_present() -> None:
    """Four races were seeded from a prior project's cache in Phase 0."""
    season_dir = config.CACHE_DIR / str(config.SEASON)
    if not season_dir.is_dir():
        pytest.skip("cache not seeded in this environment")
    assert len(list(season_dir.iterdir())) >= 4


def test_cache_contains_no_telemetry() -> None:
    """D7 guard: telemetry re-entering the cache would inflate it ~90x."""
    if not config.CACHE_DIR.is_dir():
        pytest.skip("no cache present")
    stray = [
        p.name
        for p in config.CACHE_DIR.rglob("*.ff1pkl")
        if p.name in {"car_data.ff1pkl", "position_data.ff1pkl"}
    ]
    assert not stray, f"telemetry found in cache, violating D7: {set(stray)}"
