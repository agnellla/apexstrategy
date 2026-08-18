"""``uv run python -m apexstrategy.strategy`` -- the strategy engine CLI.

A package entry point rather than ``python -m apexstrategy.strategy.engine``: the package
``__init__`` imports ``engine``, so running the module directly loads it twice and runpy
warns about it.
"""

import sys

from .engine import main

sys.exit(main())
