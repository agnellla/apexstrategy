"""``uv run python -m apexstrategy.showcase`` -- freeze the decision grid to JSON."""

import sys

from .export import main

sys.exit(main())
