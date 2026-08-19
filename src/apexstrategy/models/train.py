"""Phase 5 entry point: evaluate candidate pace models, fit the winner, persist it.

    uv run python -m apexstrategy.models.train

Prints the LORO comparison that justifies the choice, then fits the selected model on
all trainable clean laps and writes it to ``models/pace_model.joblib``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import joblib
import pandas as pd

from .. import config
from ..data import store
from ..evaluation.loro import (
    degradation_recovery,
    interval_coverage,
    run_loro,
    summarise,
)
from .pace import TARGET, GBMPaceModel, LinearPaceBaseline, add_reference_pace, feature_frame

log = logging.getLogger(__name__)

MODEL_PATH = config.MODELS_DIR / "pace_model.joblib"

#: Chosen on LORO evidence, not on principle -- see the Phase 5 build log. The boosted
#: model lost on every metric that matters, including degradation recovery.
SELECTED = LinearPaceBaseline


def training_frame() -> pd.DataFrame:
    """Clean laps from trainable circuits, with the reference pace and target attached."""
    laps = store.fittable_laps()
    allowed = {c.key for c in config.TRAINING_CIRCUITS}
    laps = laps[laps["circuit"].isin(allowed)]
    laps = add_reference_pace(laps)
    return laps[laps["is_clean"]].dropna(subset=[TARGET, "tyre_age"]).reset_index(drop=True)


def evaluate(laps: pd.DataFrame) -> pd.DataFrame:
    """LORO comparison of the candidate models."""
    return pd.DataFrame(
        [
            summarise(run_loro(laps, LinearPaceBaseline), "linear baseline"),
            summarise(run_loro(laps, GBMPaceModel), "lightgbm"),
        ]
    )


def fit_and_save(laps: pd.DataFrame):
    model = SELECTED().fit(feature_frame(laps), laps[TARGET])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("wrote %s", MODEL_PATH)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-eval", action="store_true", help="fit and save without LORO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    laps = training_frame()
    print(f"training laps: {len(laps)} across {laps['circuit'].nunique()} circuits")

    if not args.skip_eval:
        print("\n=== LORO model comparison ===")
        print(evaluate(laps).round(4).to_string(index=False))

        print("\n=== degradation recovery (selected model, s/lap) ===")
        deg = degradation_recovery(laps, SELECTED)
        print(deg.round(4).to_string(index=False))
        print(f"mean |error|: {deg['error'].abs().mean():.4f} s/lap")

        print("\n=== predictive interval coverage ===")
        print(interval_coverage(laps, SELECTED).round(3).to_string(index=False))
        print("i.i.d. residual sampling only; hierarchical structure arrives in Phase 8.")

    model = fit_and_save(laps)
    print(f"\nfitted {SELECTED.__name__}, residual sd = {model.residual_sd():.4f}s")
    print(f"saved -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
