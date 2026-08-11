"""Leave-one-race-out validation.

With one race per circuit, LORO is leave-one-*circuit*-out: the hardest honest test, and
the one that matches the real use case of predicting a race never seen.

Random lap splits are never used. Laps within a race share a track, weather and car set,
so a random split leaks massively and reports error far better than the model deserves.

Beyond aggregate error, ``degradation_recovery`` checks whether the model reproduces the
*degradation slope* per compound. This matters more than MAE: a model can post excellent
error by predicting each driver's mean pace while learning nothing about how tyres decay,
and such a model is useless to the strategy layer even though it looks accurate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.pace import TARGET, feature_frame


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def run_loro(
    laps: pd.DataFrame,
    model_factory,
    include_traffic: bool = False,
    fold_column: str = "circuit",
) -> pd.DataFrame:
    """Fit on all circuits but one, evaluate on the held-out circuit, repeat.

    A fresh model is built per fold via ``model_factory`` so nothing leaks between folds.
    """
    rows = []
    for held_out in sorted(laps[fold_column].unique()):
        train = laps[laps[fold_column] != held_out]
        test = laps[laps[fold_column] == held_out]

        model = model_factory()
        model.fit(feature_frame(train, include_traffic), train[TARGET])
        pred = model.predict(feature_frame(test, include_traffic))

        rows.append(
            {
                "held_out": held_out,
                "n_train": len(train),
                "n_test": len(test),
                **_metrics(test[TARGET].to_numpy(), pred),
            }
        )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, label: str) -> pd.Series:
    """Mean and spread across folds. Between-fold spread is the number that matters --
    it says how much the model's quality depends on which circuits it happened to see."""
    return pd.Series(
        {
            "model": label,
            "mae_mean": results["mae"].mean(),
            "mae_worst": results["mae"].max(),
            "mae_sd": results["mae"].std(),
            "rmse_mean": results["rmse"].mean(),
            "r2_mean": results["r2"].mean(),
        }
    )


def degradation_recovery(
    laps: pd.DataFrame, model_factory, include_traffic: bool = False
) -> pd.DataFrame:
    """Compare actual vs model-implied degradation slope per held-out circuit/compound.

    The slope is the strategically load-bearing quantity: it is what decides whether
    staying out costs more than a pit stop. A model with good MAE but a wrong slope
    would produce confident, wrong strategy calls.
    """
    rows = []
    for held_out in sorted(laps["circuit"].unique()):
        train = laps[laps["circuit"] != held_out]
        test = laps[laps["circuit"] == held_out]

        model = model_factory()
        model.fit(feature_frame(train, include_traffic), train[TARGET])
        test = test.assign(pred=model.predict(feature_frame(test, include_traffic)))

        for compound, g in test.groupby("compound", observed=True):
            if len(g) < 30 or g["tyre_age"].nunique() < 5:
                continue
            actual = np.polyfit(g["tyre_age"], g[TARGET], 1)[0]
            implied = np.polyfit(g["tyre_age"], g["pred"], 1)[0]
            rows.append(
                {
                    "held_out": held_out,
                    "compound": compound,
                    "actual_deg_s_per_lap": actual,
                    "model_deg_s_per_lap": implied,
                    "error": implied - actual,
                    "n": len(g),
                }
            )
    return pd.DataFrame(rows)


def interval_coverage(
    laps: pd.DataFrame,
    model_factory,
    include_traffic: bool = False,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
    n_samples: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Empirical coverage of predictive intervals, held out by circuit.

    Phase 5 samples i.i.d. residuals, so this measures only the conditional spread of a
    single lap. Phase 8 adds the hierarchical structure; coverage of *stint-level*
    aggregates is expected to be too narrow until then, and this function is the
    baseline that improvement is measured against.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for held_out in sorted(laps["circuit"].unique()):
        train = laps[laps["circuit"] != held_out]
        test = laps[laps["circuit"] == held_out]

        model = model_factory()
        model.fit(feature_frame(train, include_traffic), train[TARGET])
        draws = model.predict_dist(feature_frame(test, include_traffic), n_samples, rng)

        truth = test[TARGET].to_numpy()
        row = {"held_out": held_out}
        for level in levels:
            lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
            row[f"cover_{int(level * 100)}"] = float(np.mean((truth >= lo) & (truth <= hi)))
        rows.append(row)
    return pd.DataFrame(rows)
