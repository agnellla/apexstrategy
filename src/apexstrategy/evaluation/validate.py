"""Is the pace model underfitting, overfitting, or fitting? Diagnosed, not asserted.

    uv run python -m apexstrategy.evaluation.validate
    uv run python -m apexstrategy.evaluation.validate --skip-gbm

Every model choice in this project so far was made on five races. D11 rejected a boosted
model against the linear baseline on that evidence, which was the right call *at that
sample size* and says nothing about the right call at seventy. A linear model that wins on
five circuits because a flexible one cannot see enough structure to generalise is a model
that will underfit the moment the data arrives.

This module answers three separate questions that are easy to run together and wrong to
confuse:

1. **Is there enough data?** A learning curve over the number of *races* in the training
   set. Train and validation error converging high means underfit; a persistent gap means
   overfit; validation still falling at the right-hand edge means more races would help.
2. **Which model?** Linear against boosted, on identical grouped folds, at the current
   sample size. Re-run whenever the data grows, because the answer is a function of it.
3. **Generalise to what?** A new circuit, a new season, or both. These are different
   difficulties and reporting one number for all three hides which is failing.

## Folds are groups of laps, never laps

Laps within a race are massively correlated: a car has a pace today, a track has a grip
level today, and a random split puts lap 31 in train and lap 32 in test and reports an
accuracy that is really a measure of how well the model memorised the afternoon. Every
split here is by ``race_id`` through ``GroupKFold``, which is the same rule the rest of the
project applies as "no random train/test splits".

## Why the metric is a delta and not a lap time

The target is ``pace_delta_s``, a lap time minus that driver's race median. Raw lap times
are not comparable across circuits that span 71 to 97 seconds, so an R² on them would be
dominated by which tracks are in the test fold. The strategy layer needs differences
anyway.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .. import config
from ..data import store
from ..models.pace import (
    TARGET,
    GBMPaceModel,
    LinearPaceBaseline,
    QuadraticPaceBaseline,
    add_reference_pace,
    feature_frame,
)

log = logging.getLogger(__name__)

#: Fractions of the available races used for each point on the learning curve.
LEARNING_CURVE_FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)

#: Repeats per learning-curve point. The variance between which races land in a small
#: training set is large, so a single draw at 10% is noise rather than a measurement.
LEARNING_CURVE_REPEATS = 5

MODELS = {
    "linear": LinearPaceBaseline,
    "quadratic": QuadraticPaceBaseline,
    "gbm": GBMPaceModel,
}

#: Capacity settings swept for the boosted model. The shipped defaults were chosen for
#: "under 4,000 laps" and are deliberately conservative; comparing a linear model against a
#: boosted one still wearing small-sample handcuffs would rig the comparison in the
#: direction the project already leans. Searched on grouped folds, never on the test data.
GBM_GRID: tuple[dict, ...] = (
    {"num_leaves": 15, "min_child_samples": 40, "n_estimators": 400},
    {"num_leaves": 31, "min_child_samples": 40, "n_estimators": 600},
    {"num_leaves": 63, "min_child_samples": 20, "n_estimators": 800},
    {"num_leaves": 127, "min_child_samples": 10, "n_estimators": 1200, "learning_rate": 0.03},
)


def prepared_laps(
    seasons: tuple[int, ...] | None = None, include_dirty_air: bool = False
) -> pd.DataFrame:
    """Clean, dry, model-ready laps with the target attached.

    Deliberately *not* ``store.training_laps``: that gate exists to stop production code
    touching the held-out season, and this module needs both seasons to measure the cost of
    crossing between them. Season is a column here, and every split below is explicit about
    which side of it a fold lives on.
    """
    laps = add_reference_pace(store.read_table("laps"))
    dry = store.dry_race_ids(laps)
    out = laps[
        laps["is_clean"] & laps["race_id"].isin(dry) & ~laps["circuit"].isin(store.EXCLUDED_VENUES)
    ].dropna(subset=[TARGET, "tyre_age"])
    if seasons:
        out = out[out["season"].isin(seasons)]
    return out.reset_index(drop=True)


def _fit_predict(factory, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = factory()
    model.fit(feature_frame(train), train[TARGET])
    return model.predict(feature_frame(test))


def _mae(y: pd.Series, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y.to_numpy() - pred)))


def learning_curve(
    laps: pd.DataFrame,
    factory,
    fractions: tuple[float, ...] = LEARNING_CURVE_FRACTIONS,
    repeats: int = LEARNING_CURVE_REPEATS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Train and held-out error as a function of how many races the model has seen.

    Races, not laps, is the right x-axis: adding a thousand laps of a race already in the
    training set teaches the model far less than adding one new race, so a curve in laps
    would flatter the model by counting correlated observations as new information.
    """
    rng = np.random.default_rng(seed)
    races = laps["race_id"].unique()
    rows = []

    for fraction in fractions:
        n_train = max(2, int(round(fraction * (len(races) - 2))))
        for rep in range(repeats):
            shuffled = rng.permutation(races)
            train_ids, test_ids = shuffled[:n_train], shuffled[n_train:]
            if len(test_ids) == 0:
                continue
            train = laps[laps["race_id"].isin(train_ids)]
            test = laps[laps["race_id"].isin(test_ids)]

            model = factory()
            model.fit(feature_frame(train), train[TARGET])
            rows.append(
                {
                    "fraction": fraction,
                    "n_races": n_train,
                    "repeat": rep,
                    "train_mae": _mae(train[TARGET], model.predict(feature_frame(train))),
                    "test_mae": _mae(test[TARGET], model.predict(feature_frame(test))),
                }
            )
    return pd.DataFrame(rows)


def grouped_cv(
    laps: pd.DataFrame, factory, n_splits: int = 5, seed: int = config.SEED
) -> pd.DataFrame:
    """K-fold cross-validation with whole races as the groups."""
    groups = laps["race_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(laps, groups=groups)):
        train, test = laps.iloc[tr], laps.iloc[te]
        model = factory()
        model.fit(feature_frame(train), train[TARGET])
        rows.append(
            {
                "fold": fold,
                "n_train_races": train["race_id"].nunique(),
                "n_test_races": test["race_id"].nunique(),
                "train_mae": _mae(train[TARGET], model.predict(feature_frame(train))),
                "test_mae": _mae(test[TARGET], model.predict(feature_frame(test))),
            }
        )
    return pd.DataFrame(rows)


def tune_gbm(
    laps: pd.DataFrame, grid: tuple[dict, ...] = GBM_GRID, n_splits: int = 4
) -> tuple[dict, pd.DataFrame]:
    """Pick boosted-model capacity on grouped folds, then report the whole sweep.

    The sweep is printed rather than hidden because the *shape* of it is the finding: if
    held-out error is flat across a fourfold change in capacity, capacity was never the
    binding constraint and no amount of model class will fix it.
    """
    rows = []
    for params in grid:
        cv = grouped_cv(laps, lambda p=params: GBMPaceModel(**p), n_splits=n_splits)
        rows.append(
            {
                **{k: v for k, v in params.items()},
                "train_mae": cv["train_mae"].mean(),
                "test_mae": cv["test_mae"].mean(),
                "gap": cv["test_mae"].mean() - cv["train_mae"].mean(),
            }
        )
    table = pd.DataFrame(rows).sort_values("test_mae").reset_index(drop=True)
    # Return the grid entry itself, not the row read back out of the DataFrame: pandas
    # widens an int column to float the moment any row lacks a key, and LightGBM refuses
    # n_estimators=400.0.
    best = min(zip(grid, rows, strict=True), key=lambda pair: pair[1]["test_mae"])[0]
    return best, table


def generalisation_matrix(laps: pd.DataFrame, factory) -> pd.DataFrame:
    """Held-out circuit, held-out season, and held out on both at once.

    Three different questions that a single cross-validation number silently averages
    together. A model can transfer perfectly to a new circuit it has seen in another year
    and fail completely on a circuit it has never seen in a year it has never seen, and
    only the third row tells you that.
    """
    train_seasons = config.TRAINING_SEASONS
    holdout_season = config.HOLDOUT_SEASONS[0]
    train_all = laps[laps["season"].isin(train_seasons)]
    holdout_all = laps[laps["season"] == holdout_season]
    rows = []

    def evaluate(case: str, circuit: str, base: pd.DataFrame, test: pd.DataFrame) -> None:
        if base.empty or len(test) < 50:
            return
        model = factory()
        model.fit(feature_frame(base), base[TARGET])
        rows.append(
            {
                "case": case,
                "circuit": circuit,
                "n_train_races": base["race_id"].nunique(),
                "n_test": len(test),
                "mae": _mae(test[TARGET], model.predict(feature_frame(test))),
            }
        )

    for circuit in sorted(train_all["circuit"].unique()):
        # A circuit the model has never seen, in a season it has.
        evaluate(
            "new circuit, same season",
            circuit,
            train_all[train_all["circuit"] != circuit],
            train_all[train_all["circuit"] == circuit],
        )
        # A circuit it knows well, in a season it has never seen. The training set keeps
        # this circuit -- excluding it here was a bug that made this case identical to the
        # one below and reported the same number twice as if it were two findings.
        evaluate(
            "same circuit, new season",
            circuit,
            train_all,
            holdout_all[holdout_all["circuit"] == circuit],
        )

    for circuit in sorted(holdout_all["circuit"].unique()):
        # Both at once: the hardest case, and the one a team faces at a new venue.
        evaluate(
            "new circuit, new season",
            circuit,
            train_all[train_all["circuit"] != circuit],
            holdout_all[holdout_all["circuit"] == circuit],
        )

    return pd.DataFrame(rows)


def residual_diagnostics(laps: pd.DataFrame, factory, n_splits: int = 5) -> dict[str, pd.DataFrame]:
    """Where the model is systematically wrong, out of fold.

    Mean error rather than absolute error, because the question is bias: a model that is
    0.4 s optimistic on old hard tyres will recommend staying out, every time, at exactly
    the moment the decision is hardest.
    """
    groups = laps["race_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    held = []
    for tr, te in splitter.split(laps, groups=groups):
        train, test = laps.iloc[tr], laps.iloc[te]
        held.append(test.assign(resid=test[TARGET].to_numpy() - _fit_predict(factory, train, test)))
    out = pd.concat(held, ignore_index=True)

    out["age_bucket"] = pd.cut(
        out["tyre_age"], [0, 5, 10, 15, 20, 25, 30, 40, 100], right=True
    ).astype(str)
    return {
        "by_compound": out.groupby("compound", observed=True)["resid"]
        .agg(bias="mean", mae=lambda s: s.abs().mean(), n="size")
        .reset_index(),
        "by_tyre_age": out.groupby("age_bucket", observed=True)["resid"]
        .agg(bias="mean", mae=lambda s: s.abs().mean(), n="size")
        .reset_index(),
        "by_circuit": out.groupby("circuit", observed=True)["resid"]
        .agg(bias="mean", mae=lambda s: s.abs().mean(), n="size")
        .sort_values("bias")
        .reset_index(),
    }


def _verdict(curve: pd.DataFrame) -> str:
    """Read the learning curve. Stated as a sentence, because a number invites a squint."""
    tail = curve[curve["fraction"] == curve["fraction"].max()]
    gap = float(tail["test_mae"].mean() - tail["train_mae"].mean())
    train_mae = float(tail["train_mae"].mean())

    by_size = curve.groupby("n_races")["test_mae"].mean().sort_index()
    still_falling = len(by_size) > 2 and (by_size.iloc[-3] - by_size.iloc[-1]) > 0.005

    parts = []
    parts.append(
        f"train {train_mae:.3f}s, held-out {tail['test_mae'].mean():.3f}s, gap {gap:+.3f}s"
    )
    if gap > 0.10:
        parts.append("OVERFIT: the model does markedly better on races it has seen")
    elif gap < 0.03:
        parts.append("no overfitting: held-out error is essentially the training error")
    else:
        parts.append("mild gap, within what fold variation explains")
    parts.append(
        "held-out error is still falling with more races, so more data would help"
        if still_falling
        else "held-out error has flattened; more races of the same kind will not help"
    )
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gbm", action="store_true", help="linear baseline only")
    parser.add_argument("--repeats", type=int, default=LEARNING_CURVE_REPEATS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    laps = prepared_laps()
    races = laps.groupby("season")["race_id"].nunique()

    print("=== data ===")
    print(
        f"{len(laps):,} clean dry laps, {laps['race_id'].nunique()} races, "
        f"{laps['circuit'].nunique()} circuits"
    )
    print(races.to_string())

    train_only = laps[laps["season"].isin(config.TRAINING_SEASONS)]
    factories: dict[str, object] = {
        "linear": LinearPaceBaseline,
        "quadratic": QuadraticPaceBaseline,
    }

    if not args.skip_gbm:
        print("\n=== boosted-model capacity sweep, grouped folds ===")
        best, sweep = tune_gbm(laps)
        print(sweep.round(3).to_string(index=False))
        print(f"  selected: {best}")
        factories["gbm"] = lambda: GBMPaceModel(**{k: v for k, v in best.items()})

    names = list(factories)
    MODELS.update(factories)  # so the sections below can look factories up by name

    print("\n=== grouped cross-validation, folds are whole races ===")
    for name in names:
        cv = grouped_cv(laps, MODELS[name])
        print(
            f"  {name:7s} train {cv['train_mae'].mean():.3f}s   "
            f"held-out {cv['test_mae'].mean():.3f}s  (sd {cv['test_mae'].std():.3f})"
        )

    print("\n=== learning curve: underfit or overfit? ===")
    for name in names:
        curve = learning_curve(laps, MODELS[name], repeats=args.repeats)
        print(f"\n  {name}")
        table = curve.groupby("n_races")[["train_mae", "test_mae"]].mean().round(3)
        table["gap"] = (table["test_mae"] - table["train_mae"]).round(3)
        print(table.to_string().replace("\n", "\n  "))
        print(f"  -> {_verdict(curve)}")

    print("\n=== generalisation: to what? ===")
    for name in names:
        matrix = generalisation_matrix(laps, MODELS[name])
        if matrix.empty:
            continue
        summary = matrix.groupby("case")["mae"].agg(["mean", "std", "count"]).round(3)
        print(f"\n  {name}")
        print(summary.to_string().replace("\n", "\n  "))

    print("\n=== residual bias, out of fold (linear) ===")
    diag = residual_diagnostics(train_only, LinearPaceBaseline)
    for label, frame in diag.items():
        print(f"\n  {label}")
        print(frame.round(3).head(10).to_string(index=False).replace("\n", "\n  "))

    return 0


if __name__ == "__main__":
    sys.exit(main())
