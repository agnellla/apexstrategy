"""Phase 5: the pace/degradation model -- the one genuinely ML component of the MVP.

## What is predicted, and why it is not raw lap time

The target is **lap time relative to the driver's own reference pace in that race**, not
absolute lap time. Two reasons, one practical and one methodological:

- **LORO would be meaningless otherwise.** Leave-one-race-out holds out an entire
  circuit, and median clean lap times across the training circuits span 71s (Austria) to
  97s (Suzuka). A model predicting raw lap time for an unseen circuit would predict
  roughly the average of the others and post an enormous error that says nothing about
  whether it understands tyre degradation.
- **Strategy needs differences, not levels.** "How much time do I lose staying out three
  more laps?" is a delta. The absolute level only sets the race duration scale, and the
  simulator already knows it from observed laps.

So the model learns the *shape* of pace evolution -- how lap time moves with tyre age,
compound, race progress and conditions -- while the level is supplied from observed state.

## Known limitation in the reference pace

``reference_pace`` is the median clean lap for a (race, driver), computed over the whole
race. For training that is a normaliser rather than a feature, but in the simulator the
reference will have to be estimated from laps observed up to the decision lap only. The
model is therefore evaluated under a slightly easier reference than it will face live.
This is flagged rather than hidden and is closed at Phase 6/7, where the reference is
rebuilt from truncated state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ResidualSampler

#: Known at the prediction point and causally prior to the lap being predicted.
#: Deliberately excluded: ``position`` and ``gap_to_leader_s`` -- both are endogenous
#: (a slow car falls back), and feeding them to the simulator would make lap time depend
#: on an ordering that is itself an output of lap times.
NUMERIC_FEATURES = ["tyre_age", "lap_number", "stint_number", "track_temp", "air_temp"]
CATEGORICAL_FEATURES = ["compound", "circuit", "driver"]
BOOL_FEATURES = ["is_fresh_tyre"]

#: Added only by the dirty-air variant, which trades a cleaner training set for ~39%
#: more data plus an explicit estimate of the wake penalty.
TRAFFIC_FEATURE = "entering_gap_s"

TARGET = "pace_delta_s"


def add_reference_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``reference_pace_s`` and the model target ``pace_delta_s``.

    Reference is the median of a driver's *clean* laps in that race, so laps excluded
    for traffic or cautions cannot drag the level around.
    """
    laps = laps.copy()
    clean = laps[laps["is_clean"]] if "is_clean" in laps.columns else laps
    reference = clean.groupby(["race_id", "driver"])["lap_time_s"].median()
    laps["reference_pace_s"] = pd.MultiIndex.from_frame(laps[["race_id", "driver"]]).map(reference)
    laps[TARGET] = laps["lap_time_s"] - laps["reference_pace_s"]
    return laps


def feature_frame(laps: pd.DataFrame, include_traffic: bool = False) -> pd.DataFrame:
    cols = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *BOOL_FEATURES]
    if include_traffic:
        cols = [*cols, TRAFFIC_FEATURE]
    X = laps[cols].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    X["is_fresh_tyre"] = X["is_fresh_tyre"].astype("boolean").fillna(False).astype(bool)
    if include_traffic:
        # No car ahead is not missing data -- it is clear air. Encode it as a large gap
        # rather than a null so the model reads it as "unimpeded".
        X[TRAFFIC_FEATURE] = X[TRAFFIC_FEATURE].fillna(99.0)
    return X


class LinearPaceBaseline(ResidualSampler):
    """Per-compound linear degradation plus a linear race-progress term.

    The baseline the architecture doc requires before any boosted model is accepted:
    ``pace_delta ~ C(compound) + C(compound):tyre_age + lap_number``. Anything more
    complex has to earn its place against this.
    """

    def __init__(self) -> None:
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearPaceBaseline:
        import statsmodels.formula.api as smf

        data = X[["compound", "tyre_age", "lap_number"]].copy()
        data["compound"] = data["compound"].astype(str)
        data["_y"] = np.asarray(y)
        self.model_ = smf.ols(
            "_y ~ C(compound) + C(compound):tyre_age + lap_number", data=data
        ).fit()
        self._store_residuals(y, self.predict(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        data = X[["compound", "tyre_age", "lap_number"]].copy()
        data["compound"] = data["compound"].astype(str)
        return np.asarray(self.model_.predict(data))


class GBMPaceModel(ResidualSampler):
    """LightGBM on the full feature set, with native categorical handling.

    Chosen over XGBoost for native categoricals (circuit, driver, compound are all
    high-cardinality) and better small-sample behaviour -- the training set is under
    4,000 laps, which is small for a boosted model and drives the conservative
    hyperparameters below.
    """

    def __init__(self, **params) -> None:
        self.params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbose": -1,
            **params,
        }
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GBMPaceModel:
        import lightgbm as lgb

        self.feature_names_ = list(X.columns)
        self.model_ = lgb.LGBMRegressor(**self.params).fit(X, y)
        self._store_residuals(y, self.predict(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X[self.feature_names_]))

    def importances(self) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=self.feature_names_).sort_values(
            ascending=False
        )
