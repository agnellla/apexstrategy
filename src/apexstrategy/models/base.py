"""Common model interface.

``predict_dist`` is mandatory, not optional. The simulator only ever calls it, so
uncertainty is a first-class output from the first model rather than something bolted
on at Phase 8 -- retrofitting it later would mean rewriting the simulator.

Phase 5 models sample the *conditional* residual distribution: the spread of a single
lap around its prediction. Phase 8 adds the hierarchical structure on top (persistent
per-driver and per-stint effects), which is what stops variance collapsing over a stint.
Sampling here is deliberately i.i.d. and is not sufficient on its own.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_dist(self, X: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return an ``(n, len(X))`` array of sampled outcomes."""
        ...


class ResidualSampler:
    """Mixin providing ``predict_dist`` by bootstrapping stored training residuals.

    Non-parametric on purpose: lap-time residuals are right-skewed (a lap can go badly
    wrong, but there is a floor on how fast it can go), and a Gaussian would understate
    the slow tail that matters for strategy risk.
    """

    residuals_: np.ndarray

    def _store_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        self.residuals_ = np.asarray(y_true) - np.asarray(y_pred)

    def predict_dist(self, X: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        if not hasattr(self, "residuals_"):
            raise RuntimeError("model must be fitted before predict_dist")
        mu = self.predict(X)
        draws = rng.choice(self.residuals_, size=(n, len(mu)), replace=True)
        return mu[None, :] + draws

    def residual_sd(self) -> float:
        return float(np.std(self.residuals_))
