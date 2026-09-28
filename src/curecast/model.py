"""The model zoo and the pipeline factory.

Each candidate is a full pipeline: feature building first, then optional
scaling, then the estimator. Scaling lives inside the pipeline on purpose, so
it is refit on every cross-validation fold and never sees the held-out rows.
Linear and distance-based models get the scaler; tree ensembles do not need it.
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import RANDOM_STATE
from .features import MixFeatureBuilder


def _pipe(estimator, scale: bool) -> Pipeline:
    steps = [("features", MixFeatureBuilder())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_model_zoo() -> dict[str, Pipeline]:
    """Return the candidate pipelines keyed by a short display name.

    The linear model is here as the honest baseline. If it were to match the
    ensembles I would ship it, because a model with eight coefficients is far
    easier to defend to a structural engineer than a forest.
    """
    return {
        "linear": _pipe(LinearRegression(), scale=True),
        "ridge": _pipe(Ridge(alpha=1.0, random_state=RANDOM_STATE), scale=True),
        "elasticnet": _pipe(
            ElasticNet(alpha=0.05, l1_ratio=0.4, random_state=RANDOM_STATE, max_iter=10000),
            scale=True,
        ),
        "knn": _pipe(KNeighborsRegressor(n_neighbors=7, weights="distance"), scale=True),
        "svr": _pipe(SVR(C=50.0, gamma="scale", epsilon=1.0), scale=True),
        "random_forest": _pipe(
            RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
        "gradient_boosting": _pipe(
            GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
        "hist_gbr": _pipe(
            HistGradientBoostingRegressor(
                max_iter=600,
                learning_rate=0.05,
                max_depth=None,
                l2_regularization=0.5,
                random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
    }
