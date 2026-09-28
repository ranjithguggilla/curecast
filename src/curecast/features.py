"""Row-wise feature engineering for concrete mixes.

Every feature here is a ratio the materials literature already leans on, and
each one is computed from a single row, so there is no way for information to
leak across the train/test boundary. I still wrap them in a scikit-learn
transformer so the exact same maths runs at training time and inside the
served model, with no chance of the two drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import RAW_FEATURES

# Names of the columns this transformer appends, in order.
ENGINEERED_FEATURES = [
    "binder",
    "water_cement_ratio",
    "water_binder_ratio",
    "scm_fraction",
    "aggregate_ratio",
    "paste_fraction",
]

_EPS = 1e-6  # keeps the divisions honest when a component is zero


def add_mix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered ratio columns added."""
    out = df.copy()
    binder = out["cement"] + out["slag"] + out["ash"]
    out["binder"] = binder
    out["water_cement_ratio"] = out["water"] / (out["cement"] + _EPS)
    out["water_binder_ratio"] = out["water"] / (binder + _EPS)
    out["scm_fraction"] = (out["slag"] + out["ash"]) / (binder + _EPS)
    out["aggregate_ratio"] = out["coarseagg"] / (out["fineagg"] + _EPS)
    paste = binder + out["water"]
    aggregate = out["coarseagg"] + out["fineagg"]
    out["paste_fraction"] = paste / (paste + aggregate + _EPS)
    return out


class MixFeatureBuilder(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper around :func:`add_mix_features`.

    Stateless: ``fit`` only records the output column order so the estimator
    plays nicely with pipelines and ``get_feature_names_out``.
    """

    def fit(self, X, y=None):
        self.feature_names_out_ = list(RAW_FEATURES) + list(ENGINEERED_FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURES)
        built = add_mix_features(X)
        return built[self.feature_names_out_].to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
