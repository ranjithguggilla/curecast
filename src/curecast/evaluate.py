"""Regression metrics and the plain-language reason codes.

Nothing exotic: R-squared, RMSE, and MAE, all measured the same way whether I
am reporting cross-validation scores or the final held-out numbers, so the
figures in the README are directly comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import RAW_FEATURES


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """R-squared, RMSE and MAE as a small dictionary."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def explain_mix(mix: dict[str, float]) -> list[str]:
    """Turn a mix into short, domain-honest notes about its likely strength.

    These are rules of thumb every concrete technologist would recognise, not
    model internals. They give a person a reason to trust or question a number
    before they pour anything.
    """
    notes = []
    cement = mix.get("cement", 0.0)
    binder = cement + mix.get("slag", 0.0) + mix.get("ash", 0.0)
    wc = mix.get("water", 0.0) / (cement + 1e-6)
    wb = mix.get("water", 0.0) / (binder + 1e-6)
    scm = (mix.get("slag", 0.0) + mix.get("ash", 0.0)) / (binder + 1e-6)
    age = mix.get("age", 0.0)

    if wb <= 0.40:
        notes.append(f"Low water-to-binder ratio ({wb:.2f}) works in favour of strength.")
    elif wb >= 0.55:
        notes.append(f"High water-to-binder ratio ({wb:.2f}) tends to cut strength.")
    else:
        notes.append(f"Moderate water-to-binder ratio ({wb:.2f}).")

    if age < 7:
        notes.append(f"Only {age:.0f} days of curing, so strength is still developing.")
    elif age >= 56:
        notes.append(f"Long curing ({age:.0f} days) lets slower reactions finish.")

    if scm >= 0.4:
        notes.append(
            f"{scm * 100:.0f}% of the binder is slag or fly ash, which trades early "
            "strength for lower carbon and better late strength."
        )
    if wc >= 0.6:
        notes.append(f"Water-to-cement ratio of {wc:.2f} is on the weak side.")
    return notes


def sanity_feature_order(columns) -> bool:
    """Guard used by tests to catch silent column reordering."""
    return list(columns)[: len(RAW_FEATURES)] == list(RAW_FEATURES)
