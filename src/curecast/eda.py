"""Exploratory pass over the dataset with a statistics-first bent.

I lean on pingouin here rather than eyeballing a heatmap, because for a
structural material I want the correlation with strength backed by a confidence
interval and a p-value before I trust it. The function returns a tidy summary
and drops two charts for the README.
"""

from __future__ import annotations

import pandas as pd
import pingouin as pg

from .config import RAW_FEATURES, TARGET
from .data import load_dataset
from .features import add_mix_features
from .plots import correlation_heatmap, target_distribution


def strength_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of every input with strength, with p-values, via pingouin."""
    rows = []
    for col in RAW_FEATURES:
        res = pg.corr(df[col], df[TARGET], method="pearson")
        rows.append({
            "feature": col,
            "r": round(float(res["r"].iloc[0]), 3),
            "p_value": float(res["p_val"].iloc[0]),
            "n": int(res["n"].iloc[0]),
        })
    out = pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
    return out.reset_index(drop=True)


def run_eda() -> dict:
    """Produce the EDA summary and charts; return a small dictionary of facts."""
    df = load_dataset()
    enriched = add_mix_features(df)
    corr = enriched[RAW_FEATURES + ["water_binder_ratio", "scm_fraction", TARGET]].corr()

    target_distribution(df[TARGET])
    correlation_heatmap(corr)
    corr_table = strength_correlations(df)

    skew = float(df[TARGET].skew())
    return {
        "n_rows": int(len(df)),
        "n_features": len(RAW_FEATURES),
        "target_mean": round(float(df[TARGET].mean()), 2),
        "target_std": round(float(df[TARGET].std()), 2),
        "target_min": round(float(df[TARGET].min()), 2),
        "target_max": round(float(df[TARGET].max()), 2),
        "target_skew": round(skew, 3),
        "top_correlations": corr_table.to_dict("records"),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_eda(), indent=2))
