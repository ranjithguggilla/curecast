"""End-to-end training run: compare, pick, evaluate, explain, and save.

Running this module rebuilds everything the rest of the project depends on: the
serialised pipeline, the metrics file the README quotes, and every chart in the
assets folder. It prints a short progress log so a run is easy to follow.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from .config import (
    ASSETS_DIR,
    CV_FOLDS,
    METRICS_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    RAW_FEATURES,
    TEST_SIZE,
    co2_per_m3,
    cost_per_m3,
)
from .data import load_dataset, split_xy
from .eda import run_eda
from .evaluate import regression_metrics
from .features import ENGINEERED_FEATURES
from .model import build_model_zoo
from .optimize import optimise_mix, savings_vs_baseline
from .plots import (
    feature_importance_plot,
    model_comparison,
    predicted_vs_actual,
    residuals,
)


def _cross_validate(zoo, X, y):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in zoo.items():
        preds = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)
        results[name] = regression_metrics(y, preds)
    return results


def _baseline_mix() -> dict:
    """A conventional high-clinker 28-day mix used as the carbon reference."""
    return {
        "cement": 480.0,
        "slag": 0.0,
        "ash": 0.0,
        "water": 192.0,
        "superplastic": 4.0,
        "coarseagg": 968.0,
        "fineagg": 773.0,
        "age": 28.0,
    }


def main() -> dict:
    started = time.time()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading and validating data")
    df = load_dataset()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  {len(X_train)} train rows, {len(X_test)} test rows")

    print("running exploratory analysis")
    eda_summary = run_eda()

    print(f"cross-validating {CV_FOLDS} folds across the model zoo")
    zoo = build_model_zoo()
    cv_results = _cross_validate(zoo, X_train, y_train)
    for name, m in sorted(cv_results.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"  {name:18s} cv_rmse={m['rmse']:.3f}  cv_r2={m['r2']:.3f}")

    best_name = min(cv_results, key=lambda n: cv_results[n]["rmse"])
    print(f"winner on cv rmse: {best_name}")

    print("refitting winner on the full training split")
    best = zoo[best_name]
    best.fit(X_train, y_train)
    test_pred = best.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)
    print(f"  held-out: r2={test_metrics['r2']:.3f} rmse={test_metrics['rmse']:.3f} "
          f"mae={test_metrics['mae']:.3f}")

    # Naive baseline: always predict the training mean.
    naive_pred = np.full_like(y_test, fill_value=float(y_train.mean()), dtype=float)
    naive_metrics = regression_metrics(y_test, naive_pred)

    print("measuring permutation importance on the held-out split")
    perm = permutation_importance(
        best, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    # permutation_importance runs on the raw pipeline inputs, one score per column
    importances = dict(zip(RAW_FEATURES, [float(v) for v in perm.importances_mean], strict=True))
    imp_names = list(importances.keys())
    imp_values = list(importances.values())

    print("charting")
    model_comparison(list(cv_results.keys()),
                     [cv_results[n]["rmse"] for n in cv_results])
    predicted_vs_actual(y_test, test_pred)
    residuals(y_test, test_pred)
    feature_importance_plot(imp_names, imp_values)

    print("running the mix optimiser demo")
    baseline = _baseline_mix()
    demo = optimise_mix(target_strength=40.0, age=28.0, model=best)
    demo_savings = savings_vs_baseline(demo, baseline)

    print("saving model and metrics")
    joblib.dump(best, MODEL_FILE)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "features_raw": len(RAW_FEATURES),
            "features_engineered": len(ENGINEERED_FEATURES),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "eda": eda_summary,
        "cv": cv_results,
        "winner": best_name,
        "test": test_metrics,
        "naive_baseline": naive_metrics,
        "permutation_importance": importances,
        "optimizer_demo": {
            "target_strength_mpa": 40.0,
            "age_days": 28,
            "baseline_mix": baseline,
            "baseline_co2_kg_per_m3": round(co2_per_m3(baseline), 1),
            "baseline_cost_usd_per_m3": round(cost_per_m3(baseline), 2),
            "optimised_mix": {k: round(v, 1) for k, v in demo.mix.items()},
            "optimised_predicted_strength_mpa": round(demo.predicted_strength, 2),
            "optimised_co2_kg_per_m3": round(demo.co2, 1),
            "optimised_cost_usd_per_m3": round(demo.cost, 2),
            "meets_target": bool(demo.meets_target),
            "savings": demo_savings,
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"done in {metrics['runtime_seconds']}s")
    return metrics


if __name__ == "__main__":
    main()
