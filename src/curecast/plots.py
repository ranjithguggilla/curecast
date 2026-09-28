"""Chart helpers used by training and the exploratory pass.

All of these write PNGs into the assets folder so the README can embed them and
so a run leaves a visible paper trail of what the model actually did.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import ASSETS_DIR, FEATURE_LABELS  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True})
_PRIMARY = "#2f6f4e"
_ACCENT = "#c65b2f"


def _save(fig, name: str) -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def target_distribution(y, name="target_distribution.png") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y, bins=32, color=_PRIMARY, edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(y), color=_ACCENT, linestyle="--", label=f"mean {np.mean(y):.1f}")
    ax.set_xlabel(FEATURE_LABELS["strength"])
    ax.set_ylabel("Count")
    ax.set_title("Compressive strength distribution")
    ax.legend()
    return _save(fig, name)


def correlation_heatmap(corr, name="correlation_heatmap.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    ax.set_title("Feature correlations")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, name)


def model_comparison(names, rmses, name="model_comparison.png") -> Path:
    order = np.argsort(rmses)
    names = [names[i] for i in order]
    rmses = [rmses[i] for i in order]
    colors = [_PRIMARY if i == 0 else "#9bbfa9" for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, rmses, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated RMSE (MPa, lower is better)")
    ax.set_title("Model comparison")
    for i, v in enumerate(rmses):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
    return _save(fig, name)


def predicted_vs_actual(y_true, y_pred, name="predicted_vs_actual.png") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.5, color=_PRIMARY, edgecolor="none")
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, color=_ACCENT, linestyle="--", label="perfect")
    ax.set_xlabel("Measured strength (MPa)")
    ax.set_ylabel("Predicted strength (MPa)")
    ax.set_title("Held-out predictions")
    ax.legend()
    return _save(fig, name)


def residuals(y_true, y_pred, name="residuals.png") -> Path:
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, resid, s=16, alpha=0.5, color=_PRIMARY, edgecolor="none")
    ax.axhline(0, color=_ACCENT, linestyle="--")
    ax.set_xlabel("Predicted strength (MPa)")
    ax.set_ylabel("Residual (predicted - measured)")
    ax.set_title("Residuals against prediction")
    return _save(fig, name)


def feature_importance_plot(names, values, name="feature_importance.png") -> Path:
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, values, color=_PRIMARY)
    ax.set_xlabel("Permutation importance (drop in R2)")
    ax.set_title("What the model leans on")
    return _save(fig, name)
