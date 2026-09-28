"""The one scoring entry point the UI and tests both call.

Keeping this pure and free of any Gradio import is deliberate: the interface is
a thin shell over ``score_mix`` and ``score_mix`` is what the test-suite pins
down, so a green test genuinely means the thing users click is correct.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from .config import MODEL_FILE, RAW_FEATURES, co2_per_m3, cost_per_m3
from .evaluate import explain_mix


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_FILE)):
    """Load the trained pipeline once and cache it for repeat calls."""
    return joblib.load(path)


def _as_frame(mix: dict[str, float]) -> pd.DataFrame:
    missing = [c for c in RAW_FEATURES if c not in mix]
    if missing:
        raise ValueError(f"mix is missing inputs: {missing}")
    row = {c: float(mix[c]) for c in RAW_FEATURES}
    return pd.DataFrame([row], columns=RAW_FEATURES)


def predict_strength(mix: dict[str, float], model=None) -> float:
    """Predicted compressive strength in MPa for a single mix."""
    model = model or load_model()
    frame = _as_frame(mix)
    return float(model.predict(frame)[0])


def score_mix(mix: dict[str, float], model=None) -> dict:
    """Full scorecard for a mix: strength, carbon, cost, and reasons.

    This is intentionally the only function the app needs. Everything the user
    sees on screen comes back in this dictionary.
    """
    strength = predict_strength(mix, model=model)
    carbon = co2_per_m3(mix)
    cost = cost_per_m3(mix)
    intensity = carbon / strength if strength > 0 else float("nan")
    return {
        "predicted_strength_mpa": round(strength, 2),
        "co2_kg_per_m3": round(carbon, 1),
        "cost_usd_per_m3": round(cost, 2),
        "carbon_intensity_kg_per_mpa": round(intensity, 3),
        "reasons": explain_mix(mix),
    }
