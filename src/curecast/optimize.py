"""Search for the lowest-carbon mix that still hits a strength target.

This is the part that turns a strength predictor into something a batching
plant can act on. Given a target strength and a curing age, I search over how
much cement to keep, how much of the binder to swap for slag and fly ash, and
the water-to-binder ratio, then keep the cheapest-carbon design whose predicted
strength clears the target with a safety margin. The aggregates and
superplasticiser are held at sensible fixed values so the design space stays
interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import co2_per_m3, cost_per_m3
from .scoring import load_model, predict_strength

# Fixed choices that keep the search focused on the binder system, which is
# where nearly all of the carbon lives anyway.
_COARSE = 968.0
_FINE = 773.0
_SUPERPLASTIC = 6.0

# Physical envelope observed in the training data. The optimiser refuses to
# recommend anything outside this box, because the model has never seen such a
# mix and its prediction there would be a guess dressed up as a number.
_ENVELOPE = {
    "water": (122.0, 247.0),
    "binder": (200.0, 600.0),
    "cement": (102.0, 540.0),
}


def _within_envelope(mix: dict[str, float]) -> bool:
    binder = mix["cement"] + mix["slag"] + mix["ash"]
    checks = {
        "water": mix["water"],
        "binder": binder,
        "cement": mix["cement"],
    }
    return all(lo <= checks[k] <= hi for k, (lo, hi) in _ENVELOPE.items())


@dataclass
class MixCandidate:
    mix: dict[str, float]
    predicted_strength: float
    co2: float
    cost: float
    meets_target: bool
    margin: float = field(default=0.0)


def _build_mix(binder: float, scm_fraction: float, wb_ratio: float,
               slag_share: float, age: float) -> dict[str, float]:
    scm = binder * scm_fraction
    cement = binder - scm
    slag = scm * slag_share
    ash = scm * (1.0 - slag_share)
    water = binder * wb_ratio
    return {
        "cement": cement,
        "slag": slag,
        "ash": ash,
        "water": water,
        "superplastic": _SUPERPLASTIC,
        "coarseagg": _COARSE,
        "fineagg": _FINE,
        "age": age,
    }


def optimise_mix(
    target_strength: float,
    age: float = 28.0,
    safety_margin: float = 1.05,
    model=None,
    binder_grid=None,
    scm_grid=None,
    wb_grid=None,
    slag_share: float = 0.6,
) -> MixCandidate:
    """Return the lowest-carbon feasible mix for the target, or the closest miss.

    ``safety_margin`` inflates the target so the design keeps head-room above
    the number the model predicts. If nothing in the grid clears the bar, the
    candidate with the highest predicted strength is returned with
    ``meets_target`` set to False, so the caller always gets an answer.
    """
    model = model or load_model()
    required = target_strength * safety_margin

    binder_grid = binder_grid if binder_grid is not None else np.arange(260, 561, 20.0)
    scm_grid = scm_grid if scm_grid is not None else np.arange(0.0, 0.61, 0.1)
    wb_grid = wb_grid if wb_grid is not None else np.arange(0.32, 0.61, 0.04)

    best_feasible: MixCandidate | None = None
    best_effort: MixCandidate | None = None

    for binder in binder_grid:
        for scm in scm_grid:
            for wb in wb_grid:
                mix = _build_mix(binder, scm, wb, slag_share, age)
                if not _within_envelope(mix):
                    continue
                strength = predict_strength(mix, model=model)
                cand = MixCandidate(
                    mix=mix,
                    predicted_strength=strength,
                    co2=co2_per_m3(mix),
                    cost=cost_per_m3(mix),
                    meets_target=strength >= required,
                    margin=strength - target_strength,
                )
                if cand.meets_target:
                    if best_feasible is None or cand.co2 < best_feasible.co2:
                        best_feasible = cand
                if best_effort is None or cand.predicted_strength > best_effort.predicted_strength:
                    best_effort = cand

    return best_feasible if best_feasible is not None else best_effort


def savings_vs_baseline(candidate: MixCandidate, baseline: dict[str, float]) -> dict:
    """Carbon and cost saved by a candidate against a reference high-clinker mix."""
    base_co2 = co2_per_m3(baseline)
    base_cost = cost_per_m3(baseline)
    return {
        "co2_saved_kg_per_m3": round(base_co2 - candidate.co2, 1),
        "co2_saved_pct": round(100 * (base_co2 - candidate.co2) / base_co2, 1),
        "cost_saved_usd_per_m3": round(base_cost - candidate.cost, 2),
        "cost_saved_pct": round(100 * (base_cost - candidate.cost) / base_cost, 1),
    }
