from curecast.config import co2_per_m3
from curecast.optimize import MixCandidate, optimise_mix, savings_vs_baseline

_BASELINE = {
    "cement": 480.0, "slag": 0.0, "ash": 0.0, "water": 192.0,
    "superplastic": 4.0, "coarseagg": 968.0, "fineagg": 773.0, "age": 28.0,
}


def test_optimiser_meets_modest_target(trained_model):
    cand = optimise_mix(target_strength=35.0, age=28.0, model=trained_model)
    assert isinstance(cand, MixCandidate)
    assert cand.meets_target
    assert cand.predicted_strength >= 35.0


def test_optimised_mix_stays_in_envelope(trained_model):
    cand = optimise_mix(target_strength=40.0, age=28.0, model=trained_model)
    binder = cand.mix["cement"] + cand.mix["slag"] + cand.mix["ash"]
    assert 122.0 <= cand.mix["water"] <= 247.0
    assert 200.0 <= binder <= 600.0
    assert cand.mix["cement"] >= 102.0


def test_optimiser_beats_baseline_on_carbon(trained_model):
    cand = optimise_mix(target_strength=40.0, age=28.0, model=trained_model)
    assert cand.co2 < co2_per_m3(_BASELINE)
    save = savings_vs_baseline(cand, _BASELINE)
    assert save["co2_saved_kg_per_m3"] > 0
    assert 0 < save["co2_saved_pct"] <= 100


def test_savings_math():
    cheaper = MixCandidate(mix=dict(_BASELINE, cement=240.0), predicted_strength=40.0,
                           co2=200.0, cost=70.0, meets_target=True)
    save = savings_vs_baseline(cheaper, _BASELINE)
    assert save["co2_saved_kg_per_m3"] > 0
