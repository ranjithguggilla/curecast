import pytest

from curecast.scoring import predict_strength, score_mix


def test_predict_strength_reasonable(trained_model, sample_mix):
    s = predict_strength(sample_mix, model=trained_model)
    assert 5.0 < s < 90.0


def test_score_mix_returns_full_card(trained_model, sample_mix):
    card = score_mix(sample_mix, model=trained_model)
    for key in (
        "predicted_strength_mpa",
        "co2_kg_per_m3",
        "cost_usd_per_m3",
        "carbon_intensity_kg_per_mpa",
        "reasons",
    ):
        assert key in card
    assert card["co2_kg_per_m3"] > 0
    assert isinstance(card["reasons"], list) and card["reasons"]


def test_missing_input_raises(trained_model):
    with pytest.raises(ValueError, match="missing inputs"):
        predict_strength({"cement": 300.0}, model=trained_model)


def test_more_cement_raises_strength(trained_model, sample_mix):
    lean = dict(sample_mix, cement=200.0)
    rich = dict(sample_mix, cement=450.0)
    assert predict_strength(rich, model=trained_model) > predict_strength(lean, model=trained_model)
