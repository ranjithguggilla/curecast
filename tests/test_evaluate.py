import numpy as np

from curecast.config import RAW_FEATURES
from curecast.evaluate import explain_mix, regression_metrics, sanity_feature_order


def test_regression_metrics_perfect_fit():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["r2"] == 1.0
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0


def test_regression_metrics_known_error():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0])
    m = regression_metrics(y_true, y_pred)
    assert np.isclose(m["rmse"], 2.0)
    assert np.isclose(m["mae"], 2.0)


def test_explain_mix_flags_high_water():
    notes = explain_mix({"cement": 200, "slag": 0, "ash": 0, "water": 140, "age": 28})
    assert any("water-to-binder" in n for n in notes)


def test_explain_mix_flags_high_scm():
    notes = explain_mix({"cement": 150, "slag": 150, "ash": 100, "water": 160, "age": 28})
    assert any("slag or fly ash" in n for n in notes)


def test_sanity_feature_order():
    assert sanity_feature_order(RAW_FEATURES + ["binder"])
    assert not sanity_feature_order(["age"] + RAW_FEATURES[:-1])
