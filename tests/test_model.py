from sklearn.pipeline import Pipeline

from curecast.config import RAW_FEATURES
from curecast.data import load_dataset, split_xy
from curecast.model import build_model_zoo


def test_zoo_contains_baseline_and_ensembles():
    zoo = build_model_zoo()
    assert "linear" in zoo
    assert "gradient_boosting" in zoo
    assert all(isinstance(p, Pipeline) for p in zoo.values())


def test_scaling_lives_inside_pipeline():
    # Every linear or distance model must carry its scaler so folds refit it.
    zoo = build_model_zoo()
    for name in ("linear", "ridge", "elasticnet", "knn", "svr"):
        assert "scaler" in dict(zoo[name].named_steps)


def test_pipeline_fits_and_predicts_in_range():
    X, y = split_xy(load_dataset())
    model = build_model_zoo()["random_forest"].fit(X.head(300), y.head(300))
    preds = model.predict(X.head(10))
    assert preds.shape == (10,)
    assert (preds > 0).all() and (preds < 100).all()


def test_first_pipeline_step_is_feature_builder():
    zoo = build_model_zoo()
    first_step = list(zoo["gradient_boosting"].named_steps)[0]
    assert first_step == "features"
    assert RAW_FEATURES  # column contract exists
