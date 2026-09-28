import numpy as np

from curecast.config import RAW_FEATURES
from curecast.data import load_dataset
from curecast.features import ENGINEERED_FEATURES, MixFeatureBuilder, add_mix_features


def test_add_mix_features_adds_columns():
    df = load_dataset().head(20)
    out = add_mix_features(df)
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    assert not out[ENGINEERED_FEATURES].isna().any().any()


def test_water_binder_ratio_is_correct():
    df = load_dataset().head(5)
    out = add_mix_features(df)
    binder = df["cement"] + df["slag"] + df["ash"]
    expected = df["water"] / (binder + 1e-6)
    assert np.allclose(out["water_binder_ratio"], expected)


def test_scm_fraction_between_zero_and_one():
    out = add_mix_features(load_dataset())
    assert (out["scm_fraction"] >= 0).all()
    assert (out["scm_fraction"] <= 1).all()


def test_transformer_output_shape_and_names():
    X = load_dataset()[RAW_FEATURES].head(10)
    builder = MixFeatureBuilder().fit(X)
    arr = builder.transform(X)
    assert arr.shape == (10, len(RAW_FEATURES) + len(ENGINEERED_FEATURES))
    names = list(builder.get_feature_names_out())
    assert names[: len(RAW_FEATURES)] == RAW_FEATURES
    assert names[len(RAW_FEATURES):] == ENGINEERED_FEATURES


def test_transformer_accepts_numpy():
    X = load_dataset()[RAW_FEATURES].head(3).to_numpy()
    arr = MixFeatureBuilder().fit(X).transform(X)
    assert arr.shape[1] == len(RAW_FEATURES) + len(ENGINEERED_FEATURES)
