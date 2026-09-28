import pandas as pd
import pytest

from curecast.config import RAW_FEATURES, TARGET
from curecast.data import load_dataset, split_xy, validate


def test_load_dataset_shape():
    df = load_dataset()
    assert len(df) == 1030
    assert set(RAW_FEATURES) | {TARGET} <= set(df.columns)
    assert not df.isna().any().any()


def test_split_xy_separates_target():
    X, y = split_xy(load_dataset())
    assert list(X.columns) == RAW_FEATURES
    assert TARGET not in X.columns
    assert len(X) == len(y) == 1030


def test_validate_rejects_missing_column():
    df = load_dataset().drop(columns=["water"])
    with pytest.raises(ValueError, match="missing columns"):
        validate(df)


def test_validate_rejects_out_of_range():
    df = load_dataset().copy()
    df.loc[0, "cement"] = 9999.0
    with pytest.raises(ValueError, match="cement"):
        validate(df)


def test_validate_rejects_nan():
    df = load_dataset().copy()
    df.loc[0, "water"] = pd.NA
    with pytest.raises(ValueError, match="missing values"):
        validate(df)
