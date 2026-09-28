"""Shared fixtures.

The model fixture trains the shipped estimator on the real data once per test
session. Training the pipeline is a couple of seconds, and doing it here keeps
the tests independent of whether anyone has run the full pipeline first, which
is exactly the situation continuous integration lands in.
"""

import pytest

from curecast.data import load_dataset, split_xy
from curecast.model import build_model_zoo


@pytest.fixture(scope="session")
def trained_model():
    df = load_dataset()
    X, y = split_xy(df)
    model = build_model_zoo()["gradient_boosting"]
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def sample_mix():
    return {
        "cement": 350.0, "slag": 80.0, "ash": 60.0, "water": 180.0,
        "superplastic": 6.0, "coarseagg": 968.0, "fineagg": 773.0, "age": 28.0,
    }
