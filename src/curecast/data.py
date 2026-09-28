"""Loading and sanity-checking the concrete mix dataset.

The raw file is I-Cheng Yeh's high-performance concrete study: 1,030 real
laboratory mixes with the eight design inputs and the measured compressive
strength. I keep the validation strict because a bad row here quietly poisons
every model downstream, and the ranges below come from what is physically
sensible for a structural mix.
"""

from __future__ import annotations

import pandas as pd

from .config import DATA_FILE, RAW_FEATURES, TARGET

# Plausible bounds for each column. Anything outside these is almost certainly a
# data-entry problem rather than an exotic mix, so I fail loudly instead of
# training on it.
_BOUNDS = {
    "cement": (50.0, 700.0),
    "slag": (0.0, 400.0),
    "ash": (0.0, 300.0),
    "water": (100.0, 300.0),
    "superplastic": (0.0, 40.0),
    "coarseagg": (700.0, 1200.0),
    "fineagg": (500.0, 1000.0),
    "age": (1.0, 400.0),
    "strength": (1.0, 90.0),
}


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """Read the CSV straight off disk with no transformation."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the frame against the schema and physical bounds.

    Returns the frame unchanged when it passes so the call can be chained.
    Raises ValueError on the first problem it finds.
    """
    expected = set(RAW_FEATURES) | {TARGET}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"unexpected missing values in: {bad}")

    for col, (lo, hi) in _BOUNDS.items():
        outside = df[(df[col] < lo) | (df[col] > hi)]
        if len(outside):
            raise ValueError(
                f"{len(outside)} row(s) of '{col}' fall outside [{lo}, {hi}]"
            )
    return df


def load_dataset(path=DATA_FILE) -> pd.DataFrame:
    """Load and validate in one call. This is what the rest of the code uses."""
    df = load_raw(path)
    validate(df)
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    """Separate the design matrix from the strength target."""
    return df[RAW_FEATURES].copy(), df[TARGET].copy()
