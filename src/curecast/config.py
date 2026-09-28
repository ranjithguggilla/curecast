"""Project-wide constants: paths, the mix schema, and carbon/cost factors.

The carbon and price numbers are deliberately kept in one place so a reviewer
can see every assumption behind the money and CO2 figures without digging
through the optimiser. They are order-of-magnitude industry values, not vendor
quotes, and the README calls that out.
"""

from __future__ import annotations

from pathlib import Path

# Repository layout -----------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parents[1]
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_FILE = DATA_DIR / "concrete.csv"
MODEL_FILE = ASSETS_DIR / "curecast_model.joblib"
METRICS_FILE = ASSETS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# The eight mix inputs, in kilograms per cubic metre of concrete, plus curing
# age in days. Order matters: the trained pipeline expects exactly these names.
RAW_FEATURES = [
    "cement",
    "slag",
    "ash",
    "water",
    "superplastic",
    "coarseagg",
    "fineagg",
    "age",
]
TARGET = "strength"  # 28-day style compressive strength in megapascals

# Human-readable labels for charts and the UI.
FEATURE_LABELS = {
    "cement": "Cement (kg/m3)",
    "slag": "Blast-furnace slag (kg/m3)",
    "ash": "Fly ash (kg/m3)",
    "water": "Water (kg/m3)",
    "superplastic": "Superplasticiser (kg/m3)",
    "coarseagg": "Coarse aggregate (kg/m3)",
    "fineagg": "Fine aggregate (kg/m3)",
    "age": "Curing age (days)",
    "strength": "Compressive strength (MPa)",
}

# Embodied-carbon factors in kg CO2e per kg of material. These track the widely
# cited Inventory of Carbon and Energy ranges: Portland cement dominates,
# blast-furnace slag and fly ash are a fraction of it because they are
# industrial by-products, and aggregate and water are nearly free in carbon
# terms. Superplasticiser is carbon-heavy per kg but used in tiny doses.
CO2_FACTORS = {
    "cement": 0.912,
    "slag": 0.079,
    "ash": 0.004,
    "water": 0.0003,
    "superplastic": 1.88,
    "coarseagg": 0.0075,
    "fineagg": 0.0026,
}

# Indicative material cost in US dollars per tonne. Slag and fly ash sit below
# cement, which is what makes clinker substitution attractive on cost as well
# as carbon.
COST_PER_TONNE = {
    "cement": 125.0,
    "slag": 90.0,
    "ash": 55.0,
    "water": 1.0,
    "superplastic": 2100.0,
    "coarseagg": 18.0,
    "fineagg": 22.0,
}


def co2_per_m3(mix: dict[str, float]) -> float:
    """Embodied carbon of one cubic metre of the given mix, in kg CO2e."""
    return sum(mix.get(k, 0.0) * f for k, f in CO2_FACTORS.items())


def cost_per_m3(mix: dict[str, float]) -> float:
    """Material cost of one cubic metre of the given mix, in US dollars."""
    return sum(mix.get(k, 0.0) / 1000.0 * c for k, c in COST_PER_TONNE.items())
