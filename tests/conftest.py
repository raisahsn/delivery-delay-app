"""
conftest.py
-------------
Shared pytest fixtures.

We deliberately do NOT depend on the real trained model.pkl (which requires
the proprietary Delivery_Logistics.csv dataset) so that CI can run tests
anywhere. Instead, `dummy_pipeline` trains a tiny sklearn pipeline with the
exact same structure (ColumnTransformer with 'num'/'cat' steps + a 'clf'
step) on synthetic data, so it exercises the real code paths in
app/model_utils.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.model_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def _make_synthetic_dataframe(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    categories = {
        "delivery_partner": ["delhivery", "xpressbees", "shadowfax", "dhl", "ekart"],
        "package_type": ["electronics", "groceries", "cosmetics", "automobile parts"],
        "vehicle_type": ["bike", "ev van", "truck"],
        "delivery_mode": ["same day", "express", "standard", "two day"],
        "region": ["west", "east", "north", "south", "central"],
        "weather_condition": ["clear", "cold", "rainy", "stormy", "hot", "foggy"],
    }

    data = {
        "distance_km": rng.uniform(1, 500, n),
        "package_weight_kg": rng.uniform(0.1, 60, n),
        "expected_time_hours": rng.uniform(1, 48, n),
        "delivery_cost": rng.uniform(50, 5000, n),
    }
    for col, cats in categories.items():
        data[col] = rng.choice(cats, n)

    df = pd.DataFrame(data)

    # Simple synthetic rule so the model learns *something* non-trivial:
    # stormy weather + express mode => higher delay probability.
    risk = (
        (df["weather_condition"].isin(["stormy", "rainy"])).astype(int) * 0.5
        + (df["delivery_mode"] == "express").astype(int) * 0.3
        + rng.uniform(0, 0.4, n)
    )
    df["delayed_flag"] = (risk > 0.5).astype(int)

    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_dataframe()


@pytest.fixture(scope="session")
def dummy_pipeline(synthetic_df: pd.DataFrame) -> Pipeline:
    """A small, fast, fully-fitted pipeline matching the production structure."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]
    )
    X = synthetic_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = synthetic_df["delayed_flag"]
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def valid_input_row() -> dict:
    return {
        "distance_km": 210.0,
        "package_weight_kg": 18.4,
        "expected_time_hours": 6,
        "delivery_cost": 1200.0,
        "delivery_partner": "xpressbees",
        "package_type": "electronics",
        "vehicle_type": "bike",
        "delivery_mode": "express",
        "region": "east",
        "weather_condition": "stormy",
    }
