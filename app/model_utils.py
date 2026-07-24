"""
model_utils.py
----------------
Utility layer between the trained sklearn pipeline (model.pkl, produced by the
notebook / scripts/train_model.py) and the Streamlit UI.

Design decisions:
- The full pipeline (ColumnTransformer + classifier) is loaded once and cached.
- Dropdown options for categorical fields are read directly from the fitted
  OneHotEncoder inside the pipeline, NOT hardcoded. This guarantees the UI
  always matches whatever categories the model was actually trained on,
  even after retraining with new data.
- No feature that is only known *after* delivery completion
  (delivery_time_hours, delivery_status, delivery_rating) is ever accepted
  as input — this mirrors the leakage-avoidance decision made in the
  notebook (Section 10 conclusions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

NUMERIC_FEATURES = [
    "distance_km",
    "package_weight_kg",
    "expected_time_hours",
    "delivery_cost",
]
CATEGORICAL_FEATURES = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Reasonable fallback ranges shown in the UI as slider bounds if we cannot
# infer better bounds from config.json (e.g. before first training run).
DEFAULT_NUMERIC_RANGES = {
    "distance_km": (0.0, 1000.0, 50.0),
    "package_weight_kg": (0.0, 100.0, 5.0),
    "expected_time_hours": (0.0, 72.0, 6.0),
    "delivery_cost": (0.0, 10000.0, 500.0),
}

# High-risk threshold used purely for UI colour-coding (not a business
# decision threshold — that should come from config.json / a proper
# cost-benefit analysis by the ops team).
RISK_THRESHOLD_MEDIUM = 0.4
RISK_THRESHOLD_HIGH = 0.7


class ModelNotFoundError(FileNotFoundError):
    """Raised when model.pkl is missing so the UI can show a friendly message."""


@dataclass
class PredictionResult:
    label: str                 # "Delayed" | "On-time"
    is_delayed: bool
    probability: float         # probability of delay, 0..1
    risk_level: str            # "Low" | "Medium" | "High"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pipeline(model_path: Path = MODEL_PATH):
    """Load the trained sklearn Pipeline (ColumnTransformer + classifier).

    Raises ModelNotFoundError with a clear message if the artifact is missing,
    so the Streamlit layer can render an actionable warning instead of a raw
    traceback.
    """
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run `python scripts/train_model.py` or copy your trained "
            "model.pkl into the model/ folder first."
        )
    return joblib.load(model_path)


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load model metadata (metrics, best params, feature list). Optional file."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_categorical_options(pipeline) -> dict[str, list[str]]:
    """Read the list of known categories for each categorical feature straight
    from the fitted OneHotEncoder inside the pipeline's ColumnTransformer.

    This avoids hardcoding dropdown values in the UI and keeps them in sync
    with whatever data the model was trained on.
    """
    preprocessor = pipeline.named_steps.get("prep")
    if preprocessor is None:
        raise ValueError("Pipeline has no 'prep' step (ColumnTransformer expected).")

    encoder = preprocessor.named_transformers_["cat"]
    options: dict[str, list[str]] = {}
    for feature_name, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        options[feature_name] = sorted(str(c) for c in categories)
    return options


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def validate_input(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty list = valid)."""
    errors = []

    missing = [f for f in ALL_FEATURES if f not in data or data[f] in (None, "")]
    if missing:
        errors.append(f"Kolom wajib diisi: {', '.join(missing)}")

    for field in NUMERIC_FEATURES:
        value = data.get(field)
        if value is not None and value != "":
            try:
                if float(value) < 0:
                    errors.append(f"'{field}' tidak boleh negatif.")
            except (TypeError, ValueError):
                errors.append(f"'{field}' harus berupa angka.")

    return errors


def _risk_level(probability: float) -> str:
    if probability >= RISK_THRESHOLD_HIGH:
        return "High"
    if probability >= RISK_THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"


def predict_delay(pipeline, data: dict[str, Any]) -> PredictionResult:
    """Run a single-row prediction through the pipeline.

    `data` must contain exactly the raw features the pipeline expects
    (see ALL_FEATURES) — the pipeline itself handles scaling/encoding.
    """
    errors = validate_input(data)
    if errors:
        raise ValueError("; ".join(errors))

    row = {f: data[f] for f in ALL_FEATURES}
    X_new = pd.DataFrame([row])

    proba = float(pipeline.predict_proba(X_new)[:, 1][0])
    pred = int(pipeline.predict(X_new)[0])

    return PredictionResult(
        label="Delayed" if pred == 1 else "On-time",
        is_delayed=bool(pred == 1),
        probability=proba,
        risk_level=_risk_level(proba),
    )


def get_feature_importance(pipeline, top_n: int = 10) -> pd.Series | None:
    """Return top-N feature importances/coefficients if the model supports it."""
    clf = pipeline.named_steps.get("clf")
    preprocessor = pipeline.named_steps.get("prep")
    if clf is None or preprocessor is None:
        return None

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = abs(clf.coef_[0])
    else:
        return None

    return (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )
