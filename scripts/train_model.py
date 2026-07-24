"""
train_model.py
-----------------
Replikasi pipeline training dari notebook (Delivery_Logistics_Predictive_Analysis.ipynb)
sebagai script yang bisa dijalankan ulang untuk menghasilkan artefak model produksi.

Fitur yang dipakai (sesuai keputusan notebook — menghindari data leakage):
    numeric:     distance_km, package_weight_kg, expected_time_hours, delivery_cost
    categorical: delivery_partner, package_type, vehicle_type, delivery_mode,
                 region, weather_condition
Target: delayed_flag (1 = terlambat, 0 = tepat waktu), diturunkan dari kolom `delayed`.

Kolom yang SENGAJA dikeluarkan (hanya tersedia setelah pengiriman selesai):
    delivery_time_hours, delivery_status, delivery_rating

Usage:
    python scripts/train_model.py --data Delivery_Logistics.csv --out model/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42

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

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10], "clf__penalty": ["l2"]},
    "Random Forest": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [4, 6, 8, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def select_best_model(X_train, y_train, preprocessor) -> str:
    """Compare candidate models via 5-fold CV ROC-AUC, return the winning name."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for name, model in CANDIDATE_MODELS.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        result = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        scores[name] = result["test_score"].mean()
        print(f"  {name}: CV ROC-AUC = {scores[name]:.4f}")
    best_name = max(scores, key=scores.get)
    print(f"Model terbaik (CV ROC-AUC): {best_name}")
    return best_name


def tune_best_model(best_name, X_train, y_train, preprocessor):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocessor), ("clf", CANDIDATE_MODELS[best_name])])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_DISTRIBUTIONS[best_name],
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC (tuned): {search.best_score_:.4f}")
    return search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path ke Delivery_Logistics.csv")
    parser.add_argument("--out", default="model", help="Folder output artefak model")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip hyperparameter search (pakai default params) — untuk iterasi cepat.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Membaca data dari {args.data} ...")
    df = pd.read_csv(args.data)
    df["delayed_flag"] = df["delayed"].map({"no": 0, "yes": 1})

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["delayed_flag"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor()

    print("Membandingkan model kandidat...")
    best_name = select_best_model(X_train, y_train, preprocessor)

    if args.quick:
        best_pipeline = Pipeline([("prep", preprocessor), ("clf", CANDIDATE_MODELS[best_name])])
        best_pipeline.fit(X_train, y_train)
        best_params = {}
        cv_score = None
    else:
        print("Hyperparameter tuning...")
        search = tune_best_model(best_name, X_train, y_train, preprocessor)
        best_pipeline = search.best_estimator_
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        cv_score = float(search.best_score_)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    print("Metrik pada data test:", json.dumps(test_metrics, indent=2))

    joblib.dump(best_pipeline, out_dir / "model.pkl")

    config = {
        "task": "delivery_delay_prediction",
        "target": "delayed_flag (1 = terlambat, 0 = tepat waktu)",
        "model_type": best_name,
        "best_params": best_params,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "cv_roc_auc": cv_score,
        "test_metrics": test_metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_pipeline_steps": [name for name, _ in best_pipeline.steps],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    sample_input = X_test.iloc[[0]].to_dict(orient="records")[0]
    with open(out_dir / "sample_input.json", "w", encoding="utf-8") as f:
        json.dump(sample_input, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nArtefak tersimpan di: {out_dir.resolve()}")
    print("Isi folder:", [p.name for p in out_dir.iterdir()])


if __name__ == "__main__":
    main()
