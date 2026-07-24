from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from app.model_utils import (
    ModelNotFoundError,
    load_config,
    load_pipeline,
)


class TestLoadPipeline:
    def test_raises_friendly_error_when_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.pkl"
        with pytest.raises(ModelNotFoundError):
            load_pipeline(model_path=missing_path)

    def test_loads_successfully_when_present(self, dummy_pipeline, tmp_path):
        model_path = tmp_path / "model.pkl"
        joblib.dump(dummy_pipeline, model_path)

        loaded = load_pipeline(model_path=model_path)
        assert hasattr(loaded, "predict")
        assert hasattr(loaded, "predict_proba")


class TestLoadConfig:
    def test_returns_empty_dict_when_missing(self, tmp_path):
        missing_path = tmp_path / "config.json"
        assert load_config(config_path=missing_path) == {}

    def test_loads_existing_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text('{"model_type": "Logistic Regression"}', encoding="utf-8")

        config = load_config(config_path=config_path)
        assert config["model_type"] == "Logistic Regression"
