from __future__ import annotations

import pytest

from app.model_utils import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    PredictionResult,
    get_categorical_options,
    get_feature_importance,
    predict_delay,
    validate_input,
)


class TestGetCategoricalOptions:
    def test_returns_all_categorical_features(self, dummy_pipeline):
        options = get_categorical_options(dummy_pipeline)
        assert set(options.keys()) == set(CATEGORICAL_FEATURES)

    def test_options_are_nonempty_lists(self, dummy_pipeline):
        options = get_categorical_options(dummy_pipeline)
        for feature, values in options.items():
            assert isinstance(values, list)
            assert len(values) > 0
            assert all(isinstance(v, str) for v in values)

    def test_known_category_present(self, dummy_pipeline):
        options = get_categorical_options(dummy_pipeline)
        assert "express" in options["delivery_mode"]
        assert "stormy" in options["weather_condition"]


class TestValidateInput:
    def test_valid_input_has_no_errors(self, valid_input_row):
        assert validate_input(valid_input_row) == []

    def test_missing_field_reported(self, valid_input_row):
        incomplete = dict(valid_input_row)
        del incomplete["distance_km"]
        errors = validate_input(incomplete)
        assert any("distance_km" in e for e in errors)

    def test_negative_numeric_rejected(self, valid_input_row):
        bad = dict(valid_input_row)
        bad["package_weight_kg"] = -5
        errors = validate_input(bad)
        assert any("package_weight_kg" in e for e in errors)

    def test_non_numeric_value_rejected(self, valid_input_row):
        bad = dict(valid_input_row)
        bad["delivery_cost"] = "mahal sekali"
        errors = validate_input(bad)
        assert any("delivery_cost" in e for e in errors)

    def test_empty_string_treated_as_missing(self, valid_input_row):
        bad = dict(valid_input_row)
        bad["region"] = ""
        errors = validate_input(bad)
        assert any("region" in e for e in errors)


class TestPredictDelay:
    def test_returns_prediction_result(self, dummy_pipeline, valid_input_row):
        result = predict_delay(dummy_pipeline, valid_input_row)
        assert isinstance(result, PredictionResult)

    def test_label_matches_is_delayed_flag(self, dummy_pipeline, valid_input_row):
        result = predict_delay(dummy_pipeline, valid_input_row)
        if result.is_delayed:
            assert result.label == "Delayed"
        else:
            assert result.label == "On-time"

    def test_probability_within_bounds(self, dummy_pipeline, valid_input_row):
        result = predict_delay(dummy_pipeline, valid_input_row)
        assert 0.0 <= result.probability <= 1.0

    def test_risk_level_is_valid_category(self, dummy_pipeline, valid_input_row):
        result = predict_delay(dummy_pipeline, valid_input_row)
        assert result.risk_level in {"Low", "Medium", "High"}

    def test_high_risk_probability_higher_than_low_risk(
        self, dummy_pipeline, valid_input_row
    ):
        # storm + express (per synthetic rule in conftest) should score
        # meaningfully higher than clear weather + standard mode.
        high_risk = dict(
            valid_input_row, weather_condition="stormy", delivery_mode="express"
        )
        low_risk = dict(
            valid_input_row, weather_condition="clear", delivery_mode="standard"
        )

        high_result = predict_delay(dummy_pipeline, high_risk)
        low_result = predict_delay(dummy_pipeline, low_risk)

        assert high_result.probability > low_result.probability

    def test_raises_on_invalid_input(self, dummy_pipeline, valid_input_row):
        bad = dict(valid_input_row)
        del bad["distance_km"]
        with pytest.raises(ValueError):
            predict_delay(dummy_pipeline, bad)

    def test_uses_only_expected_features(self, dummy_pipeline, valid_input_row):
        # Extra/unexpected keys in the input dict should simply be ignored,
        # not break prediction (defensive behaviour for future UI changes).
        extra = dict(valid_input_row, delivery_status="delivered", delivery_rating=5)
        result = predict_delay(dummy_pipeline, extra)
        assert isinstance(result, PredictionResult)
        allowed_keys = set(valid_input_row.keys()) | {
            "delivery_status",
            "delivery_rating",
        }
        assert set(ALL_FEATURES).issubset(allowed_keys)


class TestFeatureImportance:
    def test_returns_series_for_supported_model(self, dummy_pipeline):
        fi = get_feature_importance(dummy_pipeline, top_n=5)
        assert fi is not None
        assert len(fi) <= 5
