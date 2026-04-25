"""Tests for feature encoding."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data, validate_data, split_by_diameter
from src.data.preprocessor import preprocess_pipeline
from src.features.encoder import prepare_features, prepare_inference_input


@pytest.fixture
def sample_df():
    """Load and preprocess a small diameter subset for testing."""
    df = load_raw_data()
    df = validate_data(df)
    df = preprocess_pipeline(df)
    splits = split_by_diameter(df)
    return splits[10]  # Use diameter 10 (smallest)


class TestPrepareFeatures:
    def test_returns_correct_types(self, sample_df):
        X, y, encoder, feature_names = prepare_features(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert encoder is not None
        assert isinstance(feature_names, list)

    def test_targets_not_in_features(self, sample_df):
        X, y, _, _ = prepare_features(sample_df)
        assert "QUALITY1" not in X.columns
        assert "QUALITY2" not in X.columns

    def test_metadata_not_in_features(self, sample_df):
        X, _, _, _ = prepare_features(sample_df)
        assert "DATE_TIME" not in X.columns
        assert "ID" not in X.columns
        assert "DIAMETER" not in X.columns

    def test_grade_encoded(self, sample_df):
        X, _, _, feature_names = prepare_features(sample_df)
        grade_cols = [c for c in feature_names if c.startswith("GRADE_")]
        assert len(grade_cols) >= 2  # At least 2 grades

    def test_all_numeric(self, sample_df):
        X, _, _, _ = prepare_features(sample_df)
        assert X.select_dtypes(include=[np.number]).shape == X.shape

    def test_feature_count(self, sample_df):
        X, _, _, feature_names = prepare_features(sample_df)
        assert X.shape[1] == len(feature_names)


class TestPrepareInferenceInput:
    def test_output_shape(self, sample_df):
        _, _, encoder, feature_names = prepare_features(sample_df)
        record = {
            "GRADE": "GR1",
            "CHEM1": 0.225, "CHEM2": 1.297, "CHEM3": 0.017, "CHEM4": 0.024,
            "CHEM5": 0.230, "CHEM6": 0.0026, "CHEM7": 0.0005, "CHEM8": 0.028,
            "CHEM9": 0.006, "CHEM10": 70.5,
            "TEMP1": 1000, "TEMP2": 965.0, "TEMP3": 70, "TEMP4": 158,
            "TEMP5": 581.0, "TEMP6": 549,
            "SPEED": 21, "PROCESS1": 2521, "PROCESS2": 2666, "PROCESS3": 1.32,
        }
        X = prepare_inference_input(record, encoder, feature_names)
        assert X.shape == (1, len(feature_names))

    def test_missing_feature_raises(self, sample_df):
        _, _, encoder, feature_names = prepare_features(sample_df)
        record = {"GRADE": "GR1", "CHEM1": 0.225}  # Missing most features
        with pytest.raises(ValueError, match="Missing feature"):
            prepare_inference_input(record, encoder, feature_names)

    def test_missing_grade_raises(self, sample_df):
        _, _, encoder, feature_names = prepare_features(sample_df)
        record = {"CHEM1": 0.225}  # No GRADE
        with pytest.raises(ValueError, match="GRADE"):
            prepare_inference_input(record, encoder, feature_names)
