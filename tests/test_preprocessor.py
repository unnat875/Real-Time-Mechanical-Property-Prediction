"""Tests for the data preprocessing pipeline."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    IMPUTE_BY_DIAMETER_COLS,
    IMPUTE_BY_GRADE_COLS,
    TARGET_COLS,
    UTS_YS_RATIO_COL,
    VALID_DIAMETERS,
)
from src.data.loader import load_raw_data, split_by_diameter, validate_data
from src.data.preprocessor import (
    add_derived_features,
    impute_by_diameter,
    impute_by_grade,
    impute_targets,
    preprocess_pipeline,
)


@pytest.fixture
def raw_df():
    """Load raw data for testing."""
    return load_raw_data()


@pytest.fixture
def validated_df(raw_df):
    """Validated raw data."""
    return validate_data(raw_df)


@pytest.fixture
def preprocessed_df(validated_df):
    """Fully preprocessed data."""
    return preprocess_pipeline(validated_df)


class TestLoader:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_load_has_rows(self, raw_df):
        assert len(raw_df) > 5000

    def test_column_rename(self, raw_df):
        """PEOCESS3 should be renamed to PROCESS3."""
        assert "PROCESS3" in raw_df.columns
        assert "PEOCESS3" not in raw_df.columns

    def test_validation_passes(self, validated_df):
        assert validated_df is not None

    def test_split_by_diameter_sizes(self, preprocessed_df):
        splits = split_by_diameter(preprocessed_df)
        total = sum(len(v) for v in splits.values())
        assert total == len(preprocessed_df)
        assert set(splits.keys()) == set(VALID_DIAMETERS)


class TestPreprocessor:
    def test_no_nulls_in_grade_cols_after_imputation(self, validated_df):
        result = impute_by_grade(validated_df)
        for col in IMPUTE_BY_GRADE_COLS:
            assert result[col].isnull().sum() == 0, f"Nulls remain in {col}"

    def test_no_nulls_in_diameter_cols_after_imputation(self, validated_df):
        result = impute_by_grade(validated_df)
        result = impute_by_diameter(result)
        for col in IMPUTE_BY_DIAMETER_COLS:
            assert result[col].isnull().sum() == 0, f"Nulls remain in {col}"

    def test_no_nulls_in_targets_after_imputation(self, validated_df):
        result = impute_by_grade(validated_df)
        result = impute_by_diameter(result)
        result = impute_targets(result)
        for col in TARGET_COLS:
            assert result[col].isnull().sum() == 0, f"Nulls remain in {col}"

    def test_full_pipeline_no_nulls(self, preprocessed_df):
        critical = IMPUTE_BY_GRADE_COLS + IMPUTE_BY_DIAMETER_COLS + TARGET_COLS
        nulls = preprocessed_df[critical].isnull().sum().sum()
        assert nulls == 0

    def test_uts_ys_ratio_exists(self, preprocessed_df):
        assert UTS_YS_RATIO_COL in preprocessed_df.columns

    def test_uts_ys_ratio_positive(self, preprocessed_df):
        assert (preprocessed_df[UTS_YS_RATIO_COL] > 0).all()

    def test_pipeline_preserves_row_count(self, validated_df):
        result = preprocess_pipeline(validated_df)
        assert len(result) == len(validated_df)
