"""Tests for the prediction module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VALID_DIAMETERS, UTS_YS_MIN_RATIO
from src.models.predictor import ModelRegistry, PredictionResult


@pytest.fixture
def registry():
    return ModelRegistry()


SAMPLE_FEATURES = {
    "CHEM1": 0.225, "CHEM2": 1.297, "CHEM3": 0.017, "CHEM4": 0.024,
    "CHEM5": 0.230, "CHEM6": 0.0026, "CHEM7": 0.0005, "CHEM8": 0.028,
    "CHEM9": 0.006, "CHEM10": 70.5,
    "TEMP1": 1000, "TEMP2": 965.0, "TEMP3": 70, "TEMP4": 158,
    "TEMP5": 581.0, "TEMP6": 549,
    "SPEED": 21, "PROCESS1": 2521, "PROCESS2": 2666, "PROCESS3": 1.32,
}


class TestModelRegistry:
    def test_available_diameters(self, registry):
        available = registry.get_available_diameters()
        assert len(available) > 0
        for d in available:
            assert d in VALID_DIAMETERS

    def test_get_metadata(self, registry):
        for d in registry.get_available_diameters():
            meta = registry.get_metadata(d)
            assert "algorithm" in meta
            assert "metrics" in meta
            assert "feature_count" in meta


class TestPrediction:
    @pytest.mark.parametrize("diameter", VALID_DIAMETERS)
    def test_prediction_returns_result(self, registry, diameter):
        result = registry.predict(
            diameter=diameter,
            grade="GR1",
            features=SAMPLE_FEATURES.copy(),
        )
        assert isinstance(result, PredictionResult)

    @pytest.mark.parametrize("diameter", VALID_DIAMETERS)
    def test_prediction_shape(self, registry, diameter):
        """Prediction should return two quality values."""
        result = registry.predict(
            diameter=diameter,
            grade="GR1",
            features=SAMPLE_FEATURES.copy(),
        )
        assert result.quality1 > 0
        assert result.quality2 > 0

    @pytest.mark.parametrize("diameter", VALID_DIAMETERS)
    def test_prediction_plausible_range(self, registry, diameter):
        """UTS and YS should be in physically plausible range (400-900 MPa)."""
        result = registry.predict(
            diameter=diameter,
            grade="GR1",
            features=SAMPLE_FEATURES.copy(),
        )
        assert 400 < result.quality1 < 900, f"UTS out of range: {result.quality1}"
        assert 400 < result.quality2 < 900, f"YS out of range: {result.quality2}"

    def test_quality_gate_pass(self, registry):
        """Test quality gate logic."""
        result = registry.predict(
            diameter=12,
            grade="GR1",
            features=SAMPLE_FEATURES.copy(),
        )
        expected_pass = result.uts_ys_ratio >= UTS_YS_MIN_RATIO
        assert result.passes_quality_gate == expected_pass

    def test_uts_ys_ratio_calculation(self, registry):
        result = registry.predict(
            diameter=12,
            grade="GR1",
            features=SAMPLE_FEATURES.copy(),
        )
        expected_ratio = round(result.quality1 / result.quality2, 4)
        assert result.uts_ys_ratio == expected_ratio

    def test_invalid_diameter_raises(self, registry):
        with pytest.raises(ValueError, match="Invalid diameter"):
            registry.predict(
                diameter=15,
                grade="GR1",
                features=SAMPLE_FEATURES.copy(),
            )

    @pytest.mark.parametrize("grade", ["GR1", "GR2", "GR3"])
    def test_all_grades(self, registry, grade):
        result = registry.predict(
            diameter=12,
            grade=grade,
            features=SAMPLE_FEATURES.copy(),
        )
        assert result.quality1 > 0
