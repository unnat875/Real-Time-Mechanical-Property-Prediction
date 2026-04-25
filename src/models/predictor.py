"""
Inference-time prediction module.

Loads trained model artifacts and provides a clean prediction API
for both single records and batch inputs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.config import MODELS_DIR, UTS_YS_MIN_RATIO, VALID_DIAMETERS
from src.features.encoder import prepare_inference_input

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for a single prediction output."""

    quality1: float  # UTS (MPa)
    quality2: float  # YS (MPa)
    uts_ys_ratio: float
    passes_quality_gate: bool
    model_name: str
    diameter: int


class ModelRegistry:
    """
    Loads and manages all diameter-specific model artifacts.

    Usage::

        registry = ModelRegistry()
        result = registry.predict(diameter=12, grade="GR1", features={...})
    """

    def __init__(self, models_dir: Path | None = None):
        self._models_dir = models_dir or MODELS_DIR
        self._cache: dict[int, dict] = {}

    def _load_diameter(self, diameter: int) -> dict:
        """Load model artifacts for a specific diameter."""
        if diameter in self._cache:
            return self._cache[diameter]

        model_dir = self._models_dir / f"diameter_{diameter}"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No trained model found for diameter {diameter}mm. "
                f"Run train.py first. Expected: {model_dir}"
            )

        artifacts = {
            "model": joblib.load(model_dir / "model.joblib"),
            "encoder": joblib.load(model_dir / "encoder.joblib"),
            "feature_names": joblib.load(model_dir / "feature_names.joblib"),
            "metadata": joblib.load(model_dir / "metadata.joblib"),
        }
        self._cache[diameter] = artifacts
        logger.info(
            f"Loaded model for diameter {diameter}mm: "
            f"{artifacts['metadata']['algorithm']}"
        )
        return artifacts

    def is_loaded(self, diameter: int) -> bool:
        """Check if a model is loaded for the given diameter."""
        return diameter in self._cache

    def get_available_diameters(self) -> list[int]:
        """Return list of diameters that have trained models."""
        available = []
        for d in VALID_DIAMETERS:
            model_dir = self._models_dir / f"diameter_{d}"
            if (model_dir / "model.joblib").exists():
                available.append(d)
        return available

    def get_metadata(self, diameter: int) -> dict:
        """Get model metadata for a diameter."""
        artifacts = self._load_diameter(diameter)
        return artifacts["metadata"]

    def predict(
        self,
        diameter: int,
        grade: str,
        features: dict,
    ) -> PredictionResult:
        """
        Make a prediction for a single rebar sample.

        Parameters
        ----------
        diameter : int
            Rebar diameter (10, 12, or 16).
        grade : str
            Steel grade (GR1, GR2, or GR3).
        features : dict
            Dictionary of feature name → value for all required features.

        Returns
        -------
        PredictionResult
            Predicted QUALITY1, QUALITY2, ratio, and quality gate status.
        """
        if diameter not in VALID_DIAMETERS:
            raise ValueError(
                f"Invalid diameter {diameter}. Must be one of {VALID_DIAMETERS}"
            )

        artifacts = self._load_diameter(diameter)
        model = artifacts["model"]
        encoder = artifacts["encoder"]
        feature_names = artifacts["feature_names"]

        # Build input record
        record = {**features, "GRADE": grade}

        # Prepare feature vector
        X = prepare_inference_input(record, encoder, feature_names)

        # Predict
        prediction = model.predict(X)
        q1 = float(prediction[0][0])
        q2 = float(prediction[0][1])
        ratio = q1 / q2 if q2 > 0 else 0.0

        return PredictionResult(
            quality1=round(q1, 3),
            quality2=round(q2, 3),
            uts_ys_ratio=round(ratio, 4),
            passes_quality_gate=ratio >= UTS_YS_MIN_RATIO,
            model_name=artifacts["metadata"]["algorithm"],
            diameter=diameter,
        )

    def predict_batch(
        self,
        diameter: int,
        records: list[dict],
    ) -> list[PredictionResult]:
        """
        Make predictions for multiple records.

        Each record must contain 'GRADE' and all feature columns.
        """
        results = []
        for record in records:
            grade = record.pop("GRADE", record.get("GRADE"))
            result = self.predict(diameter, grade, record)
            results.append(result)
        return results
