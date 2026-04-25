"""
FastAPI backend for real-time mechanical property prediction.

Endpoints:
    POST /predict   — Predict UTS & YS for a single rebar sample
    GET  /health    — Health check
    GET  /model-info — Model metadata for a given diameter
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    FEATURE_RANGES,
    VALID_DIAMETERS,
    VALID_GRADES,
)
from src.models.predictor import ModelRegistry

# ─── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="Steel Rebar Property Prediction API",
    description=(
        "Predicts Ultimate Tensile Strength (UTS) and Yield Strength (YS) "
        "of steel rebars from manufacturing parameters."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model registry on startup
registry = ModelRegistry()


# ─── Pydantic Models ────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Input schema for a prediction request."""

    diameter: int = Field(..., description="Rebar diameter: 10, 12, or 16 mm")
    grade: str = Field(..., description="Steel grade: GR1, GR2, or GR3")
    CHEM1: float = Field(..., description="Chemical Composition 1")
    CHEM2: float = Field(..., description="Chemical Composition 2")
    CHEM3: float = Field(..., description="Chemical Composition 3")
    CHEM4: float = Field(..., description="Chemical Composition 4")
    CHEM5: float = Field(..., description="Chemical Composition 5")
    CHEM6: float = Field(..., description="Chemical Composition 6")
    CHEM7: float = Field(..., description="Chemical Composition 7")
    CHEM8: float = Field(..., description="Chemical Composition 8")
    CHEM9: float = Field(..., description="Chemical Composition 9")
    CHEM10: float = Field(..., description="Chemical Composition 10")
    TEMP1: float = Field(..., description="Temperature 1")
    TEMP2: float = Field(..., description="Temperature 2")
    TEMP3: float = Field(..., description="Temperature 3")
    TEMP4: float = Field(..., description="Temperature 4")
    TEMP5: float = Field(..., description="Temperature 5")
    TEMP6: float = Field(..., description="Temperature 6")
    SPEED: float = Field(..., description="Rolling Speed")
    PROCESS1: float = Field(..., description="Process Parameter 1")
    PROCESS2: float = Field(..., description="Process Parameter 2")
    PROCESS3: float = Field(..., description="Process Parameter 3")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "diameter": 12,
                    "grade": "GR1",
                    "CHEM1": 0.225,
                    "CHEM2": 1.297,
                    "CHEM3": 0.017,
                    "CHEM4": 0.024,
                    "CHEM5": 0.230,
                    "CHEM6": 0.0026,
                    "CHEM7": 0.0005,
                    "CHEM8": 0.028,
                    "CHEM9": 0.006,
                    "CHEM10": 70.5,
                    "TEMP1": 1000,
                    "TEMP2": 965.0,
                    "TEMP3": 70,
                    "TEMP4": 158,
                    "TEMP5": 581.0,
                    "TEMP6": 549,
                    "SPEED": 21,
                    "PROCESS1": 2521,
                    "PROCESS2": 2666,
                    "PROCESS3": 1.32,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for a prediction response."""

    quality1_uts: float = Field(..., description="Predicted UTS (MPa)")
    quality2_ys: float = Field(..., description="Predicted YS (MPa)")
    uts_ys_ratio: float = Field(..., description="UTS/YS ratio")
    passes_quality_gate: bool = Field(
        ..., description="Whether UTS/YS ≥ 1.1 (IS 1786)"
    )
    model_used: str = Field(..., description="Algorithm used for prediction")
    diameter: int = Field(..., description="Rebar diameter (mm)")


class HealthResponse(BaseModel):
    status: str
    available_models: list[int]


class ModelInfoResponse(BaseModel):
    diameter: int
    algorithm: str
    feature_count: int
    metrics: dict


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and available models."""
    return HealthResponse(
        status="healthy",
        available_models=registry.get_available_diameters(),
    )


@app.get("/model-info/{diameter}", response_model=ModelInfoResponse)
def get_model_info(diameter: int):
    """Get metadata about a trained model."""
    if diameter not in VALID_DIAMETERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diameter {diameter}. Must be one of {VALID_DIAMETERS}",
        )

    try:
        metadata = registry.get_metadata(diameter)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ModelInfoResponse(
        diameter=metadata["diameter"],
        algorithm=metadata["algorithm"],
        feature_count=metadata["feature_count"],
        metrics=metadata["metrics"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict mechanical properties for a single rebar sample."""
    if request.diameter not in VALID_DIAMETERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diameter {request.diameter}. "
            f"Must be one of {VALID_DIAMETERS}",
        )
    if request.grade not in VALID_GRADES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid grade '{request.grade}'. "
            f"Must be one of {VALID_GRADES}",
        )

    # Build features dict (exclude diameter and grade)
    features = {
        k: v
        for k, v in request.model_dump().items()
        if k not in ("diameter", "grade")
    }

    try:
        result = registry.predict(
            diameter=request.diameter,
            grade=request.grade,
            features=features,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        quality1_uts=result.quality1,
        quality2_ys=result.quality2,
        uts_ys_ratio=result.uts_ys_ratio,
        passes_quality_gate=result.passes_quality_gate,
        model_used=result.model_name,
        diameter=result.diameter,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
