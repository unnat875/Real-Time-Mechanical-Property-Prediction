"""
Model training module.

Trains and compares multiple regression algorithms (Random Forest, XGBoost)
using GridSearchCV, selects the best, and saves artifacts.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

from src.config import (
    CV_FOLDS,
    MODELS_DIR,
    RANDOM_STATE,
    RF_PARAM_GRID,
    SCORING_METRIC,
    TEST_SIZE,
    XGB_PARAM_GRID,
)
from src.features.encoder import prepare_features
from src.models.evaluator import compute_metrics, generate_evaluation_plots

logger = logging.getLogger(__name__)


def get_model_configs() -> dict:
    """
    Return model definitions and their hyperparameter grids.

    Returns
    -------
    dict
        Mapping of model_name → {"model": estimator, "params": param_grid}
    """
    return {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE),
            "params": RF_PARAM_GRID,
        },
        "XGBoost": {
            "model": XGBRegressor(
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            ),
            "params": XGB_PARAM_GRID,
        },
    }


def train_single_model(
    model_name: str,
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> tuple:
    """
    Train a single model with GridSearchCV.

    Returns
    -------
    tuple
        (best_model, best_params, best_cv_score, training_time_seconds)
    """
    logger.info(f"  Training {model_name}...")
    n_fits = 1
    for v in param_grid.values():
        n_fits *= len(v)
    n_fits *= CV_FOLDS
    logger.info(f"    Grid: {param_grid}")
    logger.info(f"    Total fits: {n_fits}")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(
        f"    Best CV {SCORING_METRIC}: {grid_search.best_score_:.4f} "
        f"({elapsed:.1f}s)"
    )
    logger.info(f"    Best params: {grid_search.best_params_}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
        elapsed,
    )


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> dict:
    """
    Train all configured models and return results.

    Returns
    -------
    dict
        model_name → {model, best_params, cv_score, time}
    """
    configs = get_model_configs()
    results = {}

    for name, cfg in configs.items():
        best_model, best_params, cv_score, elapsed = train_single_model(
            name, cfg["model"], cfg["params"], X_train, y_train
        )
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "cv_score": cv_score,
            "training_time": elapsed,
        }

    return results


def select_best_model(
    results: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[str, dict]:
    """
    Compare all trained models on the test set and select the best.

    Returns
    -------
    tuple
        (best_model_name, best_result_dict)
    """
    comparison_rows = []

    for name, result in results.items():
        y_pred = result["model"].predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        result["test_metrics"] = metrics
        result["y_pred"] = y_pred

        comparison_rows.append(
            {
                "Model": name,
                "CV R²": f"{result['cv_score']:.4f}",
                "Test R²": f"{metrics['combined_r2']:.4f}",
                "RMSE": f"{metrics['combined_rmse']:.2f}",
                "MAPE": f"{metrics['combined_mape']:.4f}",
                "Time (s)": f"{result['training_time']:.1f}",
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    logger.info(f"\n  Model Comparison:\n{comparison_df.to_string(index=False)}\n")

    # Select best by test R²
    best_name = max(
        results, key=lambda n: results[n]["test_metrics"]["combined_r2"]
    )
    logger.info(f"  ★ Best model: {best_name}")

    return best_name, results[best_name], comparison_df


def save_model_artifacts(
    diameter: int,
    model,
    encoder,
    feature_names: list[str],
    model_name: str,
    metrics: dict,
) -> Path:
    """
    Save trained model, encoder, feature names, and metadata to disk.

    Returns
    -------
    Path
        Directory where artifacts were saved.
    """
    model_dir = MODELS_DIR / f"diameter_{diameter}"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(encoder, model_dir / "encoder.joblib")
    joblib.dump(feature_names, model_dir / "feature_names.joblib")

    # Save metadata
    metadata = {
        "diameter": diameter,
        "algorithm": model_name,
        "metrics": metrics,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }
    joblib.dump(metadata, model_dir / "metadata.joblib")

    logger.info(f"  Saved artifacts to {model_dir}")
    return model_dir


def train_pipeline(diameter: int, df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Full training pipeline for a single diameter.

    Steps:
    1. Prepare features (encode GRADE)
    2. Train/test split
    3. Train all models with GridSearchCV
    4. Compare and select best
    5. Generate evaluation plots
    6. Save artifacts

    Parameters
    ----------
    diameter : int
        Rebar diameter (10, 12, or 16).
    df : pd.DataFrame
        Preprocessed dataframe for this diameter.
    verbose : bool
        If True, print additional details.

    Returns
    -------
    dict
        Summary with best model name, metrics, and comparison table.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training pipeline for diameter {diameter}mm ({len(df)} samples)")
    logger.info(f"{'='*60}")

    # Step 1: Prepare features
    X, y, encoder, feature_names = prepare_features(df, fit=True)

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 3: Train all models
    results = train_all_models(X_train, y_train)

    # Step 4: Compare and select best
    best_name, best_result, comparison_df = select_best_model(
        results, X_test, y_test
    )

    # Step 5: Generate evaluation plots
    generate_evaluation_plots(
        y_test=y_test,
        y_pred=best_result["y_pred"],
        model=best_result["model"],
        feature_names=feature_names,
        diameter=diameter,
        model_name=best_name,
    )

    # Step 6: Save artifacts
    save_model_artifacts(
        diameter=diameter,
        model=best_result["model"],
        encoder=encoder,
        feature_names=feature_names,
        model_name=best_name,
        metrics=best_result["test_metrics"],
    )

    return {
        "diameter": diameter,
        "best_model": best_name,
        "metrics": best_result["test_metrics"],
        "comparison": comparison_df,
    }
