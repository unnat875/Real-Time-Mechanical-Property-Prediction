"""
Model training module (v2 — Optuna + Stacking).

Trains and compares multiple regression algorithms using Bayesian
hyperparameter optimization (Optuna), builds stacking ensembles,
and saves artifacts.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import (
    CV_FOLDS,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data.outlier_detector import clip_outliers_iqr, remove_multivariate_outliers
from src.data.preprocessor import build_imputation_pipeline
from src.features.encoder import prepare_features
from src.models.evaluator import compute_metrics, generate_evaluation_plots

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Optuna Objectives ───────────────────────────────────────


def _rf_objective(trial, X_train, y_train):
    """Random Forest objective for Optuna."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.8, None]
        ),
    }
    model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    return scores.mean()


def _xgb_objective(trial, X_train, y_train):
    """XGBoost objective for Optuna."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = MultiOutputRegressor(
        XGBRegressor(**params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    )
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    return scores.mean()


def _lgbm_objective(trial, X_train, y_train):
    """LightGBM objective for Optuna."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = MultiOutputRegressor(
        LGBMRegressor(**params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    )
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    return scores.mean()


# ─── Optuna Optimization ─────────────────────────────────────

OBJECTIVES = {
    "RandomForest": _rf_objective,
    "XGBoost": _xgb_objective,
    "LightGBM": _lgbm_objective,
}


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_trials: int = 80,
) -> tuple:
    """
    Optimize a single model with Optuna Bayesian search.

    Returns
    -------
    tuple
        (best_model, best_params, best_cv_score, training_time)
    """
    logger.info(f"  Optimizing {model_name} ({n_trials} Optuna trials)...")

    objective_fn = OBJECTIVES[model_name]
    study = optuna.create_study(direction="maximize")

    start = time.time()
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - start

    best_params = study.best_params

    # Rebuild best model
    if model_name == "RandomForest":
        best_model = RandomForestRegressor(
            **best_params, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "XGBoost":
        best_model = MultiOutputRegressor(
            XGBRegressor(**best_params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        )
    elif model_name == "LightGBM":
        best_model = MultiOutputRegressor(
            LGBMRegressor(**best_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
        )

    best_model.fit(X_train, y_train)

    logger.info(
        f"    Best CV R²: {study.best_value:.4f} ({elapsed:.1f}s)"
    )
    logger.info(f"    Best params: {best_params}")

    return best_model, best_params, study.best_value, elapsed


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_trials: int = 80,
) -> dict:
    """
    Train all configured models with Optuna and return results.
    """
    results = {}
    for name in OBJECTIVES:
        best_model, best_params, cv_score, elapsed = optimize_model(
            name, X_train, y_train, n_trials=n_trials
        )
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "cv_score": cv_score,
            "training_time": elapsed,
        }
    return results


# ─── Stacking Ensemble ───────────────────────────────────────


def build_stacking_model(
    rf_params: dict,
    xgb_params: dict,
    lgbm_params: dict,
) -> MultiOutputRegressor:
    """
    Build a Stacking Regressor with diverse base learners.

    Architecture:
        Layer 0 (Base):  RF  |  XGBoost  |  LightGBM
        Layer 1 (Meta):        Ridge

    Diversity rationale:
    - RF: Bagging (parallel trees, low variance)
    - XGBoost: Boosting (sequential trees, low bias)
    - LightGBM: Leaf-wise growth (different inductive bias)
    - Ridge meta: Linear combination prevents overfitting the stack
    """
    base_learners = [
        ("rf", RandomForestRegressor(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb", XGBRegressor(**xgb_params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)),
        ("lgbm", LGBMRegressor(**lgbm_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)),
    ]

    stacking = StackingRegressor(
        estimators=base_learners,
        final_estimator=Ridge(alpha=1.0),
        cv=CV_FOLDS,
        n_jobs=-1,
        passthrough=False,
    )

    # Wrap for multi-output prediction (UTS + YS)
    return MultiOutputRegressor(stacking)


def train_stacking(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    individual_results: dict,
) -> dict:
    """
    Train a stacking ensemble using individually tuned base models.
    """
    logger.info("  Training Stacking Ensemble...")
    start = time.time()

    rf_params = individual_results["RandomForest"]["best_params"]
    xgb_params = individual_results["XGBoost"]["best_params"]
    lgbm_params = individual_results["LightGBM"]["best_params"]

    stacking_model = build_stacking_model(rf_params, xgb_params, lgbm_params)
    stacking_model.fit(X_train, y_train)

    # CV score for stacking
    scores = cross_val_score(
        build_stacking_model(rf_params, xgb_params, lgbm_params),
        X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    cv_score = scores.mean()
    elapsed = time.time() - start

    logger.info(f"    Stacking CV R²: {cv_score:.4f} ({elapsed:.1f}s)")

    return {
        "model": stacking_model,
        "best_params": {
            "rf": rf_params,
            "xgb": xgb_params,
            "lgbm": lgbm_params,
            "meta": "Ridge(alpha=1.0)",
        },
        "cv_score": cv_score,
        "training_time": elapsed,
    }


# ─── Selection & Comparison ──────────────────────────────────


def select_best_model(
    results: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[str, dict, pd.DataFrame]:
    """
    Compare all trained models on the test set and select the best.
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
    imputer,
    feature_names: list[str],
    model_name: str,
    metrics: dict,
) -> Path:
    """Save trained model, encoder, imputer, feature names, and metadata to disk."""
    model_dir = MODELS_DIR / f"diameter_{diameter}"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(encoder, model_dir / "encoder.joblib")
    joblib.dump(imputer, model_dir / "imputer.joblib")
    joblib.dump(feature_names, model_dir / "feature_names.joblib")

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


# ─── Full Pipeline ────────────────────────────────────────────


def train_pipeline(
    diameter: int,
    df: pd.DataFrame,
    n_trials: int = 80,
    verbose: bool = False,
) -> dict:
    """
    Full training pipeline for a single diameter.

    Steps:
    1. Clip outliers (IQR)
    2. Remove multivariate outliers (Isolation Forest)
    3. Prepare features (encode GRADE)
    4. Train/test split
    5. Optimize all individual models with Optuna
    6. Build and evaluate stacking ensemble
    7. Compare all models and select best
    8. Generate evaluation plots
    9. Save artifacts
    """
    from src.models.diagnostics import detect_overfitting, plot_learning_curves

    logger.info(f"\n{'='*60}")
    logger.info(f"Training pipeline for diameter {diameter}mm ({len(df)} samples)")
    logger.info(f"{'='*60}")

    # Step 1-2: Data quality
    logger.info("  Step 1: Outlier handling...")
    df = clip_outliers_iqr(df)
    feature_cols = [c for c in df.columns if c not in ["DATE_TIME", "ID", "GRADE", "DIAMETER", "QUALITY1", "QUALITY2", "uts_ys_ratio"]]
    df = remove_multivariate_outliers(df, feature_cols, contamination=0.03)

    # Step 3: Prepare features
    X, y, encoder, feature_names = prepare_features(df, fit=True)

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Fit Imputer for robust inference later
    logger.info("  Fitting IterativeImputer for runtime robustness...")
    imputer = build_imputation_pipeline()
    imputer.fit(X_train)

    # Step 5: Optimize individual models
    individual_results = train_all_models(X_train, y_train, n_trials=n_trials)

    # Step 6: Build stacking ensemble
    stacking_result = train_stacking(X_train, y_train, individual_results)
    all_results = {**individual_results, "Stacking": stacking_result}

    # Step 7: Compare and select best
    best_name, best_result, comparison_df = select_best_model(
        all_results, X_test, y_test
    )

    # Step 8: Diagnostics
    overfitting_report = detect_overfitting(
        best_result["model"], X_train, y_train, X_test, y_test
    )
    logger.info(
        f"  Overfitting check: {overfitting_report['diagnosis']} "
        f"(gap = {overfitting_report['gap']:.4f})"
    )

    # Generate evaluation plots
    generate_evaluation_plots(
        y_test=y_test,
        y_pred=best_result["y_pred"],
        model=best_result["model"],
        feature_names=feature_names,
        diameter=diameter,
        model_name=best_name,
    )

    # Learning curves
    plot_learning_curves(
        best_result["model"], X, y, diameter,
        save_path=OUTPUTS_DIR / "diagnostics" / f"learning_curve_{diameter}mm.png",
    )

    # Step 9: Save artifacts
    save_model_artifacts(
        diameter=diameter,
        model=best_result["model"],
        encoder=encoder,
        imputer=imputer,
        feature_names=feature_names,
        model_name=best_name,
        metrics=best_result["test_metrics"],
    )

    return {
        "diameter": diameter,
        "best_model": best_name,
        "metrics": best_result["test_metrics"],
        "comparison": comparison_df,
        "overfitting": overfitting_report,
    }
