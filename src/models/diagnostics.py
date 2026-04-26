"""
Model diagnostics — learning curves, overfitting detection, CV analysis.

Provides tools to diagnose bias vs. variance tradeoffs and quantify
model generalization reliability.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    RepeatedKFold,
    cross_val_score,
    learning_curve,
)

from src.config import OUTPUTS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")


def detect_overfitting(
    model,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict:
    """
    Compare train score vs. test score to detect overfitting.

    A gap > 0.1 signals the model is memorizing training data
    rather than learning generalizable patterns.

    Returns
    -------
    dict
        Diagnosis with train_r2, test_r2, gap, diagnosis, recommendation.
    """
    y_train_arr = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
    y_test_arr = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

    train_r2 = r2_score(y_train_arr, model.predict(X_train))
    test_r2 = r2_score(y_test_arr, model.predict(X_test))
    gap = train_r2 - test_r2

    if gap > 0.2:
        diagnosis = "SEVERE OVERFITTING"
        recommendation = "Increase regularization, reduce max_depth, add more data"
    elif gap > 0.1:
        diagnosis = "MODERATE OVERFITTING"
        recommendation = "Consider early stopping, reduce model complexity"
    elif gap > 0.05:
        diagnosis = "MILD OVERFITTING"
        recommendation = "Consider feature pruning or light regularization"
    else:
        diagnosis = "HEALTHY"
        recommendation = "Model generalization looks good"

    return {
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "gap": round(gap, 4),
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def rigorous_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict:
    """
    Repeated K-Fold CV for stable performance estimates with confidence.

    n_repeats=3 with different random splits detects if performance
    is sensitive to the particular train/test split.

    Returns
    -------
    dict
        mean_r2, std_r2, ci_95, overfitting_risk.
    """
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(model, X, y, cv=rkf, scoring="r2", n_jobs=-1)

    return {
        "mean_r2": round(float(scores.mean()), 4),
        "std_r2": round(float(scores.std()), 4),
        "ci_95": (
            round(float(scores.mean() - 1.96 * scores.std()), 4),
            round(float(scores.mean() + 1.96 * scores.std()), 4),
        ),
        "all_scores": scores.tolist(),
        "overfitting_risk": (
            "HIGH" if scores.std() > 0.05
            else "MODERATE" if scores.std() > 0.03
            else "LOW"
        ),
    }


def plot_learning_curves(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    diameter: int,
    save_path: Path | None = None,
) -> None:
    """
    Generate learning curves to diagnose bias vs. variance.

    Interpretation:
    - Train & val converge HIGH → ideal (low bias, low variance)
    - Train HIGH, val LOW, big gap → overfitting (high variance)
    - Both converge LOW → underfitting (high bias, need better features)
    """
    logger.info(f"  Generating learning curves for diameter {diameter}mm...")

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="r2",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Confidence bands
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#3a7bd5",
    )
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#e94560",
    )

    # Score lines
    ax.plot(
        train_sizes, train_mean, "o-",
        color="#3a7bd5", label="Training Score", linewidth=2,
    )
    ax.plot(
        train_sizes, val_mean, "o-",
        color="#e94560", label="Validation Score", linewidth=2,
    )

    # Annotate final gap
    gap = train_mean[-1] - val_mean[-1]
    color = "red" if gap > 0.1 else "orange" if gap > 0.05 else "green"
    label = "Overfitting!" if gap > 0.1 else "Watch" if gap > 0.05 else "OK"
    ax.annotate(
        f"Gap = {gap:.3f}\n({label})",
        xy=(train_sizes[-1], val_mean[-1]),
        xytext=(-80, 30),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color=color,
        arrowprops=dict(arrowstyle="->", color=color),
    )

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title(
        f"Learning Curve — Diameter {diameter}mm",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {save_path}")
    plt.close(fig)
