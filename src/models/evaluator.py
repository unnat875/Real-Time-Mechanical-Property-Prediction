"""
Model evaluation and visualization.

Computes metrics (R², MAPE, RMSE), generates actual-vs-predicted plots,
residual distributions with SPC sigma bands, and feature importance charts.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.config import OUTPUTS_DIR, TARGET_COLS

logger = logging.getLogger(__name__)

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for multi-output predictions.

    Returns
    -------
    dict
        Per-target and combined metrics:
        - quality1_r2, quality1_rmse, quality1_mape
        - quality2_r2, quality2_rmse, quality2_mape
        - combined_r2, combined_rmse, combined_mape
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    metrics = {}

    for i, target in enumerate(TARGET_COLS):
        yt = y_true[:, i] if y_true.ndim > 1 else y_true
        yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred

        metrics[f"{target.lower()}_r2"] = r2_score(yt, yp)
        metrics[f"{target.lower()}_rmse"] = np.sqrt(mean_squared_error(yt, yp))
        metrics[f"{target.lower()}_mape"] = mean_absolute_percentage_error(yt, yp)

    # Combined metrics
    metrics["combined_r2"] = r2_score(y_true, y_pred)
    metrics["combined_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["combined_mape"] = mean_absolute_percentage_error(y_true, y_pred)

    return metrics


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    diameter: int,
    save_dir: Path | None = None,
) -> None:
    """Plot overlay of actual vs predicted for both targets."""
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, target in enumerate(TARGET_COLS):
        ax = axes[i]
        yt = y_test[:, i]
        yp = y_pred[:, i]

        ax.scatter(yt, yp, alpha=0.5, s=20, edgecolors="none")
        min_val = min(yt.min(), yp.min())
        max_val = max(yt.max(), yp.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            "r--", linewidth=1.5, label="Perfect prediction"
        )
        ax.set_xlabel(f"Actual {target}", fontsize=12)
        ax.set_ylabel(f"Predicted {target}", fontsize=12)
        ax.set_title(
            f"{target} — Diameter {diameter}mm\n"
            f"R² = {r2_score(yt, yp):.4f}",
            fontsize=13, fontweight="bold",
        )
        ax.legend()

    plt.suptitle(
        f"Actual vs Predicted — Diameter {diameter}mm",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"actual_vs_predicted_{diameter}mm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {path}")

    plt.close(fig)


def plot_residual_distribution(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    diameter: int,
    save_dir: Path | None = None,
) -> None:
    """Plot residual histograms with ±1/2/3σ SPC bands."""
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#4C72B0", "#DD8452"]

    for i, target in enumerate(TARGET_COLS):
        ax = axes[i]
        residuals = y_test[:, i] - y_pred[:, i]
        mean_r = residuals.mean()
        std_r = residuals.std()

        ax.hist(
            residuals, bins=30, alpha=0.7, color=colors[i],
            edgecolor="white", label="Residuals", density=True,
        )
        sns.kdeplot(residuals, ax=ax, color="black", linewidth=1.5)

        # Sigma bands
        for n_sigma, ls, lw in [(1, "--", 1.2), (2, ":", 1.0), (3, "-.", 0.8)]:
            ax.axvline(mean_r + n_sigma * std_r, color="red", linestyle=ls, linewidth=lw)
            ax.axvline(mean_r - n_sigma * std_r, color="red", linestyle=ls, linewidth=lw)
            if n_sigma == 1:
                ax.axvline(
                    mean_r + n_sigma * std_r, color="red", linestyle=ls,
                    linewidth=lw, label=f"±{n_sigma}σ",
                )

        ax.axvline(mean_r, color="darkred", linewidth=1.5, label="Mean")
        ax.set_xlabel("Residual", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"{target} Residuals — Diameter {diameter}mm\n"
            f"μ = {mean_r:.2f}, σ = {std_r:.2f}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9)

    plt.suptitle(
        f"Residual Distribution — Diameter {diameter}mm",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"residuals_{diameter}mm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {path}")

    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    diameter: int,
    model_name: str,
    save_dir: Path | None = None,
    top_n: int = 15,
) -> None:
    """Plot horizontal bar chart of feature importances."""
    # Try to extract feature importances from wrapped models
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # MultiOutputRegressor — check first sub-estimator
        sub = model.estimators_[0]
        if hasattr(sub, "feature_importances_"):
            importances = sub.feature_importances_
        elif hasattr(sub, "estimators_"):
            # StackingRegressor — average importances from base learners
            base_importances = []
            for est in sub.estimators_:
                if hasattr(est, "feature_importances_"):
                    base_importances.append(est.feature_importances_)
            if base_importances:
                importances = np.mean(base_importances, axis=0)

    if importances is None:
        logger.warning(f"  Model {model_name} has no extractable feature importances")
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df))),
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name}\n"
        f"Diameter {diameter}mm",
        fontsize=14, fontweight="bold",
    )

    # Add value labels
    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"feature_importance_{diameter}mm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {path}")

    plt.close(fig)


def generate_evaluation_plots(
    y_test,
    y_pred: np.ndarray,
    model,
    feature_names: list[str],
    diameter: int,
    model_name: str,
) -> None:
    """Generate all evaluation plots for a trained model."""
    logger.info(f"  Generating evaluation plots for diameter {diameter}mm...")

    plot_actual_vs_predicted(
        y_test, y_pred, diameter,
        save_dir=OUTPUTS_DIR / "model_comparison",
    )

    plot_residual_distribution(
        y_test, y_pred, diameter,
        save_dir=OUTPUTS_DIR / "residuals",
    )

    plot_feature_importance(
        model, feature_names, diameter, model_name,
        save_dir=OUTPUTS_DIR / "feature_importance",
    )
