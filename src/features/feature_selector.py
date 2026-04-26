"""
Feature selection and pruning.

Implements RFECV (Recursive Feature Elimination with Cross-Validation) and
VIF (Variance Inflation Factor) analysis for multicollinearity detection.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def select_features_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    min_features: int = 10,
) -> tuple:
    """
    Automatically select the optimal feature subset using RFECV.

    Uses a lightweight Random Forest as the evaluator to rank features
    by importance, recursively dropping the least important and
    evaluating with 5-fold CV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.DataFrame
        Training targets.
    min_features : int
        Minimum features to retain.

    Returns
    -------
    tuple
        (selector, selected_feature_names, ranking_dict)
    """
    logger.info(f"Running RFECV (min_features={min_features})...")

    estimator = RandomForestRegressor(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring="r2",
        min_features_to_select=min_features,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)

    selected = X_train.columns[selector.support_].tolist()
    eliminated = [c for c in X_train.columns if c not in selected]

    ranking = dict(zip(X_train.columns, selector.ranking_))

    logger.info(
        f"RFECV: selected {len(selected)}/{len(X_train.columns)} features "
        f"(optimal CV R² = {selector.cv_results_['mean_test_score'].max():.4f})"
    )
    if eliminated:
        logger.info(f"  Eliminated: {eliminated}")

    return selector, selected, ranking


def compute_vif(X: pd.DataFrame) -> dict[str, float]:
    """
    Compute Variance Inflation Factor for each feature.

    VIF measures how much a feature is explained by other features.
    VIF > 10 indicates problematic multicollinearity — the feature
    is redundant and may add noise rather than signal.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric only).

    Returns
    -------
    dict[str, float]
        Feature name → VIF value.
    """
    vif_data = {}
    X_numeric = X.select_dtypes(include=[np.number]).dropna()

    for col in X_numeric.columns:
        others = [c for c in X_numeric.columns if c != col]
        if len(others) == 0:
            vif_data[col] = 1.0
            continue

        r2 = LinearRegression().fit(
            X_numeric[others], X_numeric[col]
        ).score(X_numeric[others], X_numeric[col])

        vif_data[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")

    # Log high-VIF features
    high_vif = {k: v for k, v in vif_data.items() if v > 10}
    if high_vif:
        logger.warning(
            f"High VIF features (multicollinear): "
            f"{', '.join(f'{k}={v:.1f}' for k, v in sorted(high_vif.items(), key=lambda x: -x[1]))}"
        )
    else:
        logger.info("VIF check passed: no severe multicollinearity detected")

    return vif_data


def prune_high_vif_features(
    X: pd.DataFrame,
    vif_threshold: float = 10.0,
    protected: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Iteratively remove the highest-VIF feature until all VIFs < threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    vif_threshold : float
        Maximum acceptable VIF.
    protected : list[str], optional
        Features that should never be removed (e.g., encoded GRADE columns).

    Returns
    -------
    tuple
        (pruned_X, removed_feature_names)
    """
    protected = protected or []
    X = X.copy()
    removed = []

    while True:
        vif = compute_vif(X)
        # Find worst VIF among non-protected features
        candidates = {
            k: v for k, v in vif.items()
            if v > vif_threshold and k not in protected
        }

        if not candidates:
            break

        worst = max(candidates, key=candidates.get)
        logger.info(f"  Removing {worst} (VIF={candidates[worst]:.1f})")
        X = X.drop(columns=[worst])
        removed.append(worst)

    if removed:
        logger.info(f"VIF pruning removed {len(removed)} features: {removed}")
    return X, removed
