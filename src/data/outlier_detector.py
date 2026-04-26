"""
Automated outlier detection and handling.

Implements IQR clipping for univariate outliers and Isolation Forest
for multivariate anomalies. We clip rather than drop — in manufacturing,
extreme values often represent real edge cases the model should handle.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import CHEMICAL_COLS, TEMP_COLS, PROCESS_COLS

logger = logging.getLogger(__name__)


def clip_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Clip extreme values to IQR bounds (Winsorization).

    Values beyond [Q1 - factor*IQR, Q3 + factor*IQR] are clamped to
    the boundary rather than removed. This preserves sample count while
    reducing the influence of outliers on model training.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str], optional
        Columns to clip. Defaults to all chemical + temperature + process cols.
    factor : float
        IQR multiplier. 1.5 = standard, 3.0 = extreme only.

    Returns
    -------
    pd.DataFrame
        Dataframe with clipped values.
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in CHEMICAL_COLS + TEMP_COLS + PROCESS_COLS if c in df.columns]

    total_clipped = 0
    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        clipped = df[col].clip(lower, upper)
        n_clipped = (df[col] != clipped).sum()
        if n_clipped > 0:
            logger.debug(
                f"  Clipped {n_clipped} values in {col} "
                f"to [{lower:.4f}, {upper:.4f}]"
            )
            total_clipped += n_clipped

        df[col] = clipped

    logger.info(f"IQR clipping: {total_clipped} values clipped across {len(columns)} columns")
    return df


def flag_multivariate_outliers(
    X: pd.DataFrame | np.ndarray,
    contamination: float = 0.05,
) -> np.ndarray:
    """
    Identify multivariate outliers using Isolation Forest.

    Catches combinations of anomalous features that univariate IQR misses
    (e.g., normal chemistry values but abnormal temperature combinations).

    Parameters
    ----------
    X : array-like
        Feature matrix (numeric only).
    contamination : float
        Expected proportion of outliers (0.05 = 5%).

    Returns
    -------
    np.ndarray
        Boolean mask — True = inlier, False = outlier.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X)
    inlier_mask = labels == 1

    n_outliers = (~inlier_mask).sum()
    logger.info(
        f"Isolation Forest: flagged {n_outliers} multivariate outliers "
        f"({n_outliers / len(X) * 100:.1f}%)"
    )
    return inlier_mask


def remove_multivariate_outliers(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.03,
) -> pd.DataFrame:
    """
    Remove rows flagged as multivariate outliers.

    Uses a conservative contamination rate (3%) to only remove
    truly anomalous samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Columns to consider for outlier detection.
    contamination : float
        Expected proportion of outliers.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with outliers removed.
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].select_dtypes(include=[np.number])

    # Drop rows with NaN before IsolationForest
    valid_mask = X.notna().all(axis=1)
    X_valid = X[valid_mask]

    if len(X_valid) == 0:
        logger.warning("No valid rows for outlier detection")
        return df

    inlier_mask_valid = flag_multivariate_outliers(X_valid, contamination)

    # Map back to original index
    full_inlier_mask = pd.Series(True, index=df.index)
    full_inlier_mask.loc[X_valid.index] = inlier_mask_valid

    n_removed = (~full_inlier_mask).sum()
    result = df[full_inlier_mask].reset_index(drop=True)
    logger.info(f"Removed {n_removed} outlier rows → {len(result)} remaining")
    return result
