"""
Feature encoding and preparation for model training / inference.

Handles OneHotEncoding of the GRADE column and assembles the
final numeric feature matrix.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    DROP_BEFORE_TRAINING,
    GRADE_COL,
    TARGET_COLS,
)

logger = logging.getLogger(__name__)


def prepare_features(
    df: pd.DataFrame,
    fit: bool = True,
    encoder: Optional[OneHotEncoder] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list[str]]:
    """
    Prepare feature matrix X and target matrix y from a preprocessed dataframe.

    Steps:
    1. Separate targets (QUALITY1, QUALITY2)
    2. Drop metadata/non-feature columns (DATE_TIME, ID, DIAMETER, uts_ys_ratio)
    3. Extract numeric columns
    4. OneHot-encode the GRADE column
    5. Combine into final feature matrix

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (output of preprocess_pipeline).
    fit : bool
        If True, fit a new OneHotEncoder. If False, use the provided encoder.
    encoder : OneHotEncoder, optional
        Pre-fitted encoder (required when fit=False).

    Returns
    -------
    X : pd.DataFrame
        Final numeric feature matrix.
    y : pd.DataFrame
        Target variable matrix (QUALITY1, QUALITY2).
    encoder : OneHotEncoder
        Fitted encoder (for saving / reuse).
    feature_names : list[str]
        Ordered list of feature column names.
    """
    # Separate targets
    y = df[TARGET_COLS].copy()

    # Start with all columns, drop targets and metadata
    cols_to_drop = [c for c in DROP_BEFORE_TRAINING + TARGET_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Save grade column before dropping non-numeric
    grade_series = X[GRADE_COL].copy() if GRADE_COL in X.columns else None

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # OneHot-encode GRADE
    if grade_series is not None:
        grade_values = grade_series.values.reshape(-1, 1)

        if fit:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(grade_values)
        else:
            if encoder is None:
                raise ValueError("Must provide a fitted encoder when fit=False")
            encoded = encoder.transform(grade_values)

        grade_labels = encoder.get_feature_names_out(["GRADE"]).tolist()
        encoded_df = pd.DataFrame(
            encoded, columns=grade_labels, index=X.index
        )
        X = pd.concat([X, encoded_df], axis=1)

    feature_names = X.columns.tolist()
    logger.info(
        f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features "
        f"(including {len(grade_labels) if grade_series is not None else 0} "
        f"encoded grade columns)"
    )

    return X, y, encoder, feature_names


def prepare_inference_input(
    record: dict,
    encoder: OneHotEncoder,
    feature_names: list[str],
) -> np.ndarray:
    """
    Transform a single input record (dict) into a model-ready feature vector.

    Parameters
    ----------
    record : dict
        Input features, e.g. {"GRADE": "GR1", "CHEM1": 0.22, "TEMP1": 950, ...}
    encoder : OneHotEncoder
        Fitted encoder for GRADE encoding.
    feature_names : list[str]
        Ordered list of feature names (as used during training).

    Returns
    -------
    np.ndarray
        Feature vector of shape (1, n_features).
    """
    # Encode GRADE
    grade = record.pop(GRADE_COL, None) or record.pop("GRADE", None)
    if grade is None:
        raise ValueError("Input must include a 'GRADE' field")

    grade_encoded = encoder.transform([[grade]])
    grade_labels = encoder.get_feature_names_out(["GRADE"]).tolist()

    # Build feature dict with encoded grade
    feature_dict = {}
    for col in feature_names:
        if col in grade_labels:
            idx = grade_labels.index(col)
            feature_dict[col] = grade_encoded[0][idx]
        elif col in record:
            feature_dict[col] = record[col]
        else:
            # Leave missing features as NaN for the IterativeImputer
            feature_dict[col] = np.nan

    # Build array in correct order
    feature_vector = np.array(
        [[feature_dict[col] for col in feature_names]]
    )

    return feature_vector
