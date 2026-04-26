"""
Data preprocessing pipeline.

Implements the 3-stage domain-aware imputation strategy and
derived feature engineering, consolidating logic from the
original 22May.ipynb notebook.
"""

import logging

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DIAMETER_COL,
    GRADE_COL,
    IMPUTE_BY_DIAMETER_COLS,
    IMPUTE_BY_GRADE_COLS,
    TARGET_COLS,
    UTS_YS_RATIO_COL,
    VALID_DIAMETERS,
    VALID_GRADES,
)

logger = logging.getLogger(__name__)


def impute_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1: Impute chemical composition nulls using within-grade median.

    This preserves metallurgical relationships — different steel grades
    have fundamentally different chemical compositions.

    Columns imputed: CHEM1, CHEM2, CHEM5, CHEM10
    """
    df = df.copy()
    columns = IMPUTE_BY_GRADE_COLS

    for grade in VALID_GRADES:
        mask = df[GRADE_COL] == grade
        grade_df = df.loc[mask, columns]
        medians = grade_df.median()

        for col in columns:
            null_count = df.loc[mask, col].isnull().sum()
            if null_count > 0:
                df.loc[mask, col] = df.loc[mask, col].fillna(medians[col])
                logger.debug(
                    f"Imputed {null_count} nulls in {col} for grade {grade} "
                    f"(median={medians[col]:.4f})"
                )

    logger.info(f"Stage 1 complete: imputed {columns} by GRADE")
    return df


def impute_by_diameter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Impute temperature nulls using within-diameter median.

    Temperature profiles depend on rebar cross-section — larger diameters
    require different heating and cooling parameters.

    Columns imputed: TEMP2, TEMP5
    """
    df = df.copy()
    columns = IMPUTE_BY_DIAMETER_COLS

    for diameter in VALID_DIAMETERS:
        mask = df[DIAMETER_COL] == diameter
        diameter_df = df.loc[mask, columns]
        medians = diameter_df.median()

        for col in columns:
            null_count = df.loc[mask, col].isnull().sum()
            if null_count > 0:
                df.loc[mask, col] = df.loc[mask, col].fillna(medians[col])
                logger.debug(
                    f"Imputed {null_count} nulls in {col} for diameter {diameter} "
                    f"(median={medians[col]:.2f})"
                )

    logger.info(f"Stage 2 complete: imputed {columns} by DIAMETER")
    return df


def impute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3: Impute QUALITY1/QUALITY2 using (GRADE × DIAMETER) group median.

    This is the finest-grained conditioning — mechanical properties depend
    on both the steel grade and the rebar diameter.
    """
    df = df.copy()

    for col in TARGET_COLS:
        group_medians = df.groupby([GRADE_COL, DIAMETER_COL])[col].transform("median")
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(group_medians)
            logger.debug(
                f"Imputed {null_count} nulls in {col} by (GRADE × DIAMETER) median"
            )

    logger.info(f"Stage 3 complete: imputed {TARGET_COLS} by (GRADE × DIAMETER)")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features:
    - uts_ys_ratio: QUALITY1 / QUALITY2 (UTS/YS compliance metric)
    """
    df = df.copy()
    df[UTS_YS_RATIO_COL] = df["QUALITY1"] / df["QUALITY2"]
    logger.info(f"Added derived feature: {UTS_YS_RATIO_COL}")
    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline in order:
    1. Impute chemical features by GRADE
    2. Impute temperature features by DIAMETER
    3. Impute target variables by (GRADE × DIAMETER)
    4. Add derived features (uts_ys_ratio)

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated dataframe.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe with zero nulls (in imputed columns).
    """
    logger.info("Starting preprocessing pipeline...")

    df = impute_by_grade(df)
    df = impute_by_diameter(df)
    df = impute_targets(df)
    df = add_derived_features(df)

    # Verify no nulls remain in critical columns
    critical_cols = IMPUTE_BY_GRADE_COLS + IMPUTE_BY_DIAMETER_COLS + TARGET_COLS
    remaining_nulls = df[critical_cols].isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"⚠ {remaining_nulls} nulls remain after imputation!")
    else:
        logger.info("Preprocessing complete: 0 nulls in critical columns ✓")

    return df


def build_imputation_pipeline() -> Pipeline:
    """
    Build an IterativeImputer pipeline that models feature relationships
    to impute missing values — far superior to median fill.
    
    IterativeImputer uses round-robin regression: it models each feature
    with missing values as a function of all other features, iteratively
    refining estimates. This preserves inter-feature correlations.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("imputer", IterativeImputer(
            max_iter=10,
            random_state=42,
            sample_posterior=False,
            skip_complete=True,
        )),
    ])
    return pipeline
