"""
Data loading and validation utilities.

Handles reading raw CSV data, applying column fixes, validating
data integrity, and splitting by diameter.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    COLUMN_RENAME_MAP,
    DATA_RAW_DIR,
    DIAMETER_COL,
    RAW_DATA_FILENAME,
    VALID_DIAMETERS,
    VALID_GRADES,
    GRADE_COL,
    CHEMICAL_COLS,
    TARGET_COLS,
)

logger = logging.getLogger(__name__)


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the raw manufacturing dataset from CSV.

    Applies column renames (e.g., PEOCESS3 → PROCESS3) automatically.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file.  Defaults to DATA_RAW_DIR / RAW_DATA_FILENAME.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with corrected column names.
    """
    if path is None:
        path = DATA_RAW_DIR / RAW_DATA_FILENAME

    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)

    # Fix column name typos
    df = df.rename(columns=COLUMN_RENAME_MAP)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the raw dataframe for expected schema and value ranges.

    Raises
    ------
    ValueError
        If any validation check fails.

    Returns
    -------
    pd.DataFrame
        The same dataframe (unchanged) if all checks pass.
    """
    # Check required columns exist
    required_cols = (
        [DIAMETER_COL, GRADE_COL]
        + CHEMICAL_COLS
        + TARGET_COLS
    )
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check diameter values
    invalid_diameters = set(df[DIAMETER_COL].unique()) - set(VALID_DIAMETERS)
    if invalid_diameters:
        raise ValueError(
            f"Unexpected diameter values: {invalid_diameters}. "
            f"Expected: {VALID_DIAMETERS}"
        )

    # Check grade values
    invalid_grades = set(df[GRADE_COL].unique()) - set(VALID_GRADES)
    if invalid_grades:
        raise ValueError(
            f"Unexpected grade values: {invalid_grades}. "
            f"Expected: {VALID_GRADES}"
        )

    # Check no negative chemical compositions
    for col in CHEMICAL_COLS:
        if (df[col].dropna() < 0).any():
            raise ValueError(f"Negative values found in {col}")

    # Check target variables are positive where present
    for col in TARGET_COLS:
        if (df[col].dropna() <= 0).any():
            raise ValueError(f"Non-positive values found in target {col}")

    logger.info("Data validation passed ✓")
    return df


def split_by_diameter(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Split a dataframe into sub-dataframes by DIAMETER value.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a DIAMETER column.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping of diameter → subset dataframe, e.g. {10: df_10, 12: df_12, 16: df_16}.
    """
    result = {}
    for diameter in VALID_DIAMETERS:
        subset = df[df[DIAMETER_COL] == diameter].copy().reset_index(drop=True)
        result[diameter] = subset
        logger.info(f"Diameter {diameter}mm: {len(subset)} rows")

    total = sum(len(v) for v in result.values())
    if total != len(df):
        logger.warning(
            f"Split total ({total}) ≠ original ({len(df)}). "
            f"{len(df) - total} rows had unrecognized diameters."
        )

    return result
