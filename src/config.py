"""
Central configuration for the Mechanical Property Prediction pipeline.

All constants, column lists, hyperparameter grids, and thresholds
are defined here to avoid magic strings/numbers scattered across modules.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_FILENAME = "INTERN_DATA.csv"

# ─────────────────────────────────────────────
# Column Definitions
# ─────────────────────────────────────────────
# Metadata columns (dropped before training)
META_COLS = ["DATE_TIME", "ID"]

# Product specification
DIAMETER_COL = "DIAMETER"
GRADE_COL = "GRADE"

# Chemical composition features
CHEMICAL_COLS = [
    "CHEM1", "CHEM2", "CHEM3", "CHEM4", "CHEM5",
    "CHEM6", "CHEM7", "CHEM8", "CHEM9", "CHEM10",
]

# Temperature features
TEMP_COLS = ["TEMP1", "TEMP2", "TEMP3", "TEMP4", "TEMP5", "TEMP6"]

# Process parameter features
PROCESS_COLS = ["SPEED", "PROCESS1", "PROCESS2", "PROCESS3"]

# Target variables
TARGET_COLS = ["QUALITY1", "QUALITY2"]

# Derived feature
UTS_YS_RATIO_COL = "uts_ys_ratio"

# Columns to drop before model training
DROP_BEFORE_TRAINING = META_COLS + [DIAMETER_COL, UTS_YS_RATIO_COL]

# Raw data column typo fix
COLUMN_RENAME_MAP = {"PEOCESS3": "PROCESS3"}

# ─────────────────────────────────────────────
# Valid Values
# ─────────────────────────────────────────────
VALID_DIAMETERS = [10, 12, 16]
VALID_GRADES = ["GR1", "GR2", "GR3"]

# ─────────────────────────────────────────────
# Imputation Strategy
# ─────────────────────────────────────────────
# Stage 1: Impute these columns using within-GRADE median
IMPUTE_BY_GRADE_COLS = ["CHEM1", "CHEM2", "CHEM5", "CHEM10"]

# Stage 2: Impute these columns using within-DIAMETER median
IMPUTE_BY_DIAMETER_COLS = ["TEMP2", "TEMP5"]

# Stage 3: QUALITY1/QUALITY2 imputed by (GRADE × DIAMETER) group median

# ─────────────────────────────────────────────
# Quality Gate
# ─────────────────────────────────────────────
UTS_YS_MIN_RATIO = 1.1  # IS 1786 compliance threshold

# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING_METRIC = "r2"

# Hyperparameter grids
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 15, 17, 20],
}

XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1],
}

# ─────────────────────────────────────────────
# Feature Descriptions (for UI display)
# ─────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "CHEM1": "Chemical Composition 1",
    "CHEM2": "Chemical Composition 2",
    "CHEM3": "Chemical Composition 3",
    "CHEM4": "Chemical Composition 4",
    "CHEM5": "Chemical Composition 5",
    "CHEM6": "Chemical Composition 6",
    "CHEM7": "Chemical Composition 7",
    "CHEM8": "Chemical Composition 8",
    "CHEM9": "Chemical Composition 9",
    "CHEM10": "Chemical Composition 10",
    "TEMP1": "Temperature 1",
    "TEMP2": "Temperature 2",
    "TEMP3": "Temperature 3",
    "TEMP4": "Temperature 4",
    "TEMP5": "Temperature 5",
    "TEMP6": "Temperature 6",
    "SPEED": "Rolling Speed",
    "PROCESS1": "Process Parameter 1",
    "PROCESS2": "Process Parameter 2",
    "PROCESS3": "Process Parameter 3",
}

# Typical ranges for UI sliders (min, max, default/median)
FEATURE_RANGES = {
    "CHEM1": (0.16, 0.26, 0.225),
    "CHEM2": (0.69, 1.60, 1.297),
    "CHEM3": (0.006, 0.060, 0.017),
    "CHEM4": (0.012, 0.040, 0.024),
    "CHEM5": (0.14, 0.34, 0.230),
    "CHEM6": (0.001, 0.006, 0.0026),
    "CHEM7": (0.0001, 0.004, 0.0005),
    "CHEM8": (0.015, 0.12, 0.028),
    "CHEM9": (0.001, 0.29, 0.006),
    "CHEM10": (0.0, 160.0, 70.5),
    "TEMP1": (800, 1200, 1000),
    "TEMP2": (700.0, 1200.0, 965.0),
    "TEMP3": (0, 155, 70),
    "TEMP4": (0, 325, 158),
    "SPEED": (11, 32, 21),
    "PROCESS1": (1380, 3580, 2521),
    "PROCESS2": (1460, 3740, 2666),
    "PROCESS3": (0.85, 1.40, 1.32),
    "TEMP5": (500.0, 650.0, 581.0),
    "TEMP6": (500, 640, 549),
}
