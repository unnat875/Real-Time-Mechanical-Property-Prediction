#!/usr/bin/env python3
"""
Training CLI — Entry point for the model training pipeline.

Usage:
    # Train all diameter models
    python train.py

    # Train with custom data path
    python train.py --data data/raw/INTERN_DATA.csv

    # Train single diameter
    python train.py --diameter 12

    # Verbose output
    python train.py --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_RAW_DIR, RAW_DATA_FILENAME, VALID_DIAMETERS
from src.data.loader import load_raw_data, split_by_diameter, validate_data
from src.data.preprocessor import preprocess_pipeline
from src.models.trainer import train_pipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner() -> None:
    """Print a nice startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     Real-Time Mechanical Property Prediction — Training     ║
║     Steel Rebar UTS & YS Prediction Pipeline                ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(results: list[dict]) -> None:
    """Print final training summary table."""
    print("\n")
    print("=" * 72)
    print("  TRAINING SUMMARY")
    print("=" * 72)
    print(
        f"  {'Diameter':<12} {'Best Model':<16} {'R² (combined)':<16} "
        f"{'RMSE':<10} {'MAPE':<10}"
    )
    print("  " + "-" * 64)

    for r in results:
        m = r["metrics"]
        print(
            f"  {r['diameter']}mm{'':<8} "
            f"{r['best_model']:<16} "
            f"{m['combined_r2']:<16.4f} "
            f"{m['combined_rmse']:<10.2f} "
            f"{m['combined_mape']:<10.4f}"
        )

    print("=" * 72)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train mechanical property prediction models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_RAW_DIR / RAW_DATA_FILENAME),
        help="Path to raw CSV data file.",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        choices=VALID_DIAMETERS,
        default=None,
        help="Train only for a specific diameter (10, 12, or 16).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    print_banner()

    start_total = time.time()

    # ── Step 1: Load & validate ──────────────────────────────
    logger.info("Step 1/4: Loading raw data...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = load_raw_data(data_path)
    df = validate_data(df)

    # ── Step 2: Preprocess ───────────────────────────────────
    logger.info("Step 2/4: Preprocessing...")
    df = preprocess_pipeline(df)

    # ── Step 3: Split by diameter ────────────────────────────
    logger.info("Step 3/4: Splitting by diameter...")
    diameter_dfs = split_by_diameter(df)

    # ── Step 4: Train models ─────────────────────────────────
    logger.info("Step 4/4: Training models...")

    diameters_to_train = (
        [args.diameter] if args.diameter else VALID_DIAMETERS
    )

    results = []
    for diameter in diameters_to_train:
        result = train_pipeline(
            diameter=diameter,
            df=diameter_dfs[diameter],
            verbose=args.verbose,
        )
        results.append(result)

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start_total
    print_summary(results)
    logger.info(f"Total training time: {elapsed:.1f}s")
    logger.info("All models saved to models/ directory ✓")


if __name__ == "__main__":
    main()
