"""Loading, validation and cleaning of the raw UCI Cleveland file.

Design note: cleaning here is limited to *deterministic, row-local* operations
(parsing '?', type casting, target binarisation, removing exact duplicates,
rejecting impossible values). Anything *learned from data* (imputation medians,
scaling statistics, one-hot vocabularies) lives inside the scikit-learn
Pipeline so it is fitted on training folds only - this prevents data leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from heartrisk import paths
from heartrisk.schema import (
    ALLOWED_CODES,
    FEATURES,
    NUMERIC_RANGES,
    RAW_COLUMNS,
    RAW_TARGET,
    TARGET,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the raw data violates the expected schema."""


@dataclass
class CleaningReport:
    rows_raw: int = 0
    rows_clean: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    missing_by_column: dict = field(default_factory=dict)
    rows_with_missing: int = 0
    target_counts: dict = field(default_factory=dict)
    raw_num_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_raw(path: Path | str = paths.RAW_FILE) -> pd.DataFrame:
    """Read the header-less UCI file, treating '?' as missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Structural checks: column set, numeric dtypes, non-empty, target present."""
    if list(df.columns) != RAW_COLUMNS:
        raise DataValidationError(
            f"Unexpected columns {list(df.columns)}; expected {RAW_COLUMNS}"
        )
    if df.empty:
        raise DataValidationError("Dataset is empty")
    non_numeric = [c for c in RAW_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric values found in columns: {non_numeric}")
    if df[RAW_TARGET].isna().any():
        raise DataValidationError("Target column 'num' contains missing values")


def invalid_value_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with a value outside the documented codes / plausible ranges.

    Missing values are *not* invalid here; they are imputed later in the pipeline.
    """
    bad = pd.Series(False, index=df.index)
    for col, allowed in ALLOWED_CODES.items():
        if col in df:
            present = df[col].notna()
            bad |= present & ~df[col].isin(allowed)
    for col, (lo, hi) in NUMERIC_RANGES.items():
        present = df[col].notna()
        bad |= present & ((df[col] < lo) | (df[col] > hi))
    return bad


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and clean the raw frame; return the clean frame and a report."""
    validate_schema(df)
    report = CleaningReport(rows_raw=len(df))
    report.raw_num_counts = {int(k): int(v) for k, v in df[RAW_TARGET].value_counts().sort_index().items()}

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report.duplicates_removed = before - len(df)

    bad = invalid_value_mask(df)
    report.invalid_rows_removed = int(bad.sum())
    if report.invalid_rows_removed:
        logger.warning("Removing %d rows with out-of-range values", report.invalid_rows_removed)
    df = df.loc[~bad].reset_index(drop=True)

    report.missing_by_column = {c: int(n) for c, n in df[FEATURES].isna().sum().items() if n}
    report.rows_with_missing = int(df[FEATURES].isna().any(axis=1).sum())

    # Binary target: 0 = absence, 1..4 = presence (UCI convention for binary studies)
    df[TARGET] = (df[RAW_TARGET] > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET])

    # Integer-coded columns stored as nullable Int64 so missing stays missing
    for col in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "slope", "ca", "thal"]:
        df[col] = df[col].round().astype("Int64")
    df["oldpeak"] = df["oldpeak"].astype(float)

    report.rows_clean = len(df)
    report.target_counts = {int(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    return df, report


def load_clean(path: Path | str = paths.CLEAN_FILE) -> pd.DataFrame:
    """Load the cleaned CSV produced by `python -m heartrisk.data`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m heartrisk.data` first.")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    return X, y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and clean the raw UCI file.")
    parser.add_argument("--raw", default=str(paths.RAW_FILE))
    parser.add_argument("--out", default=str(paths.CLEAN_FILE))
    parser.add_argument("--report", default=str(paths.REPORTS_DIR / "cleaning_report.json"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = load_raw(args.raw)
    clean_df, report = clean(raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out, index=False)
    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report.to_dict(), indent=2))

    logger.info("Raw rows: %d | clean rows: %d", report.rows_raw, report.rows_clean)
    logger.info("Duplicates removed: %d | invalid rows removed: %d",
                report.duplicates_removed, report.invalid_rows_removed)
    logger.info("Missing values by column: %s (rows affected: %d)",
                report.missing_by_column, report.rows_with_missing)
    logger.info("Raw 'num' distribution: %s", report.raw_num_counts)
    logger.info("Binary target distribution: %s", report.target_counts)
    logger.info("Clean data written to %s", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DataValidationError", "CleaningReport", "load_raw", "validate_schema",
    "invalid_value_mask", "clean", "load_clean", "split_xy",
]
