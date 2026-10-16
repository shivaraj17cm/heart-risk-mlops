"""Data acquisition, validation and cleaning tests."""

import pandas as pd
import pytest

from heartrisk.data import DataValidationError, clean, invalid_value_mask, load_raw, split_xy, validate_schema
from heartrisk.schema import FEATURES, RAW_COLUMNS, TARGET


def test_load_raw_parses_question_marks_as_missing(raw_df):
    assert list(raw_df.columns) == RAW_COLUMNS
    assert raw_df["ca"].isna().sum() == 2
    assert raw_df["thal"].isna().sum() == 2
    assert all(pd.api.types.is_numeric_dtype(raw_df[c]) for c in RAW_COLUMNS)


def test_load_raw_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw(tmp_path / "nope.data")


def test_clean_binarises_target(raw_df):
    df, report = clean(raw_df)
    assert TARGET in df and "num" not in df
    assert set(df[TARGET].unique()) <= {0, 1}
    # every raw num > 0 must map to 1
    dedup = raw_df.drop_duplicates().reset_index(drop=True)
    assert df[TARGET].sum() == (dedup["num"] > 0).sum()
    assert sum(report.target_counts.values()) == len(df)


def test_clean_removes_exact_duplicates(raw_df):
    df, report = clean(raw_df)
    assert report.duplicates_removed == 2
    assert not df.duplicated().any()
    assert report.rows_clean == report.rows_raw - 2


def test_clean_keeps_missing_for_pipeline_imputation(raw_df):
    df, report = clean(raw_df)
    assert report.missing_by_column == {"ca": 2, "thal": 2}
    assert df[["ca", "thal"]].isna().sum().sum() == 4  # not imputed here -> no leakage


def test_clean_drops_out_of_range_rows(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "chol"] = 5000        # impossible cholesterol
    bad.loc[1, "cp"] = 9             # undefined chest-pain code
    assert invalid_value_mask(bad).sum() == 2
    df, report = clean(bad)
    assert report.invalid_rows_removed == 2


def test_validate_schema_rejects_wrong_columns(raw_df):
    with pytest.raises(DataValidationError, match="Unexpected columns"):
        validate_schema(raw_df.drop(columns=["chol"]))


def test_validate_schema_rejects_non_numeric(raw_df):
    bad = raw_df.copy()
    bad["age"] = bad["age"].astype(object)
    bad.loc[0, "age"] = "sixty"
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_schema(bad)


def test_validate_schema_rejects_missing_target(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "num"] = None
    with pytest.raises(DataValidationError, match="Target"):
        validate_schema(bad)


def test_validate_schema_rejects_empty():
    with pytest.raises(DataValidationError, match="empty"):
        validate_schema(pd.DataFrame(columns=RAW_COLUMNS, dtype=float))


def test_split_xy_shapes_and_order(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(clean_df)
    assert y.dtype.kind == "i"
