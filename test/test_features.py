"""Feature engineering and preprocessing-pipeline tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartrisk.data import split_xy
from heartrisk.features import ClinicalFeatureBuilder, build_pipeline, build_preprocessor, transformed_feature_names


def test_hr_ratio_formula():
    X = pd.DataFrame([{"age": 60, "thalach": 160}])
    X = X.reindex(columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                           "exang", "oldpeak", "slope", "ca", "thal"]).fillna(0)
    X.loc[0, "age"], X.loc[0, "thalach"] = 60, 160
    out = ClinicalFeatureBuilder(add_hr_ratio=True).fit(X).transform(X)
    assert out.loc[0, "hr_pct_predicted"] == pytest.approx(160 / 160)


def test_hr_ratio_can_be_disabled(clean_df):
    X, _ = split_xy(clean_df)
    out = ClinicalFeatureBuilder(add_hr_ratio=False).fit(X).transform(X)
    assert "hr_pct_predicted" not in out


def test_builder_rejects_missing_columns(clean_df):
    X, _ = split_xy(clean_df)
    with pytest.raises(ValueError, match="missing required columns"):
        ClinicalFeatureBuilder().fit(X.drop(columns=["age"]))


def test_preprocessor_outputs_no_nan_and_expected_width(clean_df):
    X, _ = split_xy(clean_df)
    Xb = ClinicalFeatureBuilder().fit_transform(X)
    pre = build_preprocessor().fit(Xb)
    Z = pre.transform(Xb)
    assert not np.isnan(Z).any(), "imputation must remove all missing values"
    names = list(pre.get_feature_names_out())
    # 6 numeric + 1 engineered + 3 binary + one-hot(cp 4, restecg 3, slope<=3, thal<=3)
    assert Z.shape[1] == len(names)
    assert "num__hr_pct_predicted" in names
    assert any(n.startswith("cat__cp_") for n in names)


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(ClinicalFeatureBuilder().fit_transform(X))
    Z = pre.transform(ClinicalFeatureBuilder().fit_transform(X))
    names = list(pre.get_feature_names_out())
    i = names.index("num__chol")
    assert Z[:, i].mean() == pytest.approx(0, abs=1e-9)
    assert Z[:, i].std() == pytest.approx(1, abs=1e-6)


def test_unknown_category_at_inference_does_not_crash(clean_df):
    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["restecg"] = 2 if 2 not in set(X["restecg"]) else row["restecg"]
    row["ca"], row["thal"] = np.nan, np.nan  # missing at inference -> imputed
    proba = pipe.predict_proba(row)
    assert proba.shape == (1, 2) and np.isclose(proba.sum(), 1)


def test_pipeline_fits_preprocessing_on_training_data_only(clean_df):
    """Imputer medians must come from the data passed to fit (no leakage)."""
    X, y = split_xy(clean_df)
    train = X.iloc[:80]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(train, y.iloc[:80])
    num_imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    cols = list(pipe.named_steps["preprocess"].transformers_[0][2])
    expected = ClinicalFeatureBuilder().fit_transform(train)[cols].median().values
    assert np.allclose(num_imputer.statistics_, expected)
    assert len(transformed_feature_names(pipe)) > 13
