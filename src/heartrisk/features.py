"""Feature engineering and the reusable preprocessing pipeline.

Pipeline layout (fitted end-to-end, so every learned statistic comes from the
training fold only):

    ClinicalFeatureBuilder  -> adds hr_pct_predicted (optional, tunable)
    ColumnTransformer
      numeric     : median impute -> StandardScaler
      binary      : most-frequent impute (values already 0/1)
      categorical : most-frequent impute -> OneHotEncoder(handle_unknown="ignore")
    classifier
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartrisk.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
)


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds domain-derived features.

    hr_pct_predicted = thalach / (220 - age)
        Peak heart rate as a fraction of the age-predicted maximum heart rate
        (the classic "220 - age" rule). Low chronotropic response during
        exercise is a known marker of cardiac risk, and the ratio removes the
        strong age dependence of raw `thalach`.

    `add_hr_ratio` is exposed as a hyper-parameter so the grid search decides
    (with cross-validation evidence) whether the feature helps.
    """

    def __init__(self, add_hr_ratio: bool = True):
        self.add_hr_ratio = add_hr_ratio

    def fit(self, X, y=None):
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.add_hr_ratio:
            denom = (220.0 - X["age"].astype(float)).clip(lower=1.0)
            X["hr_pct_predicted"] = X["thalach"].astype(float) / denom
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.add_hr_ratio:
            names += ENGINEERED_FEATURES
        return np.asarray(names, dtype=object)


def _exact(names: list[str]) -> str:
    """Regex that matches exactly the given column names (only those present)."""
    return "^(?:" + "|".join(names) + ")$"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            # pattern selector: the engineered column is included only when present
            ("num", numeric, make_column_selector(pattern=_exact(NUMERIC_FEATURES + ENGINEERED_FEATURES))),
            ("bin", binary, make_column_selector(pattern=_exact(BINARY_FEATURES))),
            ("cat", categorical, make_column_selector(pattern=_exact(CATEGORICAL_FEATURES))),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(classifier, add_hr_ratio: bool = True) -> Pipeline:
    return Pipeline([
        ("features", ClinicalFeatureBuilder(add_hr_ratio=add_hr_ratio)),
        ("preprocess", build_preprocessor()),
        ("clf", classifier),
    ])


def transformed_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())
