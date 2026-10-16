"""Shared fixtures.

Tests never touch the network or the real dataset: a deterministic synthetic
file in the *exact* UCI raw format (no header, 14 columns, '?' for missing)
is generated, so CI is fast and hermetic. The real-data pipeline is exercised
separately by the CI `train` job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from heartrisk.data import clean, load_raw


def make_raw_lines(n: int = 160, seed: int = 7, n_missing: int = 4, n_dupes: int = 2) -> list[str]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        sick = rng.random() < 0.45
        age = int(rng.integers(35, 75))
        row = [
            float(age),
            float(rng.integers(0, 2)),
            float(4 if sick and rng.random() < 0.7 else rng.integers(1, 4)),
            float(rng.integers(100, 180)),
            float(rng.integers(150, 380)),
            float(rng.random() < 0.15),
            float(rng.integers(0, 3)),
            float(rng.integers(95, 150) if sick else rng.integers(140, 195)),
            float(1 if sick and rng.random() < 0.6 else 0),
            round(float(rng.uniform(0.8, 4.0) if sick else rng.uniform(0, 1.5)), 1),
            float(2 if sick else rng.integers(1, 3)),
            float(rng.integers(1, 4) if sick else 0),
            float(7 if sick and rng.random() < 0.7 else 3),
            int(rng.integers(1, 5)) if sick else 0,
        ]
        rows.append(row)
    lines = [",".join(str(v) for v in r) for r in rows]
    # inject '?' into ca (index 11) and thal (index 12) like the real file
    for i in range(n_missing):
        parts = lines[i * 7].split(",")
        parts[11 if i % 2 == 0 else 12] = "?"
        lines[i * 7] = ",".join(parts)
    lines += lines[-n_dupes:]  # exact duplicates
    return lines


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("raw") / "processed.cleveland.data"
    p.write_text("\n".join(make_raw_lines()) + "\n")
    return p


@pytest.fixture(scope="session")
def raw_df(raw_file) -> pd.DataFrame:
    return load_raw(raw_file)


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    df, _ = clean(raw_df)
    return df


@pytest.fixture(scope="session")
def clean_file(clean_df, tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("processed") / "heart_clean.csv"
    clean_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def small_config(tmp_path_factory) -> Path:
    """Tiny grids so the full training path (incl. MLflow) runs in seconds."""
    cfg = {
        "seed": 42, "test_size": 0.25, "cv_folds": 3, "refit_metric": "roc_auc",
        "decision_threshold": 0.5,
        "mlflow": {"experiment_name": "pytest-heart", "registered_model_name": "pytest-heart-model"},
        "selection": {"metric": "roc_auc", "tie_tolerance": 0.01,
                      "simplicity_order": ["logistic_regression", "random_forest"]},
        "models": {
            "logistic_regression": {
                "estimator": "sklearn.linear_model.LogisticRegression",
                "fixed_params": {"solver": "liblinear", "max_iter": 500},
                "param_grid": {"clf__C": [0.1, 1.0], "features__add_hr_ratio": [True, False]}},
            "random_forest": {
                "estimator": "sklearn.ensemble.RandomForestClassifier",
                "fixed_params": {"n_estimators": 30, "n_jobs": 1},
                "param_grid": {"clf__max_depth": [3, None]}},
        },
    }
    p = tmp_path_factory.mktemp("cfg") / "train.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture(scope="session")
def trained(tmp_path_factory, clean_file, small_config) -> dict:
    from heartrisk.train import run

    base = tmp_path_factory.mktemp("train")
    model_dir, reports_dir = base / "model", base / "reports"
    out = run(config_path=small_config, data_path=clean_file, model_dir=model_dir,
              reports_dir=reports_dir, tracking_uri=f"sqlite:///{(base / 'mlflow.db').as_posix()}",
              register=True)
    return {"out": out, "model_dir": model_dir, "reports_dir": reports_dir,
            "tracking_uri": f"sqlite:///{(base / 'mlflow.db').as_posix()}"}


@pytest.fixture
def sample_payload() -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "samples" / "sample_request.json").read_text())
