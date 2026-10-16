"""Training, evaluation, experiment tracking and packaging tests."""

import json

import joblib
import mlflow
import numpy as np
import pytest

from heartrisk.data import split_xy
from heartrisk.evaluate import compute_metrics
from heartrisk.model_io import load_model
from heartrisk.schema import FEATURES
from heartrisk.train import make_estimator, select_best


def test_compute_metrics_known_values():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.4, 0.9]          # one FP, one FN
    m = compute_metrics(y, p, threshold=0.5)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)


def test_make_estimator_injects_seed():
    rf = make_estimator("sklearn.ensemble.RandomForestClassifier", {"n_estimators": 5}, seed=123)
    assert rf.random_state == 123 and rf.n_estimators == 5


def test_selection_rule_prefers_simpler_model_within_tolerance():
    rows = [{"model": "random_forest", "cv_roc_auc_mean": 0.905},
            {"model": "logistic_regression", "cv_roc_auc_mean": 0.900}]
    name, reason = select_best(rows, "roc_auc", 0.01, ["logistic_regression", "random_forest"])
    assert name == "logistic_regression" and "tie tolerance" in reason


def test_selection_rule_takes_clear_winner():
    rows = [{"model": "random_forest", "cv_roc_auc_mean": 0.95},
            {"model": "logistic_regression", "cv_roc_auc_mean": 0.90}]
    name, _ = select_best(rows, "roc_auc", 0.01, ["logistic_regression", "random_forest"])
    assert name == "random_forest"


def test_training_produces_two_models_and_reports(trained):
    out = trained["out"]
    assert {m["model"] for m in out["models"]} == {"logistic_regression", "random_forest"}
    for m in out["models"]:
        for k in ("cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean", "cv_roc_auc_mean",
                  "test_accuracy", "test_precision", "test_recall", "test_roc_auc"):
            assert 0.0 <= m[k] <= 1.0
    assert (trained["reports_dir"] / "model_comparison.csv").exists()
    assert (trained["reports_dir"] / "metrics.json").exists()


def test_packaged_model_roundtrip(trained, clean_df):
    X, _ = split_xy(clean_df)
    pipe = joblib.load(trained["model_dir"] / "model.joblib")
    loaded = load_model(trained["model_dir"])
    a = pipe.predict_proba(X)[:, 1]
    b = np.array([r["probability_disease"] for r in loaded.predict_records(X.to_dict("records"))])
    assert np.allclose(a, b, atol=1e-4)


def test_metadata_is_complete(trained):
    meta = json.loads((trained["model_dir"] / "model_metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert meta["model_family"] in {"logistic_regression", "random_forest"}
    for key in ("model_version", "mlflow_run_id", "best_params", "cv_metrics", "test_metrics", "data_sha256"):
        assert key in meta


def test_prediction_output_structure(trained, sample_payload):
    res = load_model(trained["model_dir"]).predict_records([sample_payload])[0]
    assert set(res) == {"prediction", "label", "probability_disease", "confidence"}
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1.0
    expected_conf = res["probability_disease"] if res["prediction"] else 1 - res["probability_disease"]
    assert res["confidence"] == pytest.approx(expected_conf, abs=1e-4)


def test_missing_optional_features_are_imputed(trained, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k not in ("ca", "thal")}
    res = load_model(trained["model_dir"]).predict_records([payload])[0]
    assert res["prediction"] in (0, 1)


def test_model_load_fails_clearly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run training first"):
        load_model(tmp_path)


def test_mlflow_tracked_all_runs(trained):
    mlflow.set_tracking_uri(trained["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=["pytest-heart"])
    parents = runs[runs["tags.run_level"] == "parent"]
    children = runs[runs["tags.run_level"] == "child"]
    assert set(parents["tags.model_family"]) == {"logistic_regression", "random_forest"}
    assert len(children) == 4 + 2          # every grid combination is a run
    assert (runs["tags.run_level"] == "selection").sum() == 1
    for col in ("metrics.cv_roc_auc_mean", "metrics.test_roc_auc", "metrics.test_precision",
                "metrics.test_recall", "metrics.test_accuracy"):
        assert parents[col].notna().all()
    # logged values must equal what training returned
    lr = parents[parents["tags.model_family"] == "logistic_regression"].iloc[0]
    lr_out = next(m for m in trained["out"]["models"] if m["model"] == "logistic_regression")
    assert lr["metrics.test_roc_auc"] == pytest.approx(lr_out["test_roc_auc"])
    client = mlflow.MlflowClient()
    arts = [a.path for a in client.list_artifacts(lr["run_id"], "plots")]
    assert any(a.endswith("roc_curve.png") for a in arts)
    assert any(a.endswith("confusion_matrix.png") for a in arts)
    assert client.get_model_version_by_alias("pytest-heart-model", "champion") is not None
