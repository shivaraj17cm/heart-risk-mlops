"""Train, tune, compare and package the heart-disease classifiers, tracked with MLflow.

Run:  python -m heartrisk.train            (uses configs/train.yaml)

MLflow layout per invocation
----------------------------
experiment  heart-disease-uci
  run  <family>                 (parent, one per model family)
     run  <family>-grid-NN      (child, one per hyper-parameter combination: CV metrics)
     -> best params, CV mean/std, hold-out test metrics, ROC / confusion / importance
        plots, fitted pipeline logged as an MLflow model
  run  model-selection          (comparison table + chart, selected model, registry)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heartrisk import __version__, paths
from heartrisk.data import load_clean, split_xy
from heartrisk.evaluate import (
    compute_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc,
    positive_proba,
)
from heartrisk.features import build_pipeline, transformed_feature_names
from heartrisk.schema import FEATURES, TARGET_LABELS

logger = logging.getLogger("heartrisk.train")

# zero_division=0: a heavily regularised configuration (e.g. L1 with C=0.01) can predict
# a single class on a fold; precision is then defined as 0 instead of raising warnings.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


# --------------------------------------------------------------------------- helpers
def load_config(path: Path | str = paths.CONFIG_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_estimator(dotted: str, fixed_params: dict, seed: int):
    module, cls = dotted.rsplit(".", 1)
    klass = getattr(importlib.import_module(module), cls)
    params = dict(fixed_params or {})
    if "random_state" in klass().get_params():
        params["random_state"] = seed
    return klass(**params)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - git is optional (e.g. inside Docker)
        return "unknown"


def serve_requirements() -> list[str]:
    req = paths.PROJECT_ROOT / "requirements-serve.txt"
    if not req.exists():
        return [f"scikit-learn=={sklearn.__version__}"]
    return [ln.split("#")[0].strip() for ln in req.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _clean_value(v):
    return "None" if v is None else v


def select_best(rows: list[dict], metric: str, tolerance: float, simplicity_order: list[str]) -> tuple[str, str]:
    """Apply the documented selection rule. Returns (model_name, human-readable reason)."""
    key = f"cv_{metric}_mean"
    best_score = max(r[key] for r in rows)
    contenders = [r for r in rows if best_score - r[key] <= tolerance]
    rank = {name: i for i, name in enumerate(simplicity_order)}
    chosen = sorted(contenders, key=lambda r: (rank.get(r["model"], 99), -r[key]))[0]
    top = max(rows, key=lambda r: r[key])
    if chosen["model"] == top["model"]:
        reason = (f"{chosen['model']} has the highest mean CV {metric} "
                  f"({chosen[key]:.4f}).")
    else:
        reason = (f"{top['model']} has the highest mean CV {metric} ({top[key]:.4f}), but "
                  f"{chosen['model']} ({chosen[key]:.4f}) is within the {tolerance} tie tolerance "
                  f"and is simpler, so it is preferred.")
    return chosen["model"], reason


def feature_importance(pipe) -> tuple[list[str], np.ndarray, str] | None:
    names = transformed_feature_names(pipe)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return names, clf.coef_.ravel(), "Coefficient (standardised features; + raises risk)"
    if hasattr(clf, "feature_importances_"):
        return names, clf.feature_importances_, "Impurity-based importance"
    return None


# --------------------------------------------------------------------------- training
def train_family(name: str, spec: dict, cfg: dict, X_train, y_train, X_test, y_test,
                 cv, common_tags: dict, out_dir: Path) -> dict:
    seed = cfg["seed"]
    estimator = make_estimator(spec["estimator"], spec.get("fixed_params"), seed)
    pipe = build_pipeline(estimator)
    grid = {k: list(v) for k, v in spec["param_grid"].items()}

    search = GridSearchCV(pipe, grid, scoring=SCORING, refit=cfg["refit_metric"], cv=cv,
                          n_jobs=-1, return_train_score=True, error_score="raise")

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({**common_tags, "model_family": name, "run_level": "parent"})
        mlflow.log_params({
            "model_family": name,
            "estimator": spec["estimator"],
            "cv_folds": cfg["cv_folds"],
            "refit_metric": cfg["refit_metric"],
            "test_size": cfg["test_size"],
            "seed": seed,
            "grid_size": int(np.prod([len(v) for v in grid.values()])),
            "decision_threshold": cfg["decision_threshold"],
            "preprocessing": "median-impute+scale(num) | mode-impute(bin) | mode-impute+onehot(cat)",
            **{f"fixed__{k}": v for k, v in (spec.get("fixed_params") or {}).items()},
        })

        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        res = search.cv_results_

        # one child run per hyper-parameter combination -> every tried config is tracked
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-grid-{i:02d}", nested=True):
                mlflow.set_tags({**common_tags, "model_family": name, "run_level": "child"})
                mlflow.log_params({k: _clean_value(v) for k, v in params.items()})
                child_metrics = {}
                for m in SCORING:
                    child_metrics[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
                    child_metrics[f"cv_{m}_std"] = float(res[f"std_test_{m}"][i])
                child_metrics["cv_train_roc_auc_mean"] = float(res["mean_train_roc_auc"][i])
                child_metrics["rank_by_refit_metric"] = int(res[f"rank_test_{cfg['refit_metric']}"][i])
                mlflow.log_metrics(child_metrics)

        best = search.best_estimator_
        bi = search.best_index_
        cv_summary = {}
        for m in SCORING:
            cv_summary[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][bi])
            cv_summary[f"cv_{m}_std"] = float(res[f"std_test_{m}"][bi])
        cv_summary["cv_train_roc_auc_mean"] = float(res["mean_train_roc_auc"][bi])
        cv_summary["overfit_gap_roc_auc"] = cv_summary["cv_train_roc_auc_mean"] - cv_summary["cv_roc_auc_mean"]

        # per-fold scores of the best configuration (for the report)
        fold_scores = [float(res[f"split{k}_test_roc_auc"][bi]) for k in range(cfg["cv_folds"])]

        y_proba = positive_proba(best, X_test)
        test = compute_metrics(y_test, y_proba, cfg["decision_threshold"])

        mlflow.log_params({f"best__{k}": _clean_value(v) for k, v in search.best_params_.items()})
        mlflow.log_metrics({**cv_summary, "fit_seconds": fit_seconds,
                            **{f"test_{k}": v for k, v in test.items()}})
        for k, s in enumerate(fold_scores):
            mlflow.log_metric("cv_best_fold_roc_auc", s, step=k)

        fam_dir = out_dir / name
        fam_dir.mkdir(parents=True, exist_ok=True)
        figs = [
            plot_roc(y_test, y_proba, f"ROC curve (hold-out test) - {name}", fam_dir / "roc_curve.png"),
            plot_confusion(test, f"Confusion matrix (hold-out test) - {name}", fam_dir / "confusion_matrix.png"),
        ]
        fi = feature_importance(best)
        if fi:
            figs.append(plot_feature_importance(fi[0], fi[1], f"Top features - {name}", fi[2],
                                                fam_dir / "feature_importance.png"))
        pd.DataFrame(res).drop(columns=["params"]).to_csv(fam_dir / "cv_results.csv", index=False)
        (fam_dir / "summary.json").write_text(json.dumps({
            "best_params": {k: _clean_value(v) for k, v in search.best_params_.items()},
            "cv": cv_summary, "cv_best_fold_roc_auc": fold_scores, "test": test}, indent=2))
        for f in figs:
            mlflow.log_artifact(str(f), artifact_path="plots")
        mlflow.log_artifact(str(fam_dir / "cv_results.csv"), artifact_path="cv")
        mlflow.log_artifact(str(fam_dir / "summary.json"))

        signature = infer_signature(X_train.head(5), best.predict_proba(X_train.head(5)))
        # code_paths bundles the heartrisk package (custom transformer) so the MLflow
        # model is self-contained; pip requirements mirror requirements-serve.txt
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_train.head(2),
                                 code_paths=[str(paths.PROJECT_ROOT / "src" / "heartrisk")],
                                 pip_requirements=serve_requirements())

        logger.info("%-20s CV ROC-AUC %.4f ± %.4f | test ROC-AUC %.4f | best %s",
                    name, cv_summary["cv_roc_auc_mean"], cv_summary["cv_roc_auc_std"],
                    test["roc_auc"], search.best_params_)

        return {
            "model": name,
            "run_id": parent.info.run_id,
            "best_params": {k: _clean_value(v) for k, v in search.best_params_.items()},
            "cv_folds": cfg["cv_folds"],
            **cv_summary,
            **{f"test_{k}": v for k, v in test.items()},
            "estimator": best,
        }


def run(config_path: Path | str = paths.CONFIG_FILE, data_path: Path | str = paths.CLEAN_FILE,
        model_dir: Path | str = paths.MODEL_DIR, reports_dir: Path | str = paths.REPORTS_DIR,
        tracking_uri: str | None = None, register: bool = True) -> dict:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    np.random.seed(seed)
    model_dir, reports_dir = Path(model_dir), Path(reports_dir)

    if tracking_uri is None:
        tracking_uri = f"sqlite:///{(paths.PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=(paths.PROJECT_ROOT / "mlruns").as_uri()
                                 if tracking_uri.startswith("sqlite") else None)
    mlflow.set_experiment(exp_name)

    data_path = Path(data_path)
    df = load_clean(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    common_tags = {
        "project": "heartrisk",
        "package_version": __version__,
        "git_commit": git_commit(),
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "n_rows": str(len(df)),
        "n_train": str(len(X_train)),
        "n_test": str(len(X_test)),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "host": platform.node(),
    }
    logger.info("Data %s: %d rows (train %d / test %d), positive rate %.3f",
                data_path.name, len(df), len(X_train), len(X_test), y.mean())

    fig_dir = reports_dir / "figures" / "models"
    rows = []
    for name, spec in cfg["models"].items():
        rows.append(train_family(name, spec, cfg, X_train, y_train, X_test, y_test, cv,
                                 common_tags, fig_dir))

    sel_cfg = cfg["selection"]
    chosen_name, reason = select_best(rows, sel_cfg["metric"], sel_cfg["tie_tolerance"],
                                      sel_cfg["simplicity_order"])
    chosen = next(r for r in rows if r["model"] == chosen_name)
    logger.info("Selected model: %s - %s", chosen_name, reason)

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("estimator",)} for r in rows])
    table["best_params"] = table["best_params"].apply(json.dumps)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    comp_fig = plot_model_comparison(rows, fig_dir / "model_comparison.png")

    # ---------------------------------------------------------------- packaging
    model_dir.mkdir(parents=True, exist_ok=True)
    best_pipe = chosen["estimator"]
    joblib.dump(best_pipe, model_dir / paths.MODEL_FILE_NAME)
    metadata = {
        "model_name": cfg["mlflow"]["registered_model_name"],
        "model_family": chosen_name,
        "model_version": f"{__version__}+{common_tags['git_commit']}",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_reason": reason,
        "mlflow_run_id": chosen["run_id"],
        "best_params": chosen["best_params"],
        "decision_threshold": cfg["decision_threshold"],
        "features": FEATURES,
        "target_labels": {str(k): v for k, v in TARGET_LABELS.items()},
        "cv_metrics": {k: chosen[k] for k in chosen if k.startswith("cv_")},
        "test_metrics": {k: chosen[k] for k in chosen if k.startswith("test_")},
        "data_sha256": common_tags["data_sha256"],
        "python": common_tags["python"],
        "sklearn": sklearn.__version__,
    }
    (model_dir / paths.METADATA_FILE_NAME).write_text(json.dumps(metadata, indent=2))

    # Round-trip check: the packaged artifact must reproduce in-memory predictions exactly
    reloaded = joblib.load(model_dir / paths.MODEL_FILE_NAME)
    if not np.allclose(positive_proba(reloaded, X_test), positive_proba(best_pipe, X_test)):
        raise RuntimeError("Reloaded model predictions differ from the trained model")

    metrics_out = {
        "selected_model": chosen_name,
        "selection_reason": reason,
        "models": [{k: v for k, v in r.items() if k != "estimator"} for r in rows],
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    with mlflow.start_run(run_name="model-selection"):
        mlflow.set_tags({**common_tags, "run_level": "selection", "selected_model": chosen_name})
        mlflow.log_params({"selection_metric": sel_cfg["metric"],
                           "tie_tolerance": sel_cfg["tie_tolerance"],
                           "simplicity_order": ",".join(sel_cfg["simplicity_order"]),
                           "selected_model": chosen_name,
                           "selected_run_id": chosen["run_id"]})
        mlflow.log_metrics({f"{r['model']}__cv_roc_auc_mean": r["cv_roc_auc_mean"] for r in rows})
        mlflow.log_metrics({f"selected__{k}": v for k, v in metadata["test_metrics"].items()})
        mlflow.log_text(reason, "selection_reason.txt")
        mlflow.log_artifact(str(reports_dir / "model_comparison.csv"))
        mlflow.log_artifact(str(comp_fig), artifact_path="plots")
        mlflow.log_artifact(str(model_dir / paths.METADATA_FILE_NAME))
        mlflow.log_artifact(str(paths.CONFIG_FILE if Path(config_path) == paths.CONFIG_FILE
                                else config_path), artifact_path="config")

    if register:
        model_uri = f"runs:/{chosen['run_id']}/model"
        reg_name = cfg["mlflow"]["registered_model_name"]
        mv = mlflow.register_model(model_uri, reg_name)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(reg_name, "champion", mv.version)
        client.set_model_version_tag(reg_name, mv.version, "model_family", chosen_name)
        logger.info("Registered %s version %s with alias 'champion'", reg_name, mv.version)

    logger.info("Packaged model -> %s", model_dir / paths.MODEL_FILE_NAME)
    return metrics_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and track heart-disease models.")
    parser.add_argument("--config", default=str(paths.CONFIG_FILE))
    parser.add_argument("--data", default=str(paths.CLEAN_FILE))
    parser.add_argument("--model-dir", default=str(paths.MODEL_DIR))
    parser.add_argument("--reports-dir", default=str(paths.REPORTS_DIR))
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: sqlite:///mlflow.db in the repo root)")
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registry")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run(args.config, args.data, args.model_dir, args.reports_dir, args.tracking_uri,
              register=not args.no_register)
    print(json.dumps({"selected_model": out["selected_model"],
                      "selection_reason": out["selection_reason"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run", "select_best", "load_config", "make_estimator"]
