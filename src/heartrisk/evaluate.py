"""Metric computation and evaluation plots.

Positive class = 1 ("heart disease present"). ROC-AUC always uses the
probability of class 1 (column index found from `classes_`, never assumed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in CI, Docker and remote labs
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heartrisk import plotting  # noqa: E402
from heartrisk.model_io import positive_proba  # noqa: E402,F401  (re-exported)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_roc(y_true, y_proba, title: str, out: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plotting.new_figure(5.2, 4.6)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=plotting.MUTED, label="Chance (AUC 0.50)")
    ax.plot(fpr, tpr, linewidth=2, color=plotting.SERIES[0], label=f"Model (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    return plotting.save(fig, out)


def plot_confusion(metrics: dict, title: str, out: Path) -> Path:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plotting.new_figure(4.6, 4.2)
    ax.imshow(cm, cmap=plotting.sequential_cmap())
    labels = ["No disease (0)", "Disease (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    vmax = cm.max() if cm.max() else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > 0.55 * vmax else plotting.TEXT)
    ax.set_title(title)
    ax.grid(False)
    return plotting.save(fig, out)


def plot_feature_importance(names: list[str], values: np.ndarray, title: str,
                            xlabel: str, out: Path, top: int = 15) -> Path:
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plotting.new_figure(6.4, 0.32 * len(order) + 1.4)
    colors = [plotting.SERIES[1] if values[i] > 0 else plotting.SERIES[0] for i in order]
    ax.barh([names[i] for i in order], values[order], color=colors, height=0.6)
    ax.axvline(0, color=plotting.MUTED, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return plotting.save(fig, out)


def plot_model_comparison(rows: list[dict], out: Path) -> Path:
    """Grouped bars: CV mean (with std error bars) for each metric and model."""
    metrics = ["accuracy", "precision", "recall", "roc_auc"]
    models = [r["model"] for r in rows]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    fig, ax = plotting.new_figure(7.2, 4.4)
    for k, r in enumerate(rows):
        means = [r[f"cv_{m}_mean"] for m in metrics]
        stds = [r[f"cv_{m}_std"] for m in metrics]
        ax.bar(x + (k - (len(models) - 1) / 2) * width, means, width * 0.92, yerr=stds,
               capsize=3, color=plotting.SERIES[k % len(plotting.SERIES)], label=r["model"],
               error_kw={"elinewidth": 1, "ecolor": plotting.TEXT})
    ax.set_xticks(x, [m.replace("_", "-").upper() if m == "roc_auc" else m.title() for m in metrics])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel(f"{rows[0].get('cv_folds', 5)}-fold CV score (mean ± std)")
    ax.set_title("Model comparison - cross-validation on training split")
    ax.legend(frameon=False, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return plotting.save(fig, out)
