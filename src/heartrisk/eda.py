"""Exploratory data analysis: figures + a JSON summary of the statistics quoted in the report.

Run:  python -m heartrisk.eda
The notebook notebooks/01_eda.ipynb calls the same functions and adds commentary.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from heartrisk import paths, plotting
from heartrisk.data import load_clean, load_raw
from heartrisk.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger("heartrisk.eda")

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
PRETTY = {
    "age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)", "oldpeak": "ST depression (oldpeak)",
    "ca": "Major vessels (ca)", "sex": "Sex (1 = male)", "fbs": "Fasting sugar > 120",
    "exang": "Exercise angina", "cp": "Chest pain type", "restecg": "Resting ECG",
    "slope": "ST slope", "thal": "Thalassemia", TARGET: "Target",
}
CLASS_NAMES = {0: "No disease (0)", 1: "Disease (1)"}


def plot_class_balance(df: pd.DataFrame, out: Path) -> Path:
    counts = df[TARGET].value_counts().sort_index()
    total = counts.sum()
    fig, ax = plotting.new_figure(5.0, 3.8)
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, width=0.55,
                  color=[plotting.CLASS_COLORS[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01, f"{v}  ({v / total:.1%})",
                ha="center", va="bottom", fontsize=9, color=plotting.TEXT)
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(f"Class balance (n = {total})")
    ax.grid(axis="x", visible=False)
    return plotting.save(fig, out)


def plot_raw_num_distribution(raw: pd.DataFrame, out: Path) -> Path:
    counts = raw["num"].value_counts().sort_index()
    fig, ax = plotting.new_figure(5.0, 3.6)
    colors = [plotting.CLASS_COLORS[0] if k == 0 else plotting.CLASS_COLORS[1] for k in counts.index]
    ax.bar([str(int(k)) for k in counts.index], counts.values, color=colors, width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=9, color=plotting.TEXT)
    ax.set_xlabel("Original UCI diagnosis code 'num' (0 = absent, 1-4 = present)")
    ax.set_ylabel("Patients")
    ax.set_title("Raw target before binarisation")
    ax.grid(axis="x", visible=False)
    return plotting.save(fig, out)


def plot_histograms(df: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), dpi=110)
    for ax, col in zip(axes.ravel(), CONTINUOUS):
        data = df[[col, TARGET]].dropna()
        bins = np.histogram_bin_edges(data[col], bins=18)
        for cls in (0, 1):
            vals = data.loc[data[TARGET] == cls, col]
            ax.hist(vals, bins=bins, alpha=0.35, color=plotting.CLASS_COLORS[cls], label=CLASS_NAMES[cls])
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.6, color=plotting.CLASS_COLORS[cls])
        ax.set_title(PRETTY[col])
        ax.set_ylabel("Patients")
    last = axes.ravel()[-1]
    data = df[["ca", TARGET]].dropna()
    ct = pd.crosstab(data["ca"], data[TARGET])
    x = np.arange(len(ct.index))
    for k, cls in enumerate((0, 1)):
        last.bar(x + (k - 0.5) * 0.38, ct.get(cls, 0), 0.36, color=plotting.CLASS_COLORS[cls])
    last.set_xticks(x, [str(int(v)) for v in ct.index])
    last.set_title(PRETTY["ca"])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Distribution of numeric features by target class", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_categorical(df: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    cols = CATEGORICAL_FEATURES + BINARY_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0), dpi=110)
    for ax, col in zip(axes.ravel(), cols):
        data = df[[col, TARGET]].dropna()
        rate = data.groupby(col)[TARGET].agg(["mean", "size"])
        x = np.arange(len(rate))
        ax.bar(x, rate["mean"], color=plotting.SERIES[1], width=0.6)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=7, color=plotting.TEXT_SECONDARY)
        ax.set_xticks(x, [str(int(v)) for v in rate.index])
        ax.set_ylim(0, 1.1)
        ax.set_title(PRETTY[col])
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share with disease")
    fig.suptitle("Disease rate by category (bar height) with group size", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_correlation(df: pd.DataFrame, out: Path) -> Path:
    corr = df[FEATURES + [TARGET]].astype(float).corr(method="pearson")
    fig, ax = plotting.new_figure(8.4, 7.0)
    im = ax.imshow(corr.values, cmap=plotting.diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else plotting.TEXT)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.set_title("Correlation heatmap (features and binary target)")
    return plotting.save(fig, out)


def plot_missing(raw: pd.DataFrame, out: Path) -> Path:
    miss = raw[FEATURES].isna().sum()
    fig, ax = plotting.new_figure(6.0, 3.4)
    ax.bar(miss.index, miss.values, color=[plotting.SERIES[1] if v else plotting.MUTED for v in miss.values])
    for i, v in enumerate(miss.values):
        if v:
            ax.text(i, v + 0.1, str(v), ha="center", fontsize=9, color=plotting.TEXT)
    ax.set_ylabel("Missing values ('?')")
    ax.set_title("Missing values per column in the raw file")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    return plotting.save(fig, out)


def summarise(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    corr = df[FEATURES + [TARGET]].astype(float).corr()[TARGET].drop(TARGET)
    return {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "n_features": len(FEATURES),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "missing_raw": {k: int(v) for k, v in raw[FEATURES].isna().sum().items() if v},
        "raw_num_counts": {str(int(k)): int(v) for k, v in raw["num"].value_counts().sort_index().items()},
        "target_counts": {str(int(k)): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "positive_rate": round(float(df[TARGET].mean()), 4),
        "numeric_describe": df[NUMERIC_FEATURES].astype(float).describe().round(2).to_dict(),
        "numeric_mean_by_class": df.groupby(TARGET)[NUMERIC_FEATURES].mean().round(2).to_dict(),
        "corr_with_target_sorted": {k: round(float(v), 3)
                                    for k, v in corr.reindex(corr.abs().sort_values(ascending=False).index).items()},
        "disease_rate_by_category": {
            c: {str(int(k)): round(float(v), 3) for k, v in df.groupby(c)[TARGET].mean().items()}
            for c in CATEGORICAL_FEATURES + BINARY_FEATURES},
    }


def run(raw_path=paths.RAW_FILE, clean_path=paths.CLEAN_FILE, fig_dir=paths.FIGURES_DIR / "eda",
        summary_path=paths.REPORTS_DIR / "eda_summary.json") -> dict:
    raw = load_raw(raw_path)
    df = load_clean(clean_path)
    fig_dir = Path(fig_dir)
    outputs = [
        plot_class_balance(df, fig_dir / "class_balance.png"),
        plot_raw_num_distribution(raw, fig_dir / "raw_target_num.png"),
        plot_histograms(df, fig_dir / "numeric_histograms.png"),
        plot_categorical(df, fig_dir / "categorical_disease_rate.png"),
        plot_correlation(df, fig_dir / "correlation_heatmap.png"),
        plot_missing(raw, fig_dir / "missing_values.png"),
    ]
    summary = summarise(raw, df)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    for o in outputs:
        logger.info("figure: %s", o)
    logger.info("summary: %s", summary_path)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate EDA figures and summary.")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = run()
    print(json.dumps({k: s[k] for k in ("rows_raw", "rows_clean", "missing_raw",
                                        "target_counts", "corr_with_target_sorted")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
