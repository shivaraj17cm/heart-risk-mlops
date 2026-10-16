"""Loading the packaged model and running inference (shared by the CLI and the API)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from heartrisk import paths
from heartrisk.schema import FEATURES, TARGET_LABELS

# NOTE: this module is imported by the API image, which ships only requirements-serve.txt.
# Keep it free of training/plotting dependencies (matplotlib, mlflow, ...).


def positive_proba(model, X) -> np.ndarray:
    """Probability of class 1, located via `classes_` rather than assuming column order."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    def predict_records(self, records: list[dict]) -> list[dict]:
        """Predict for a list of feature dicts. Missing optional keys become NaN (imputed)."""
        frame = pd.DataFrame.from_records(records)
        for col in FEATURES:
            if col not in frame:
                frame[col] = float("nan")
        frame = frame[FEATURES].astype(float)  # enforce column order and dtype
        p1 = positive_proba(self.pipeline, frame)
        results = []
        for p in p1:
            pred = int(p >= self.threshold)
            results.append({
                "prediction": pred,
                "label": TARGET_LABELS[pred],
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if pred == 1 else 1.0 - p), 4),
            })
        return results


def load_model(model_dir: Path | str = paths.MODEL_DIR) -> LoadedModel:
    model_dir = Path(model_dir)
    model_file = model_dir / paths.MODEL_FILE_NAME
    meta_file = model_dir / paths.METADATA_FILE_NAME
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_file}. Run training first.")
    metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return LoadedModel(pipeline=joblib.load(model_file), metadata=metadata)
