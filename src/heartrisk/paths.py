"""Central place for repository-relative paths (no absolute paths anywhere)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_FILE = RAW_DIR / "processed.cleveland.data"
RAW_MANIFEST = RAW_DIR / "manifest.json"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"

CONFIG_FILE = PROJECT_ROOT / "configs" / "train.yaml"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
MODEL_FILE_NAME = "model.joblib"
METADATA_FILE_NAME = "model_metadata.json"
