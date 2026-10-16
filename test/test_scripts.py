"""End-to-end tests of the command-line entry points (offline)."""

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from heartrisk import data, eda, predict

ROOT = Path(__file__).resolve().parents[1]


def _load_download_module():
    spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_data_cli_writes_clean_csv_and_report(raw_file, tmp_path):
    out, rep = tmp_path / "clean.csv", tmp_path / "report.json"
    assert data.main(["--raw", str(raw_file), "--out", str(out), "--report", str(rep)]) == 0
    report = json.loads(rep.read_text())
    assert out.exists() and report["duplicates_removed"] == 2
    assert report["rows_clean"] == len(data.load_clean(out))


def test_eda_generates_all_required_figures(raw_file, clean_file, tmp_path):
    summary = eda.run(raw_path=raw_file, clean_path=clean_file, fig_dir=tmp_path / "fig",
                      summary_path=tmp_path / "eda_summary.json")
    for name in ("class_balance", "numeric_histograms", "correlation_heatmap",
                 "categorical_disease_rate", "missing_values", "raw_target_num"):
        f = tmp_path / "fig" / f"{name}.png"
        assert f.exists() and f.stat().st_size > 5000, name
    assert summary["missing_raw"] == {"ca": 2, "thal": 2}
    assert sum(summary["target_counts"].values()) == summary["rows_clean"]


def test_predict_cli_json(trained, tmp_path, capsys):
    out = tmp_path / "pred.json"
    rc = predict.main(["--input", str(ROOT / "samples" / "sample_request.json"),
                       "--model-dir", str(trained["model_dir"]), "--output", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())[0]
    assert res["prediction"] in (0, 1) and "confidence" in res


def test_download_validate_accepts_uci_format(raw_file):
    dl = _load_download_module()
    assert dl.validate(raw_file.read_bytes()) > 100


def test_download_validate_rejects_bad_rows():
    dl = _load_download_module()
    with pytest.raises(ValueError, match="columns"):
        dl.validate(b"1,2,3\n4,5,6\n")


def test_download_extracts_from_official_zip_layout(raw_file):
    dl = _load_download_module()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("processed.cleveland.data", raw_file.read_bytes())
        zf.writestr("heart-disease.names", "docs")
    assert dl.from_zip(buf.getvalue()) == raw_file.read_bytes()
