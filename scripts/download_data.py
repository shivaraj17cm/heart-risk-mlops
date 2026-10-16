#!/usr/bin/env python
"""Reproducible acquisition of the UCI Heart Disease (Cleveland) dataset.

Source  : UCI Machine Learning Repository, "Heart Disease" (dataset id 45)
          https://archive.ics.uci.edu/dataset/45/heart+disease
File    : processed.cleveland.data  (303 rows, 14 columns, no header, '?' = missing)
Licence : CC BY 4.0 (Janosi, Steinbrunn, Pfisterer & Detrano, 1989)

Behaviour
- Tries the direct file URL, then the official dataset ZIP as a fallback.
- Retries with back-off; validates the file (row/column count, parseable).
- Writes data/raw/manifest.json with URL, SHA-256, size and timestamp.
- Idempotent: if the file already exists and matches the manifest checksum it is
  reused (use --force to re-download). If the network is unavailable but a
  checksum-verified copy exists, that copy is used and a warning is printed.

Usage:  python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET_FILE = RAW_DIR / "processed.cleveland.data"
MANIFEST = RAW_DIR / "manifest.json"

DIRECT_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "heart-disease/processed.cleveland.data")
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
EXPECTED_COLUMNS = 14
EXPECTED_ROWS = 303


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, attempts: int = 3, timeout: int = 30) -> bytes:
    last = None
    for i in range(1, attempts + 1):
        try:
            print(f"[download] GET {url} (attempt {i}/{attempts})")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 * i)
    raise ConnectionError(f"Failed to download {url}: {last}")


def from_zip(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("processed.cleveland.data"))
        return zf.read(name)


def validate(content: bytes) -> int:
    lines = [ln for ln in content.decode("utf-8", errors="strict").splitlines() if ln.strip()]
    bad = [i for i, ln in enumerate(lines) if len(ln.split(",")) != EXPECTED_COLUMNS]
    if bad:
        raise ValueError(f"{len(bad)} rows do not have {EXPECTED_COLUMNS} columns (first: line {bad[0] + 1})")
    if len(lines) != EXPECTED_ROWS:
        print(f"[download] WARNING: expected {EXPECTED_ROWS} rows, found {len(lines)}")
    return len(lines)


def load_manifest() -> dict | None:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-download even if a verified copy exists")
    args = parser.parse_args(argv)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    if TARGET_FILE.exists() and manifest and not args.force:
        local = sha256(TARGET_FILE.read_bytes())
        if local == manifest["sha256"]:
            print(f"[download] {TARGET_FILE.relative_to(ROOT)} already present, SHA-256 verified ({local[:12]}...)")
            return 0
        print("[download] Local file checksum does not match manifest - re-downloading")

    content, source = None, None
    errors = []
    for url, extractor in ((DIRECT_URL, None), (ZIP_URL, from_zip)):
        try:
            blob = fetch(url)
            content = extractor(blob) if extractor else blob
            source = url
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            print(f"[download] {exc}")

    if content is None:
        if TARGET_FILE.exists() and manifest and sha256(TARGET_FILE.read_bytes()) == manifest["sha256"]:
            print("[download] WARNING: network unavailable, using checksum-verified local copy")
            return 0
        print("[download] ERROR: could not obtain the dataset:\n  " + "\n  ".join(errors))
        return 1

    rows = validate(content)
    digest = sha256(content)
    if manifest and manifest.get("sha256") and manifest["sha256"] != digest:
        print(f"[download] WARNING: checksum changed from {manifest['sha256'][:12]} to {digest[:12]} "
              "- upstream file differs from the recorded version")
    TARGET_FILE.write_bytes(content)
    MANIFEST.write_text(json.dumps({
        "dataset": "UCI Heart Disease - Cleveland (processed.cleveland.data)",
        "uci_dataset_id": 45,
        "source_url": source,
        "landing_page": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "license": "CC BY 4.0",
        "sha256": digest,
        "bytes": len(content),
        "rows": rows,
        "columns": EXPECTED_COLUMNS,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    print(f"[download] Saved {rows} rows to {TARGET_FILE.relative_to(ROOT)} (SHA-256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
