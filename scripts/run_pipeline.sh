#!/usr/bin/env bash
# Full reproducible pipeline on Linux/macOS (e.g. the Prayogshala lab). Stops at the first error.
# Usage:  bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== host: $(hostname) | $(python --version) | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== 1/6 acquire data";        python scripts/download_data.py
echo "== 2/6 validate + clean";     python -m heartrisk.data
echo "== 3/6 EDA figures";          python -m heartrisk.eda
echo "== 4/6 train + MLflow";       python -m heartrisk.train
echo "== 5/6 unit tests";           pytest -q
echo "== 6/6 packaged-model check"; python -m heartrisk.predict --input samples/sample_request.json
echo "== pipeline finished OK"
