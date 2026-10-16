"""Command-line inference with the packaged model.

Usage:
  python -m heartrisk.predict --input samples/sample_request.json
  python -m heartrisk.predict --input patients.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from heartrisk import paths
from heartrisk.model_io import load_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict heart-disease risk.")
    parser.add_argument("--input", required=True, help="JSON object/list or CSV with feature columns")
    parser.add_argument("--output", help="Optional CSV/JSON output path")
    parser.add_argument("--model-dir", default=str(paths.MODEL_DIR))
    args = parser.parse_args(argv)

    model = load_model(args.model_dir)
    src = Path(args.input)
    if src.suffix.lower() == ".csv":
        records = pd.read_csv(src).to_dict(orient="records")
    else:
        payload = json.loads(src.read_text())
        records = payload if isinstance(payload, list) else [payload]

    results = model.predict_records(records)
    for r in results:
        r["model_version"] = model.version

    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".csv":
            pd.DataFrame(results).to_csv(out, index=False)
        else:
            out.write_text(json.dumps(results, indent=2))
    json.dump(results if len(results) > 1 else results[0], sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
