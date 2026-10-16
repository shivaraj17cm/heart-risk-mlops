#!/usr/bin/env python
"""Send realistic traffic to the API so logs and the Grafana dashboard show activity.

Mixes valid predictions (random but plausible patients) with a share of invalid
requests, so request rate, latency, class mix and error panels are all populated.

Usage:  python scripts/send_traffic.py --url http://localhost:8080 --n 200
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import requests


def random_patient(rng: random.Random) -> dict:
    return {
        "age": rng.randint(30, 77), "sex": rng.choice([0, 1]), "cp": rng.choice([1, 2, 3, 4]),
        "trestbps": rng.randint(95, 190), "chol": rng.randint(130, 400), "fbs": rng.choice([0, 0, 0, 1]),
        "restecg": rng.choice([0, 1, 2]), "thalach": rng.randint(90, 200), "exang": rng.choice([0, 1]),
        "oldpeak": round(rng.uniform(0, 4.5), 1), "slope": rng.choice([1, 2, 3]),
        "ca": rng.choice([0, 0, 1, 2, 3, None]), "thal": rng.choice([3, 3, 6, 7, None]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--invalid-share", type=float, default=0.1)
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    stats = {"200": 0, "422": 0, "other": 0, "pred_1": 0}
    for i in range(args.n):
        body = random_patient(rng)
        if rng.random() < args.invalid_share:
            body[rng.choice(["sex", "cp", "chol"])] = 999           # deliberately invalid
        try:
            r = requests.post(f"{args.url}/predict", json=body, timeout=5)
        except requests.RequestException as exc:
            print(f"request {i} failed: {exc}", file=sys.stderr)
            return 1
        key = str(r.status_code) if str(r.status_code) in stats else "other"
        stats[key] += 1
        if r.status_code == 200:
            stats["pred_1"] += r.json()["prediction"]
        time.sleep(args.delay)
    print(f"sent {args.n} requests -> 200: {stats['200']} | 422: {stats['422']} | other: {stats['other']} "
          f"| predicted disease: {stats['pred_1']}")
    return 0 if stats["other"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
