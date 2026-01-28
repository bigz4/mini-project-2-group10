"""
evaluate.py
Evaluate a saved model artifact on a dataset CSV.

Reads from --artifacts:
- model.joblib
- metadata.json

Writes to --outdir:
- results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

import preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset CSV for evaluation.")
    ap.add_argument("--artifacts", default="artifacts", help="Directory containing model.joblib and metadata.json.")
    ap.add_argument("--outdir", default="artifacts", help="Directory to write results.json.")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold (default uses metadata threshold).")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts / "model.joblib"
    meta_path = artifacts / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing artifacts. Expected {model_path} and {meta_path}.")

    model = joblib.load(model_path)
    metadata: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

    df = preprocess.load_dataset(args.csv)
    X, y = preprocess.split_X_y(df)

    # Ensure missingness indicators exist at inference time too
    X, _ = preprocess.add_missingness_indicators(X)

    prob = model.predict_proba(X)[:, 1]
    thr = float(args.threshold) if args.threshold is not None else float(metadata.get("threshold", 0.5))
    pred = (prob >= thr).astype(int)

    results = {
        "strategy": metadata.get("strategy"),
        "threshold": thr,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

    out_path = outdir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
