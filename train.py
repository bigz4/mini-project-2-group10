"""
train.py
Train a model for Mini Project II (Loan Approval Prediction) and save artifacts.

Strategies:
- cw: Logistic Regression with class_weight='balanced' + missingness indicators + OneHot preprocessing
- smotenc: Logistic Regression + SMOTENC (categorical-aware oversampling) + missingness indicators (ordinal encoding)

Outputs in --outdir:
- model.joblib
- metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

import preprocess


def threshold_tune(y_true, y_prob, cost_fp: float = 5.0, cost_fn: float = 1.0) -> Dict[str, Any]:
    """Choose threshold minimizing expected cost = cost_fp*FP + cost_fn*FN."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best = None
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        expected_cost = cost_fp * fp + cost_fn * fn
        if best is None or expected_cost < best["expected_cost"]:
            best = {
                "threshold": float(t),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "expected_cost": float(expected_cost),
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            }
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to loan dataset CSV (train CSV).");
    ap.add_argument("--outdir", default="artifacts", help="Directory to save model artifacts.")
    ap.add_argument("--strategy", choices=["cw", "smotenc"], default="cw", help="Training strategy.")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--tune", action="store_true", help="Run small GridSearchCV (recommended for smotenc).");
    ap.add_argument("--tune_cv", type=int, default=3, help="CV folds for tuning (Colab-safe default=3).");
    ap.add_argument("--tune_jobs", type=int, default=1, help="n_jobs for tuning (Colab-safe default=1).");

    ap.add_argument("--threshold_tune", action="store_true", help="Tune decision threshold by cost on hold-out test set.");
    ap.add_argument("--cost_fp", type=float, default=5.0, help="False positive cost multiplier.");
    ap.add_argument("--cost_fn", type=float, default=1.0, help="False negative cost multiplier.");
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = preprocess.load_dataset(args.csv)
    X, y = preprocess.split_X_y(df)

    # A-upgrade: missingness indicators
    X, indicator_cols = preprocess.add_missingness_indicators(X)

    numeric_features, categorical_features = preprocess.infer_feature_types(X)

    X_train, X_test, y_train, y_test = preprocess.make_train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    metadata: Dict[str, Any] = {
        "strategy": args.strategy,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missingness_indicators": indicator_cols,
        "threshold": 0.5,
        "threshold_tuning": None,
        "tuning": None,
        "test_metrics_at_threshold": None,
    }

    if args.strategy == "cw":
        pre = preprocess.build_preprocessor_onehot(numeric_features, categorical_features)
        pipe = ImbPipeline(steps=[
            ("preprocessor", pre),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)),
        ])
        pipe.fit(X_train, y_train)

    else:
        pre, cat_indices = preprocess.build_preprocessor_smotenc(numeric_features, categorical_features)
        sm = SMOTENC(categorical_features=cat_indices, random_state=args.random_state, k_neighbors=5)
        pipe = ImbPipeline(steps=[
            ("preprocessor", pre),
            ("smotenc", sm),
            ("model", LogisticRegression(max_iter=2000, random_state=args.random_state)),
        ])

        if args.tune:
            grid = {"model__C": [0.01, 0.1, 1.0]}
            gs = GridSearchCV(pipe, param_grid=grid, scoring="f1", cv=args.tune_cv, n_jobs=args.tune_jobs)
            gs.fit(X_train, y_train)
            pipe = gs.best_estimator_
            metadata["tuning"] = {
                "param_grid": grid,
                "best_params": gs.best_params_,
                "best_cv_score_f1": float(gs.best_score_),
                "cv": args.tune_cv,
            }
        else:
            pipe.fit(X_train, y_train)

    # Evaluate on hold-out
    prob = pipe.predict_proba(X_test)[:, 1]

    if args.threshold_tune:
        tt = threshold_tune(y_test.values, prob, cost_fp=args.cost_fp, cost_fn=args.cost_fn)
        metadata["threshold"] = tt["threshold"]
        metadata["threshold_tuning"] = {
            "cost_fp": args.cost_fp,
            "cost_fn": args.cost_fn,
            "chosen": tt,
        }

    thr = float(metadata["threshold"])
    pred = (prob >= thr).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "threshold": float(thr),
    }
    metadata["test_metrics_at_threshold"] = metrics

    # Save artifacts
    model_path = outdir / "model.joblib"
    meta_path = outdir / "metadata.json"
    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
