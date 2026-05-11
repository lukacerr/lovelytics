"""Train the fraud classifier.

### Why these choices

- **5-fold stratified CV** rather than a single 80/20 holdout: with only 100
  rows a 20-row test set produces metrics with very high variance; CV gives
  a mean ± std across 5 different splits without losing data.
- **Stratified** because the target is binary; we want each fold to keep the
  50/50 class balance.
- **`shuffle=True, random_state=42`** so folds are reproducible but not
  ordered by any latent CSV-row pattern.
- **Per-fold metrics: ROC-AUC, PR-AUC, F1, accuracy**. ROC-AUC is the headline
  number for ranking; PR-AUC is more honest under class imbalance (less
  relevant here at 50/50, but kept because real fraud is always imbalanced and
  this code outlives the toy dataset). F1 + accuracy at threshold 0.5 give a
  human-readable point estimate.
- **Refit on full data after CV** so the deployed model uses every row; CV is
  for evaluation only.
- **Permutation importance** on the full data, `n_repeats=10`, scored on
  ROC-AUC — chosen because it matches the headline metric and gives stable
  rankings (single-shuffle importances are noisy).
- **Importance computed on the raw pipeline input** (not the one-hot expanded
  matrix) so the result is one number per agent-facing feature, ready to drop
  into `top_features` without any post-hoc aggregation over OHE columns.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from app.config import settings
from app.ml.preprocess import build_preprocessor

DROP_COLS = ["transaction_id", "customer_id"]
TARGET = "fraud"
CATEGORICAL_COLS = ["transaction_type", "merchant_category", "country", "device_type"]
RANDOM_STATE = 42
N_SPLITS = 5
PERMUTATION_REPEATS = 10


def _build_pipeline(numeric_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, CATEGORICAL_COLS)),
            (
                "model",
                HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )


def train_fraud() -> tuple[Pipeline, dict[str, Any]]:
    csv_path = settings.DATASETS_DIR / "fraud_dataset.csv"
    df = pd.read_csv(csv_path)
    y = df[TARGET].to_numpy()
    x = df.drop(columns=[*DROP_COLS, TARGET])
    numeric_cols = [c for c in x.columns if c not in CATEGORICAL_COLS]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "f1": [],
        "accuracy": [],
    }

    for train_idx, test_idx in cv.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline = _build_pipeline(numeric_cols)
        pipeline.fit(x_train, y_train)
        proba = pipeline.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_metrics["roc_auc"].append(float(roc_auc_score(y_test, proba)))
        fold_metrics["average_precision"].append(float(average_precision_score(y_test, proba)))
        fold_metrics["f1"].append(float(f1_score(y_test, pred)))
        fold_metrics["accuracy"].append(float(accuracy_score(y_test, pred)))

    cv_summary = {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in fold_metrics.items()
    }

    # Refit on full data for deployment.
    final = _build_pipeline(numeric_cols)
    final.fit(x, y)

    # `permutation_importance` returns a `Bunch`, but sklearn types it as `dict`.
    perm = cast(
        Bunch,
        permutation_importance(
            final,
            x,
            y,
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=1,
        ),
    )
    importance = {
        col: float(mean)
        for col, mean in zip(x.columns, perm.importances_mean, strict=True)
    }

    metrics = {"cv": cv_summary, "permutation_importance": importance}
    return final, metrics
