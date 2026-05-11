"""Train the purchase-amount regressor.

### Why these choices

- **5-fold KFold** (not stratified — target is continuous): same rationale as
  the fraud trainer; with n=100 a single holdout is too noisy.
- **Per-fold metrics: MAE, RMSE, R²**. MAE and RMSE are in dollars (the same
  unit as the target) so they're directly interpretable. RMSE penalises large
  errors more than MAE — useful for spotting catastrophic predictions. R²
  expresses how much variance the model explains; with n=100 expect modest
  values.
- **Refit on full data after CV**, same rationale as fraud.
- **Permutation importance** scored on R² with `n_repeats=10` — same
  reasoning as fraud, scoring matched to the headline regression metric.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from app.config import settings
from app.ml.preprocess import build_preprocessor

DROP_COLS = ["customer_id"]
TARGET = "purchase_amount"
CATEGORICAL_COLS = [
    "gender",
    "income_bracket",
    "membership_tier",
    "preferred_category",
    "preferred_payment_method",
    "location_type",
    "occupation_category",
    "education_level",
    "marital_status",
]
RANDOM_STATE = 42
N_SPLITS = 5
PERMUTATION_REPEATS = 10


def _build_pipeline(numeric_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, CATEGORICAL_COLS)),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )


def train_purchase() -> tuple[Pipeline, dict[str, Any]]:
    csv_path = settings.DATASETS_DIR / "product_purchase_dataset.csv"
    df = pd.read_csv(csv_path)
    y = df[TARGET].to_numpy()
    x = df.drop(columns=[*DROP_COLS, TARGET])
    numeric_cols = [c for c in x.columns if c not in CATEGORICAL_COLS]

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": []}

    for train_idx, test_idx in cv.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline = _build_pipeline(numeric_cols)
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        fold_metrics["mae"].append(float(mean_absolute_error(y_test, pred)))
        fold_metrics["rmse"].append(float(root_mean_squared_error(y_test, pred)))
        fold_metrics["r2"].append(float(r2_score(y_test, pred)))

    cv_summary = {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in fold_metrics.items()
    }

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
            scoring="r2",
            n_jobs=1,
        ),
    )
    importance = {
        col: float(mean)
        for col, mean in zip(x.columns, perm.importances_mean, strict=True)
    }

    metrics = {"cv": cv_summary, "permutation_importance": importance}
    return final, metrics
