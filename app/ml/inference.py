"""Inference helpers used by the agent's `predict_*` tools.

Models are loaded lazily via `lru_cache` so the FastAPI process pays the
joblib-load cost once. Permutation-importance tables are loaded from
`metrics.json` (also cached) — we never recompute importance at request time.
"""

import json
from functools import lru_cache
from typing import Any, cast

import joblib
from sklearn.pipeline import Pipeline

from app.config import settings
from app.ml.schemas import (
    FeatureContribution,
    FraudFeatures,
    FraudPrediction,
    PurchaseFeatures,
    PurchasePrediction,
)

_TOP_N_FEATURES = 5


@lru_cache(maxsize=1)
def _load_fraud_model() -> Pipeline:
    return cast(Pipeline, joblib.load(settings.MODELS_DIR / "fraud.joblib"))


@lru_cache(maxsize=1)
def _load_purchase_model() -> Pipeline:
    return cast(Pipeline, joblib.load(settings.MODELS_DIR / "purchase.joblib"))


@lru_cache(maxsize=1)
def _load_metrics() -> dict[str, Any]:
    with (settings.MODELS_DIR / "metrics.json").open() as f:
        return cast(dict[str, Any], json.load(f))


def _top_features(model_key: str, n: int = _TOP_N_FEATURES) -> list[FeatureContribution]:
    importances = cast(
        dict[str, float],
        _load_metrics()[model_key]["permutation_importance"],
    )
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [FeatureContribution(feature=name, importance=score) for name, score in ranked]


def predict_fraud(features: FraudFeatures) -> FraudPrediction:
    model = _load_fraud_model()
    frame = features.to_frame()
    proba = float(model.predict_proba(frame)[0, 1])
    return FraudPrediction(
        probability=proba,
        label=int(proba >= 0.5),
        top_features=_top_features("fraud"),
    )


def predict_purchase(features: PurchaseFeatures) -> PurchasePrediction:
    model = _load_purchase_model()
    frame = features.to_frame()
    amount = float(model.predict(frame)[0])
    return PurchasePrediction(
        predicted_amount=amount,
        top_features=_top_features("purchase"),
    )
