"""Tests for ML schemas and inference.

Focused on things that could plausibly break in a real change:
- Schema field validators (range/enum) — easy to change a field accidentally.
- Inference output shape and bounds — catches model-vs-schema drift.
- `metrics.json` structure — the inference path reads from it; if `train_all`
  changes the shape, predictions silently lose `top_features`.

Pure pydantic behaviour (extra="forbid", missing field raises) is not tested
here — that's pydantic's contract, not ours.
"""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pydantic import ValidationError

from app.config import settings
from app.ml.inference import predict_fraud, predict_purchase
from app.ml.schemas import FraudFeatures, PurchaseFeatures

FRAUD_CSV = Path("datasets/fraud_dataset.csv")
PURCHASE_CSV = Path("datasets/product_purchase_dataset.csv")


def _fraud_row(idx: int = 0) -> dict[str, Any]:
    df = pd.read_csv(FRAUD_CSV)
    row = df.drop(columns=["transaction_id", "customer_id", "fraud"]).iloc[idx]
    return cast(dict[str, Any], row.to_dict())


def _purchase_row(idx: int = 0) -> dict[str, Any]:
    df = pd.read_csv(PURCHASE_CSV)
    row = df.drop(columns=["customer_id", "purchase_amount"]).iloc[idx]
    return cast(dict[str, Any], row.to_dict())


# --- Schema validators that we own (not pydantic builtins) -------------------


def test_fraud_schema_rejects_invalid_categorical():
    bad = _fraud_row()
    bad["transaction_type"] = "wire_transfer"  # not in Literal
    with pytest.raises(ValidationError):
        FraudFeatures(**bad)  # pyright: ignore[reportAny]


def test_fraud_schema_rejects_out_of_range():
    bad = _fraud_row()
    bad["ip_reputation_score"] = 1.5  # constrained to [0, 1]
    with pytest.raises(ValidationError):
        FraudFeatures(**bad)  # pyright: ignore[reportAny]


def test_purchase_schema_rejects_invalid_membership_tier():
    bad = _purchase_row()
    bad["membership_tier"] = "diamond"
    with pytest.raises(ValidationError):
        PurchaseFeatures(**bad)  # pyright: ignore[reportAny]


# --- Inference shape and bounds ----------------------------------------------


def test_predict_fraud_returns_valid_probability():
    features = FraudFeatures(**_fraud_row())  # pyright: ignore[reportAny]
    result = predict_fraud(features)
    assert 0.0 <= result.probability <= 1.0
    assert result.label in (0, 1)
    assert len(result.top_features) == 5


def test_predict_purchase_returns_positive_amount():
    features = PurchaseFeatures(**_purchase_row())  # pyright: ignore[reportAny]
    result = predict_purchase(features)
    # All training targets are > 0, so a sane regressor should never go
    # meaningfully negative on in-distribution input.
    assert result.predicted_amount > 0
    assert len(result.top_features) == 5


# --- Metrics file is what inference assumes ---------------------------------


def test_metrics_json_has_expected_structure():
    metrics = cast(
        dict[str, Any],
        json.loads((settings.MODELS_DIR / "metrics.json").read_text()),
    )
    for model_key in ("fraud", "purchase"):
        assert "cv" in metrics[model_key]
        assert "permutation_importance" in metrics[model_key]
        assert metrics[model_key]["permutation_importance"], (
            f"{model_key} has empty permutation_importance"
        )
