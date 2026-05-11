"""CLI: train both ML models and write artifacts to `models/`.

Run via `make train` or `uv run python -m scripts.train_all`. Idempotent —
overwrites existing `fraud.joblib`, `purchase.joblib`, and `metrics.json`.
"""

import json
import logging

import joblib

from app.config import settings
from app.ml.train_fraud import train_fraud
from app.ml.train_purchase import train_purchase

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("training fraud classifier")
    fraud_model, fraud_metrics = train_fraud()
    joblib.dump(fraud_model, settings.MODELS_DIR / "fraud.joblib")
    logger.info("fraud CV: %s", fraud_metrics["cv"])  # pyright: ignore[reportAny]

    logger.info("training purchase regressor")
    purchase_model, purchase_metrics = train_purchase()
    joblib.dump(purchase_model, settings.MODELS_DIR / "purchase.joblib")
    logger.info("purchase CV: %s", purchase_metrics["cv"])  # pyright: ignore[reportAny]

    metrics = {"fraud": fraud_metrics, "purchase": purchase_metrics}
    with (settings.MODELS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info("wrote %s", settings.MODELS_DIR / "metrics.json")


if __name__ == "__main__":
    main()
