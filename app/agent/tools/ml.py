"""ML prediction tools.

Tool inputs are pydantic schemas (see `app.ml.schemas`); LangChain's `@tool`
decorator picks them up automatically as the JSON-schema arguments. Outputs
are encoded with TOON (https://github.com/python-toon/python-toon) instead of
JSON to roughly halve the token count of the structured response — non-trivial
when `top_features` lists land in the model's context every turn.

The tools are declared `async def` even though the underlying sklearn
inference is CPU-bound and synchronous. The reason is interop: when *any*
tool in the agent's tool set is async (e.g. `kb_search`), LangGraph's tool
node enters the async execution path and refuses to call sync `StructuredTool`
methods. Declaring every tool async keeps the dispatch path uniform — see
AGENTS.md §2 for the project-wide async preference.
"""

import toon
from langchain_core.tools import tool

from app.ml import inference
from app.ml.schemas import FraudFeatures, PurchaseFeatures


@tool
async def predict_fraud(features: FraudFeatures) -> str:
    """Predict whether a single transaction is fraudulent.

    Pass the transaction's features as a structured object. Returns a TOON-
    encoded result with `probability` (0-1), `label` (0 or 1), and
    `top_features` — the five features the model relied on most (precomputed
    permutation importance, scored on ROC-AUC).
    """
    result = inference.predict_fraud(features)
    return toon.encode(result.model_dump())


@tool
async def predict_purchase(features: PurchaseFeatures) -> str:
    """Predict the expected purchase amount (in USD) for a customer.

    Pass the customer's features as a structured object. Returns a TOON-
    encoded result with `predicted_amount` and `top_features` — the five
    features the model relied on most (precomputed permutation importance,
    scored on R²).
    """
    result = inference.predict_purchase(features)
    return toon.encode(result.model_dump())
