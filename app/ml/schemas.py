"""Pydantic schemas for ML tool inputs.

These are the contract the agent's `predict_*` tools speak. They double as the
single source of truth for column ordering at inference time: `to_frame()`
produces a single-row DataFrame with the columns the trained pipeline expects.

Categorical fields with small, stable cardinalities use `Literal` so the agent
gets immediate validation feedback. Open-ended categoricals (`country`,
`merchant_category`) accept any `str` because:
- `OneHotEncoder(handle_unknown="ignore")` handles novel values gracefully.
- Pinning ~50 country names in a `Literal` would be brittle and unhelpful to
  the LLM.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# --- Fraud --------------------------------------------------------------------

FraudTransactionType = Literal["purchase", "withdrawal"]
FraudDeviceType = Literal["ATM", "desktop", "mobile"]


class FraudFeatures(BaseModel):
    """Inputs for `predict_fraud`.

    Mirrors `datasets/fraud_dataset.csv` minus IDs and the target. Field order
    here also defines DataFrame column order via `to_frame()`.
    """

    model_config = ConfigDict(extra="forbid")

    customer_age: int = Field(ge=0, le=120)
    account_age_days: int = Field(ge=0)
    transaction_amount: float = Field(ge=0)
    transaction_type: FraudTransactionType
    merchant_category: str
    country: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_international: int = Field(ge=0, le=1)
    distance_from_home_km: float = Field(ge=0)
    distance_from_last_transaction_km: float = Field(ge=0)
    transaction_velocity_24h: int = Field(ge=0)
    avg_transaction_amount_30d: float = Field(ge=0)
    num_transactions_24h: int = Field(ge=0)
    num_transactions_7d: int = Field(ge=0)
    failed_transactions_24h: int = Field(ge=0)
    device_type: FraudDeviceType
    ip_reputation_score: float = Field(ge=0, le=1)
    customer_risk_score: float = Field(ge=0, le=1)
    account_balance: float
    credit_limit: float = Field(ge=0)
    debt_to_income_ratio: float = Field(ge=0)
    previous_fraud_reports: int = Field(ge=0)
    merchant_risk_score: float = Field(ge=0, le=1)
    shipping_address_match: int = Field(ge=0, le=1)
    billing_address_match: int = Field(ge=0, le=1)
    cvv_match: int = Field(ge=0, le=1)
    card_present: int = Field(ge=0, le=1)
    is_recurring: int = Field(ge=0, le=1)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


# --- Purchase -----------------------------------------------------------------

PurchaseGender = Literal["F", "M"]
PurchaseIncomeBracket = Literal[
    "20000-40000", "40000-60000", "60000-80000", "80000-100000", "100000+"
]
PurchaseMembershipTier = Literal["bronze", "silver", "gold", "platinum"]
PurchasePaymentMethod = Literal["credit_card", "debit_card", "digital_wallet"]
PurchaseLocationType = Literal["urban", "suburban"]
PurchaseOccupation = Literal[
    "executive", "management", "professional", "retired", "service", "student", "technical"
]
PurchaseEducation = Literal["high_school", "some_college", "bachelors", "masters", "phd"]
PurchaseMaritalStatus = Literal["single", "married", "widowed"]


class PurchaseFeatures(BaseModel):
    """Inputs for `predict_purchase`.

    Mirrors `datasets/product_purchase_dataset.csv` minus the customer ID and
    the target.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=120)
    gender: PurchaseGender
    income_bracket: PurchaseIncomeBracket
    customer_tenure_days: int = Field(ge=0)
    membership_tier: PurchaseMembershipTier
    num_transactions_last_month: int = Field(ge=0)
    num_transactions_last_year: int = Field(ge=0)
    avg_transaction_value: float = Field(ge=0)
    total_spent_last_year: float = Field(ge=0)
    preferred_category: str
    preferred_payment_method: PurchasePaymentMethod
    last_purchase_days_ago: int = Field(ge=0)
    cart_abandonment_rate: float = Field(ge=0, le=1)
    customer_satisfaction_score: float = Field(ge=0, le=5)
    loyalty_points: int = Field(ge=0)
    email_engagement_rate: float = Field(ge=0, le=1)
    mobile_app_user: int = Field(ge=0, le=1)
    social_media_follower: int = Field(ge=0, le=1)
    location_type: PurchaseLocationType
    distance_to_nearest_store_km: float = Field(ge=0)
    has_credit_card: int = Field(ge=0, le=1)
    has_children: int = Field(ge=0, le=1)
    occupation_category: PurchaseOccupation
    education_level: PurchaseEducation
    marital_status: PurchaseMaritalStatus
    owns_home: int = Field(ge=0, le=1)
    num_customer_service_contacts: int = Field(ge=0)
    product_return_rate: float = Field(ge=0, le=1)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


# --- Prediction outputs -------------------------------------------------------


class FeatureContribution(BaseModel):
    feature: str
    importance: float


class FraudPrediction(BaseModel):
    probability: float
    label: int
    top_features: list[FeatureContribution]


class PurchasePrediction(BaseModel):
    predicted_amount: float
    top_features: list[FeatureContribution]
