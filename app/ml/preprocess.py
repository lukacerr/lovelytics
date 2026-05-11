"""Shared training utilities: preprocessor + CV evaluator.

`HistGradientBoosting{Classifier,Regressor}` handles missing values and feature
scaling natively, so the preprocessor only needs to one-hot the categoricals.
We use `OneHotEncoder(handle_unknown="ignore")` so novel categorical values at
inference time produce all-zero rows instead of crashing — this matters because
the agent may pass country/merchant strings that weren't in the training set.
"""

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(numeric_cols)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_cols),
            ),
        ],
        remainder="drop",
    )


def expanded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names *after* one-hot expansion.

    Used to align permutation-importance results back to human-readable labels
    in `metrics.json`.
    """
    return list(preprocessor.get_feature_names_out())
