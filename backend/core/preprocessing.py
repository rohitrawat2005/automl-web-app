"""
AutoML Preprocessing Module
Builds a robust sklearn ColumnTransformer pipeline and returns train/test
splits with feature name tracking.
"""

from __future__ import annotations

import pandas as pd
import numpy as np  # noqa: F401 — used transitively
from typing import Any, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, Any, ColumnTransformer, List[str]]:
    """
    Build a preprocessing pipeline and split data.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
    preprocessor : ColumnTransformer
    feature_names : list[str]
        Human-readable feature names after transformation.
    """

    X = df.drop(columns=[target])
    y = df[target]

    # Identify column types
    numeric_features: List[str] = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features: List[str] = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numeric pipeline: impute → scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute → one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Derive feature names (fit on training data only)
    preprocessor.fit(X_train)
    try:
        feature_names: List[str] = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = numeric_features + categorical_features

    # Clean up prefixes for readability (e.g., "num__Age" → "Age")
    feature_names = [
        n.split("__", 1)[1] if "__" in n else n for n in feature_names
    ]

    return X_train, X_test, y_train, y_test, preprocessor, feature_names
