"""
AutoML Validator Module
Detects whether the ML task is classification or regression
based on the target column characteristics.
"""

from __future__ import annotations

import pandas as pd


def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """
    Detect whether the target column represents a classification or
    regression task.

    Heuristics
    ----------
    - object / category / bool dtypes → classification
    - integer with ≤ 20 unique values → classification
    - otherwise → regression
    """

    col = df[target]

    if col.dtype in ("object", "category", "bool"):
        return "classification"

    if pd.api.types.is_integer_dtype(col) and col.nunique() <= 20:
        return "classification"

    return "regression"
