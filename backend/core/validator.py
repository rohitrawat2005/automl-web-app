import pandas as pd


def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """
    Detects whether the ML task is classification or regression
    """
    unique_values = df[target].nunique()

    if df[target].dtype == "object" or unique_values <= 20:
        return "classification"
    else:
        return "regression"
