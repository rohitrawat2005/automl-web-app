"""
AutoML Trainer Module
Trains and evaluates multiple ML models for classification and regression tasks.
Supports all major scikit-learn algorithms plus optional XGBoost.
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso,
    LogisticRegression,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional XGBoost
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore[import-untyped]
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.info("XGBoost not installed — skipping XGBoost models.")


# ============================= MODEL REGISTRY ==============================

def _get_regression_models() -> Dict[str, Any]:
    """Return a dictionary of regression model instances."""
    models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoostRegressor": AdaBoostRegressor(n_estimators=100, random_state=42),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
    }
    if _XGBOOST_AVAILABLE:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False
        )
    return models


def _get_classification_models() -> Dict[str, Any]:
    """Return a dictionary of classification model instances."""
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100, random_state=42),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(kernel="rbf", probability=True, random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
    }
    if _XGBOOST_AVAILABLE:
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric="logloss",
        )
    return models


# ============================== METRICS ====================================

def _compute_regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute RMSE, MAE, and R² for regression."""
    rmse: float = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae: float = float(mean_absolute_error(y_true, y_pred))
    r2: float = float(r2_score(y_true, y_pred))
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }


def _compute_classification_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute Accuracy, Precision, Recall, F1, and Confusion Matrix for classification."""
    avg = "weighted" if len(set(y_true)) > 2 else "binary"
    acc: float = float(accuracy_score(y_true, y_pred))
    prec: float = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec: float = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    f1: float = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ========================= FEATURE IMPORTANCE ==============================

def _extract_feature_importance(pipeline: Pipeline) -> Optional[List[float]]:
    """
    Extract feature importances from the final estimator in the pipeline.
    Supports tree-based (feature_importances_) and linear (coef_) models.
    Returns None when extraction is not possible.
    """
    try:
        estimator = pipeline.steps[-1][1]

        if hasattr(estimator, "feature_importances_"):
            importances: Any = estimator.feature_importances_
            return [round(float(v), 6) for v in importances]

        if hasattr(estimator, "coef_"):
            coefs: Any = np.abs(estimator.coef_)
            if coefs.ndim > 1:
                coefs = coefs.mean(axis=0)
            return [round(float(v), 6) for v in coefs]

    except Exception as exc:
        logger.warning("Could not extract feature importance: %s", exc)

    return None


# ========================= MAIN TRAIN LOOP =================================

def _is_better(new: float, old: float) -> bool:
    """Return True if the new score is better (higher) than the old."""
    return new > old


def train_and_evaluate(
    problem_type: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    preprocessor: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str], Optional[Pipeline], Optional[List[float]]]:
    """
    Train all models for the detected problem type, evaluate them,
    and return results + best model.

    Returns
    -------
    results : dict          Per-model metric dictionaries
    best_model_name : str   Name of the top-performing model
    best_pipeline : Pipeline  Fitted sklearn Pipeline (preprocessor + model)
    feature_importance : list | None  Feature importances of the best model
    """

    compute_metrics: Callable[[Any, Any], Dict[str, Any]]

    if problem_type == "regression":
        models = _get_regression_models()
        compute_metrics = _compute_regression_metrics
        primary_metric = "R2"
    else:
        models = _get_classification_models()
        compute_metrics = _compute_classification_metrics
        primary_metric = "Accuracy"

    results: Dict[str, Dict[str, Any]] = {}
    best_score: Optional[float] = None
    best_model_name: Optional[str] = None
    best_pipeline: Optional[Pipeline] = None

    for name, model in models.items():
        try:
            logger.info("Training %s …", name)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = compute_metrics(y_test, predictions)
            results[name] = metrics

            score: float = float(metrics[primary_metric])
            if best_score is None or _is_better(score, best_score):
                best_score = score
                best_model_name = name
                best_pipeline = pipeline

        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)
            results[name] = {"error": str(exc)}

    # Feature importance from the best model
    feature_importance: Optional[List[float]] = (
        _extract_feature_importance(best_pipeline) if best_pipeline else None
    )

    return results, best_model_name, best_pipeline, feature_importance