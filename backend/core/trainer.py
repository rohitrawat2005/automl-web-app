from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import make_pipeline
import numpy as np


def train_and_evaluate(problem_type, X_train, X_test, y_train, y_test, preprocessor):

    results = {}
    best_score = None
    best_model_name = None
    best_pipeline = None

    if problem_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42)
        }

        for name, model in models.items():
            pipeline = make_pipeline(preprocessor, model)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            results[name] = {
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            }

            # Choose best based on R2
            if best_score is None or r2 > best_score:
                best_score = r2
                best_model_name = name
                best_pipeline = pipeline

    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(random_state=42)
        }

        for name, model in models.items():
            pipeline = make_pipeline(preprocessor, model)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            results[name] = {
                "Accuracy": round(accuracy, 4)
            }

            # Choose best based on accuracy
            if best_score is None or accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                best_pipeline = pipeline

    return results, best_model_name, best_pipeline
