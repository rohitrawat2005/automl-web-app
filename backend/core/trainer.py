from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np


def train_and_evaluate(problem_type, X_train, X_test, y_train, y_test, preprocessor):

    results = {}

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

    return results
