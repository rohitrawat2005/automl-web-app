# AutoML Web App — Upgrade Walkthrough

## What Changed

### Backend — [trainer.py](file:///c:/automl-web-app/backend/core/trainer.py) (full rewrite)
- **9 regression models**: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting, XGBoost (optional), AdaBoost, ExtraTrees
- **10 classification models**: LogisticRegression, DecisionTree, RandomForest, GradientBoosting, XGBoost (optional), AdaBoost, ExtraTrees, SVC, KNeighbors, GaussianNB
- **Regression metrics**: RMSE, MAE, R²
- **Classification metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- XGBoost gracefully skipped if not installed
- Feature importance works for tree-based (`.feature_importances_`) AND linear models (`.coef_`)
- Per-model error handling — a failing model won't crash the pipeline

### Backend — [preprocessing.py](file:///c:/automl-web-app/backend/core/preprocessing.py)
- Feature name tracking via `get_feature_names_out()` — labels survive one-hot encoding
- Broader dtype support (`int32`, `float32`, `bool`, `category`)
- Median imputation for numerics (more robust to outliers)

### Backend — [validator.py](file:///c:/automl-web-app/backend/core/validator.py)
- Added `bool` and `category` dtype detection
- Cleaner heuristic: only integer columns use the ≤20 unique threshold

### Backend — [main.py](file:///c:/automl-web-app/backend/main.py)
- **CORS middleware** enabled
- **`/health`** endpoint added
- **`/preview`** endpoint — explore dataset without training
- Upload response enriched with full dataset statistics (dtypes, missing per column, descriptive stats)
- `feature_names` returned alongside [feature_importance](file:///c:/automl-web-app/backend/core/trainer.py#112-134)
- Structured error handling throughout

### Frontend — [app.py](file:///c:/automl-web-app/frontend/app.py) (full rewrite)
- **Premium SaaS design**: gradient hero header, styled metric cards, modern spacing, Inter font
- **5 metric cards**: Rows, Columns, Numeric, Categorical, Missing Values
- **Plotly charts**: RMSE/R² bar charts (regression), Accuracy/F1 bar charts (classification), grouped all-metrics comparison
- **Confusion matrix heatmap** for best classification model
- **Model rankings** with medal emojis
- **Feature importance** — horizontal bar chart sorted by importance
- **Inline download button** — no more link-only download

### Dependencies
- Added `xgboost>=2.0.0` and `plotly>=5.18.0` to [requirements.txt](file:///c:/automl-web-app/requirements.txt)

## Verification

| Check | Result |
|-------|--------|
| All backend imports | ✅ Pass |
| xgboost + plotly installed | ✅ Pass |
| Python syntax (all files) | ✅ Pass |

## How to Run

```bash
# Terminal 1 — Backend
cd c:\automl-web-app
venv\Scripts\activate
uvicorn backend.main:app --reload

# Terminal 2 — Frontend
cd c:\automl-web-app
venv\Scripts\activate
streamlit run frontend/app.py
```
