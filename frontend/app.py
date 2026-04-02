"""
AutoML Web App — Premium Streamlit Frontend
A modern SaaS-style dashboard for automated machine learning.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────── CONFIG ───────────────────────────

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = "http://127.0.0.1:8000"

# ─────────────────────────── CUSTOM CSS ───────────────────────

st.markdown("""
<style>
/* ── Import Google Font ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ─── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Hero Header ─── */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.25);
}
.hero-header h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section Headers ─── */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
    display: inline-block;
}

/* ── Metric Cards ─── */
.metric-card {
    background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    border: 1px solid #e8ecf4;
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.12);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #667eea;
    line-height: 1.2;
}
.metric-card .metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.3rem;
}

/* ── Best Model Banner ─── */
.best-model-banner {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 6px 25px rgba(17, 153, 142, 0.3);
    margin: 1rem 0;
}

/* ── Card Container ─── */
.card-container {
    background: #ffffff;
    border: 1px solid #edf0f7;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    margin-bottom: 1rem;
}

/* ── Upload Area ─── */
.upload-area {
    background: linear-gradient(135deg, #fafbff 0%, #f0f2ff 100%);
    border: 2px dashed #667eea;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* ── Footer ─── */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid #edf0f7;
}

/* ── Spinner style ─── */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* ── Streamlit overrides ─── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35);
    color: white;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── HELPERS ──────────────────────────

def metric_card(label: str, value, col):
    """Render a styled metric card inside a Streamlit column."""
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────── HEADER ───────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>⚡ AutoML Studio</h1>
    <p>Upload your dataset • Train 19 ML models • Download the best one — all in one click</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── UPLOAD ───────────────────────────

st.markdown('<div class="section-header">📂 Upload Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop your CSV file here",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ───────────────── DATASET PREVIEW ─────────────────
    st.markdown('<div class="section-header">🔍 Dataset Preview</div>', unsafe_allow_html=True)

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    # ───────────────── DATASET STATS ───────────────────
    st.markdown('<div class="section-header">📊 Dataset Statistics</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").shape[1]
    cat_cols = df.select_dtypes(include=["object", "category"]).shape[1]
    missing = int(df.isnull().sum().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card("Total Rows", f"{df.shape[0]:,}", c1)
    metric_card("Total Columns", df.shape[1], c2)
    metric_card("Numeric", num_cols, c3)
    metric_card("Categorical", cat_cols, c4)
    metric_card("Missing Values", f"{missing:,}", c5)

    with st.expander("📋 Descriptive Statistics", expanded=False):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ───────────────── TARGET SELECTION ─────────────────
    st.markdown('<div class="section-header">🎯 Target Column</div>', unsafe_allow_html=True)

    target_column = st.selectbox(
        "Select the column you want to predict",
        options=df.columns,
        label_visibility="collapsed",
    )

    # Detect problem type for user info
    unique_vals = df[target_column].nunique()
    if df[target_column].dtype == "object" or (pd.api.types.is_integer_dtype(df[target_column]) and unique_vals <= 20):
        detected_type = "Classification"
    else:
        detected_type = "Regression"

    st.info(f"🔎 Detected problem type: **{detected_type}** • Target has **{unique_vals}** unique values")

    # ───────────────── TRAIN BUTTON ────────────────────
    if st.button("🚀 Start Training", use_container_width=True):
        with st.spinner("⏳ Training 19 models… this may take a moment…"):
            uploaded_file.seek(0)
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": uploaded_file},
                    data={"target": target_column},
                    timeout=600,
                )
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the backend. Make sure the FastAPI server is running.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Request failed: {e}")
                st.stop()

        if response.status_code != 200:
            st.error("❌ Training failed!")
            try:
                st.json(response.json())
            except Exception:
                st.code(response.text)
            st.stop()

        result = response.json()

        st.success("✅ Training completed successfully!")

        # ════════════════════════════════════════════════
        #  RESULTS
        # ════════════════════════════════════════════════

        model_results = result.get("model_results", {})
        best_model = result.get("best_model")
        feature_importance = result.get("feature_importance")
        feature_names = result.get("feature_names")
        problem_type = result.get("problem_type", "")
        dataset_id = result.get("dataset_id")

        # ───── Best Model Banner ─────
        if best_model:
            primary_metric = "R2" if problem_type == "regression" else "Accuracy"
            best_score = model_results.get(best_model, {}).get(primary_metric, "N/A")
            st.markdown(f"""
            <div class="best-model-banner">
                🏆 Best Model: {best_model} &nbsp;|&nbsp; {primary_metric}: {best_score}
            </div>
            """, unsafe_allow_html=True)

        # ───── Model Comparison Table ─────
        if model_results:
            st.markdown('<div class="section-header">📋 Model Comparison</div>', unsafe_allow_html=True)

            # Filter out error-only results for the table
            valid_results = {
                k: {mk: mv for mk, mv in v.items() if mk != "ConfusionMatrix"}
                for k, v in model_results.items()
                if "error" not in v
            }
            if valid_results:
                df_results = pd.DataFrame(valid_results).T
                df_results.index.name = "Model"
                df_results = df_results.reset_index()

                # Sort by primary metric
                sort_col = "R2" if problem_type == "regression" else "Accuracy"
                if sort_col in df_results.columns:
                    df_results = df_results.sort_values(sort_col, ascending=False)

                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True,
                )

            # Show errors if any
            error_models = {k: v["error"] for k, v in model_results.items() if "error" in v}
            if error_models:
                with st.expander("⚠️ Models with errors"):
                    for name, err in error_models.items():
                        st.warning(f"**{name}**: {err}")

        # ───── Model Performance Charts ─────
        if model_results:
            st.markdown('<div class="section-header">📈 Performance Visualizations</div>', unsafe_allow_html=True)

            chart_data = {
                k: v for k, v in model_results.items()
                if "error" not in v
            }

            if problem_type == "regression":
                # RMSE + R² grouped bar chart
                chart_df = pd.DataFrame(chart_data).T.reset_index()
                chart_df.columns = ["Model"] + list(chart_df.columns[1:])

                col_a, col_b = st.columns(2)

                with col_a:
                    if "RMSE" in chart_df.columns:
                        fig_rmse = px.bar(
                            chart_df.sort_values("RMSE"),
                            x="Model", y="RMSE",
                            color="RMSE",
                            color_continuous_scale="RdYlGn_r",
                            title="RMSE by Model (lower is better)",
                        )
                        fig_rmse.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter"),
                            showlegend=False,
                            xaxis_tickangle=-45,
                            height=420,
                        )
                        st.plotly_chart(fig_rmse, use_container_width=True)

                with col_b:
                    if "R2" in chart_df.columns:
                        fig_r2 = px.bar(
                            chart_df.sort_values("R2", ascending=False),
                            x="Model", y="R2",
                            color="R2",
                            color_continuous_scale="Viridis",
                            title="R² Score by Model (higher is better)",
                        )
                        fig_r2.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter"),
                            showlegend=False,
                            xaxis_tickangle=-45,
                            height=420,
                        )
                        st.plotly_chart(fig_r2, use_container_width=True)

            else:
                # Classification: Accuracy + F1 chart
                chart_df = pd.DataFrame({
                    name: {k: v for k, v in metrics.items() if k != "ConfusionMatrix"}
                    for name, metrics in chart_data.items()
                }).T.reset_index()
                chart_df.columns = ["Model"] + list(chart_df.columns[1:])

                col_a, col_b = st.columns(2)

                with col_a:
                    if "Accuracy" in chart_df.columns:
                        fig_acc = px.bar(
                            chart_df.sort_values("Accuracy", ascending=False),
                            x="Model", y="Accuracy",
                            color="Accuracy",
                            color_continuous_scale="Viridis",
                            title="Accuracy by Model",
                        )
                        fig_acc.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter"),
                            showlegend=False,
                            xaxis_tickangle=-45,
                            height=420,
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)

                with col_b:
                    if "F1" in chart_df.columns:
                        fig_f1 = px.bar(
                            chart_df.sort_values("F1", ascending=False),
                            x="Model", y="F1",
                            color="F1",
                            color_continuous_scale="Plasma",
                            title="F1 Score by Model",
                        )
                        fig_f1.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter"),
                            showlegend=False,
                            xaxis_tickangle=-45,
                            height=420,
                        )
                        st.plotly_chart(fig_f1, use_container_width=True)

                # All metrics radar-style grouped bar
                if len(chart_df.columns) > 2:
                    metric_cols = [c for c in chart_df.columns if c not in ("Model",)]
                    fig_grouped = go.Figure()
                    colors = px.colors.qualitative.Set2
                    for i, metric in enumerate(metric_cols):
                        fig_grouped.add_trace(go.Bar(
                            name=metric,
                            x=chart_df["Model"],
                            y=chart_df[metric],
                            marker_color=colors[i % len(colors)],
                        ))
                    fig_grouped.update_layout(
                        barmode="group",
                        title="All Metrics Comparison",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter"),
                        xaxis_tickangle=-45,
                        height=450,
                    )
                    st.plotly_chart(fig_grouped, use_container_width=True)

        # ───── Model Ranking ─────
        if model_results:
            st.markdown('<div class="section-header">🏅 Model Rankings</div>', unsafe_allow_html=True)

            ranking_data = {
                k: v for k, v in model_results.items()
                if "error" not in v
            }
            if ranking_data:
                sort_metric = "R2" if problem_type == "regression" else "Accuracy"
                ranked = sorted(
                    ranking_data.items(),
                    key=lambda x: x[1].get(sort_metric, 0),
                    reverse=True,
                )
                medals = ["🥇", "🥈", "🥉"]
                for i, (name, metrics) in enumerate(ranked):
                    medal = medals[i] if i < 3 else f"  {i + 1}."
                    score = metrics.get(sort_metric, "N/A")
                    st.markdown(f"**{medal} {name}** — {sort_metric}: `{score}`")

        # ───── Feature Importance ─────
        if feature_importance:
            st.markdown('<div class="section-header">🧬 Feature Importance</div>', unsafe_allow_html=True)

            fi_labels = feature_names if feature_names and len(feature_names) == len(feature_importance) else [
                f"Feature {i}" for i in range(len(feature_importance))
            ]

            fi_df = pd.DataFrame({
                "Feature": fi_labels,
                "Importance": feature_importance,
            }).sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                fi_df,
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title=f"Feature Importance — {best_model}",
            )
            fig_fi.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=max(350, len(fi_labels) * 28),
                yaxis=dict(tickfont=dict(size=11)),
                showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # ───── Confusion Matrix Heatmap ─────
        if problem_type == "classification" and best_model:
            cm_data = model_results.get(best_model, {}).get("ConfusionMatrix")
            if cm_data:
                st.markdown('<div class="section-header">🔥 Confusion Matrix — Best Model</div>', unsafe_allow_html=True)

                cm_array = pd.DataFrame(cm_data)
                labels = sorted(df[target_column].unique())
                if len(labels) == len(cm_array):
                    cm_array.index = labels
                    cm_array.columns = labels

                fig_cm = px.imshow(
                    cm_array.values,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[str(l) for l in cm_array.columns],
                    y=[str(l) for l in cm_array.index],
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title=f"Confusion Matrix — {best_model}",
                )
                fig_cm.update_layout(
                    font=dict(family="Inter"),
                    height=450,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # ───── Download ─────
        if dataset_id:
            st.markdown('<div class="section-header">📥 Download Best Model</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="card-container" style="text-align:center;">
                <p style="font-size:1rem; color:#4b5563; margin-bottom:0.3rem;">
                    Your best model is ready for deployment
                </p>
                <p style="font-size:0.85rem; color:#9ca3af;">
                    Saved as a scikit-learn Pipeline (.pkl) — load with <code>joblib.load()</code>
                </p>
            </div>
            """, unsafe_allow_html=True)

            try:
                model_response = requests.get(f"{BACKEND_URL}/download/{dataset_id}", timeout=60)
                if model_response.status_code == 200:
                    st.download_button(
                        label="⬇️  Download Best Model (.pkl)",
                        data=model_response.content,
                        file_name=f"{dataset_id}_best_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
                else:
                    st.warning("Model file not available for download.")
            except Exception:
                download_url = f"{BACKEND_URL}/download/{dataset_id}"
                st.markdown(f"[⬇️ Download Model]({download_url})")

# ─────────────────────────── FOOTER ───────────────────────────

st.markdown("""
<div class="footer">
    Built with ❤️ using FastAPI + Streamlit &nbsp;|&nbsp; AutoML Studio v2.0
</div>
""", unsafe_allow_html=True)