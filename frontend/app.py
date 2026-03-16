import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="AutoML Web App", layout="centered")

st.title("🚀 AutoML Web App")
st.write("Upload your dataset and select the target column")

BACKEND_URL = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    target_column = st.selectbox(
        "Select target column",
        options=df.columns
    )

    if st.button("Upload Dataset"):

        with st.spinner("Training models..."):

            uploaded_file.seek(0)

            files = {"file": uploaded_file}

            data = {"target": target_column}

            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files,
                    data=data
                )
            except Exception as e:
                st.error(f"Backend connection failed: {e}")
                st.stop()

        if response.status_code == 200:

            result = response.json()

            st.success("Dataset uploaded successfully 🎉")

            st.subheader("📊 Model Comparison")

            results = result.get("model_results")

            if results:
                df_results = pd.DataFrame(results).T
                st.dataframe(df_results)

            best_model = result.get("best_model")

            if best_model:
                st.success(f"🏆 Best Model Selected: **{best_model}**")

            feature_importance = result.get("feature_importance")

            if feature_importance:

                st.subheader("📊 Feature Importance")

                features = df.columns.drop(target_column)

                feature_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": feature_importance
                })

                feature_df = feature_df.sort_values(by="Importance", ascending=False)

                st.bar_chart(feature_df.set_index("Feature"))

            dataset_id = result.get("dataset_id")

            if dataset_id:

                download_url = f"{BACKEND_URL}/download/{dataset_id}"

                st.subheader("📥 Download Best Model")

                st.markdown(f"[Download Trained Model]({download_url})")

        else:

            st.error("Upload failed ❌")

            try:
                st.json(response.json())
            except:
                st.write(response.text)