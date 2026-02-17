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
        with st.spinner("Uploading dataset..."):
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
            try:
                result = response.json()
            except Exception:
                st.error("Backend returned invalid response")
                st.stop()

            if not result:
                st.error("Empty response from backend")
                st.stop()

            st.success("Dataset uploaded successfully 🎉")
            st.json(result)

            dataset_id = result.get("dataset_id")

            if dataset_id:
                download_url = f"{BACKEND_URL}/download/{dataset_id}"

                st.markdown("### 📥 Download Best Model")
                st.markdown(f"[Click here to download model]({download_url})")
            else:
                st.warning("Model not available for download.")

        else:
            st.error("Upload failed ❌")
            try:
                st.json(response.json())
            except:
                st.write(response.text)
