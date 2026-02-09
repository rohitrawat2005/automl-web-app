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

            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                data=data
            )

        if response.status_code == 200:
            st.success("Dataset uploaded successfully 🎉")
            st.json(response.json())
        else:
            st.error("Upload failed ❌")
            st.json(response.json())
