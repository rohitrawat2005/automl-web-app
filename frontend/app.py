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

target_column = st.text_input("Enter target column name")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Upload Dataset"):
     with st.spinner("Uploading dataset..."):
        uploaded_file.seek(0)  # 🔥 THIS IS THE FIX

        files = {
            "file": uploaded_file
        }
        data = {
            "target": target_column
        }

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

