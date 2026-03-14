
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Literature Mining App", layout="wide")

st.title("Literature Mining Demo App")

st.write("Upload a CSV file to preview data.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data preview:")
    st.dataframe(df.head())

st.success("App running successfully!")
