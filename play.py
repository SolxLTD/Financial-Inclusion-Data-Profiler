import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from faker import Faker
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport


# Title
st.set_page_config(page_title="Financial Inclusion App", layout="wide")
st.title("💰 Financial Inclusion Data Profiler & Synthetic Generator")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ---- Data Profiling ----
    with st.spinner("Generating data profile..."):
        profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
        html(profile.to_html(), height=800, scrolling=True)

    # ---- Data Quality Score ----
    st.subheader("🧮 Data Quality Score")
    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        synthetic = df.sample(n=min(200, len(df)))  # small mock
        quality_score = evaluate_quality(df, synthetic, metadata)
        st.success(f"✅ Data Quality Score: {quality_score['score']:.2f}")
    except Exception as e:
        st.warning(f"Could not compute quality score: {e}")

    # ---- Synthetic Data Generation ----
    st.subheader("🧬 Generate Synthetic Data")
    synth_button = st.button("Generate Synthetic Dataset")

    if synth_button:
        with st.spinner("Training CTGAN model... this may take a minute ⏳"):
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            synth = CTGANSynthesizer(metadata)
            synth.fit(df)
            synthetic_data = synth.sample(num_rows=len(df))
            st.success("✅ Synthetic data generated successfully!")
            st.dataframe(synthetic_data.head())

            # Download button
            csv = synthetic_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Synthetic CSV",
                data=csv,
                file_name="synthetic_financial_data.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Please upload a CSV file to begin.")