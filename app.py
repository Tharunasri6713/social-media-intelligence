from pathlib import Path
import streamlit as st
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
REPORTS = BASE / "reports"

st.set_page_config(
    page_title="Social Media Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Large-Scale Social Media Intelligence Platform")
st.caption("DS-03 Capstone | Synthetic demonstration data and YouTube API metadata")

st.warning(
    "Interpret results carefully: the 600-record development dataset is synthetic. "
    "The 50-record YouTube dataset contains video titles and descriptions, "
    "not viewer opinions or verified user interactions. Model outputs are predictions."
)

@st.cache_data
def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

youtube = load_csv(DATA / "youtube_sentiment_results.csv")
synthetic = load_csv(DATA / "transformer_sentiment_predictions.csv")
topics = load_csv(DATA / "youtube_topic_summary.csv")
network = load_csv(DATA / "youtube_network_metrics.csv")
misinfo = load_csv(
    REPORTS / "misinformation_evaluation_report.csv"
    if (REPORTS / "misinformation_evaluation_report.csv").exists()
    else BASE / "misinformation_evaluation_report.csv"
)
trend = load_csv(DATA / "youtube_sentiment_trends_by_year.csv")

source = st.sidebar.selectbox(
    "Sentiment dataset",
    ["YouTube API sample (50 records)", "Synthetic development sample"]
)

df = youtube if source.startswith("YouTube") else synthetic

if not df.empty:
    st.subheader("Sentiment overview")
    st.caption(f"Dataset selected: {source}")
    st.metric("Records analyzed", len(df))

    sentiment_col = next(
        (c for c in ["sentiment", "predicted_sentiment", "label", "prediction"]
         if c in df.columns), None
    )

    if sentiment_col:
        counts = df[sentiment_col].fillna("Unknown").astype(str).value_counts()
        left, right = st.columns([2, 3])
        with left:
            st.write("Sentiment counts")
            st.dataframe(
                counts.rename_axis("Sentiment").reset_index(name="Count"),
                use_container_width=True,
                hide_index=True
            )
        with right:
            st.bar_chart(counts)
    else:
        st.info("Sentiment column not detected. Preview the available data below.")

    with st.expander("View sentiment records"):
        st.dataframe(df, use_container_width=True)

    text_col = next(
        (c for c in ["text", "title", "combined_text", "description"]
         if c in df.columns), None
    )
    if text_col:
        query = st.text_input("Search text in the selected dataset")
        if query:
            matches = df[
                df[text_col].fillna("").astype(str).str.contains(
                    query, case=False, regex=False
                )
            ]
            st.write(f"Matching records: {len(matches)}")
            st.dataframe(matches, use_container_width=True)
else:
    st.info("No data found for this selection. Check the CSV path and columns.")

st.divider()
st.subheader("YouTube topic modeling")
if not topics.empty:
    st.dataframe(topics, use_container_width=True)
else:
    st.info("Topic summary CSV not found or empty.")

st.subheader("Sentiment trends over publication year")
if not trend.empty:
    st.dataframe(trend, use_container_width=True)
else:
    st.info("Trend CSV not found or empty.")

st.subheader("Video-topic network metrics")
if not network.empty:
    st.dataframe(network, use_container_width=True)
else:
    st.info("Network metrics CSV not found or empty.")

st.subheader("Illustrative misinformation baseline")
if not misinfo.empty:
    st.dataframe(misinfo, use_container_width=True)
else:
    st.info("Misinformation evaluation report not found or empty.")

st.caption(
    "Limitations: small samples, synthetic labels, English-focused sentiment "
    "model, and illustrative misinformation labels. This is a research "
    "demonstration, not a fact-checking or real-world influence detection system."
)
