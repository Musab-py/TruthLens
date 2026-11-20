
import streamlit as st

from src.truthlens.pipeline import TruthLensPipeline


# Page configuration
st.set_page_config(
    page_title="TruthLens",
    page_icon="🔎",
    layout="centered"
)


# Application title
st.title("🔎 TruthLens")
st.subheader("AI-Powered Fake News Detection")

st.write(
    "Analyze news articles using a Machine Learning model."
)


# Load the model
@st.cache_resource
def load_pipeline():
    return TruthLensPipeline()


truthlens = load_pipeline()


# User input
news = st.text_area(
    "Enter your news article:",
    height=200,
    placeholder="Paste a news article here..."
)


# Analyze button
if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        result, probability = truthlens.predict_news(news)

        st.divider()

        st.subheader("Analysis Result")

        if result == "REAL":
            st.success("Prediction: REAL")
        else:
            st.error("Prediction: FAKE")

        st.metric(
            "Model Probability",
            f"{probability:.2f}%"
        )

        st.info(
            "This is a machine learning prediction, "
            "not independent verification of the news. "
            "Accuracy is currently limited by the small "
            "training dataset."
        )
