import streamlit as st
import pandas as pd
import os
import json
import joblib
from model.sentiment_model import (
    normalize_text,
    remove_noise,
    remove_stopwords,
    lemmatize_text,
)

@st.cache_resource
def load_model():
    model = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

def predict_review(review):
    review_vector = tfidf_vectorizer.transform([review])
    return sentiment_model.predict(review_vector)[0]
sentiment_model, tfidf_vectorizer = load_model()


@st.cache_data
def metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)
    


load_metrics = metrics()
scalar_metrics = {
    "Accuracy":load_metrics['accuracy_score'],
    "Precision":load_metrics["precision_score"],
    "Recall":load_metrics['recall_score'],
    "F1_score":load_metrics['f1_score']
}


st.title("Review Sentiment Analysis System")

text = st.text_area("Enter Review text here ...")

text_submit = st.button("Analyze Review")

if text_submit:
    text = normalize_text(text)
    text = text.split()

    text = remove_noise(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    text = " ".join(text)

    x = tfidf_vectorizer.transform([text])

    predict = sentiment_model.predict(x)[0]

    st.write(f"Prediction: {predict}")

    # st.write(prediction)

# text  = text.apply(lambda x: " ".join(x))

df_positives = pd.read_csv('./data/positive_themes.csv')

st.header("Model predicted Top 20 positive Themes :")

st.dataframe(df_positives)

st.header("Model predicted Top 20 negative themes")

df_negatives = pd.read_csv('./data/negative_themes.csv')
st.dataframe(df_negatives)

st.divider()


metrics_df = pd.DataFrame.from_dict(scalar_metrics,orient="index",columns=['values'])
st.header("Model Metrics from the test Data")
st.dataframe(metrics_df)

st.divider()


PLOTS_DIR = 'plots'

# List all files in the plots folder
all_plots = os.listdir(PLOTS_DIR) 

# Create a dropdown selector in the UI
selected_plot = st.selectbox("Select a plot to view:", all_plots)

# Display the one they selected
full_plot_path = os.path.join(PLOTS_DIR, selected_plot)
st.image(full_plot_path, use_container_width=True)