# Review-Sentiment-and-Theme-Analysis-System


## Overview

Review Sentiment Intelligence Platform is an end-to-end Machine Learning application that analyzes customer reviews, predicts sentiment, extracts key positive and negative themes, and presents actionable insights through an interactive Streamlit dashboard.

The project was developed to move beyond simple sentiment classification by providing business-oriented insights from customer feedback.

---

## Problem Statement

Businesses receive thousands of customer reviews across products and services. Manually analyzing these reviews is time-consuming and inefficient.

This project aims to:

* Automatically classify reviews as Positive or Negative.
* Identify recurring positive themes.
* Identify recurring negative themes.
* Visualize insights through an interactive dashboard.

---

## Features

### Sentiment Classification

* TF-IDF Vectorization
* Logistic Regression Classifier
* Real-time review sentiment prediction

### Theme Extraction

* Top Positive Themes
* Top Negative Themes
* N-gram based TF-IDF extraction

### Dashboard

* Review Sentiment Prediction
* Model Performance Metrics
* Positive Theme Analysis
* Negative Theme Analysis
* Interactive Visualizations

---

## Dataset

Amazon Customer Reviews Dataset

The dataset contains:

* Review Text
* Sentiment Labels
* Product Reviews

---

## Project Pipeline

Data Collection

↓

Data Cleaning

↓

Text Preprocessing

↓

TF-IDF Vectorization

↓

Logistic Regression

↓

Model Evaluation

↓

Theme Extraction

↓

Streamlit Dashboard

---

## Text Preprocessing

The following preprocessing techniques were applied:

* Lowercasing
* URL Removal
* HTML Tag Removal
* Special Character Removal
* Tokenization
* Stopword Removal
* Lemmatization

---

## Machine Learning Models

### Production Model

TF-IDF + Logistic Regression

Reasons for selection:

* Strong performance on imbalanced data
* Better negative review detection
* Fast inference
* Lightweight deployment

### Experimental Model

LSTM Neural Network

Implemented and evaluated for comparison purposes.

The LSTM model was trained using:

* Tokenization
* Padding
* Embedding Layer
* LSTM Layer
* Class Weight Balancing
* Early Stopping

After evaluation, Logistic Regression achieved better minority-class performance and was selected for deployment.

---

## Model Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Cross Validation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow/Keras
* NLTK
* Matplotlib
* Joblib
* Streamlit

---

## Project Structure

Review-Sentiment-Intelligence/

├── app.py

├── model/

│ ├── sentiment_model.pkl

│ ├── tfidf_vectorizer.pkl

│ └── sentiment_model.py

├── data/

│ ├── positive_themes.csv

│ └── negative_themes.csv

├── plots/

│ ├── sentiment_distribution.png

│ ├── positive_themes.png

│ └── negative_themes.png

├── metrics.json

├── requirements.txt

└── README.md

---

## Future Enhancements

* Aspect-Based Sentiment Analysis
* Transformer Models (BERT)
* Review Summarization
* Multi-Class Sentiment Classification
* Real-Time Review Monitoring

---

## Author

Lokesh Gopal Meka

Aspiring Data Scientist | Machine Learning Enthusiast | Full Stack Learner
