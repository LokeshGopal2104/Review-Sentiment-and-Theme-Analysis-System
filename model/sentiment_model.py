import json
import os
import re
import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

# Ensure NLTK resources are available
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ==========================================
# TEXT PREPROCESSING FUNCTIONS
# ==========================================


def normalize_text(text):
    # 1. Changing text to lower case
    text = str(text).lower()

    # 2. Remove urls
    text = re.sub(r"https?://\s+|www\.\S+", "", text)

    # 3. Remove html tags
    text = re.sub(r"<.*>", "", text)

    # 4. Remove emojis and special characters (keep only a-z spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # 5. Remove extra whitespaces
    text = " ".join(text.split())

    return text


def remove_noise(word_list):
    clean_tokens = [re.sub(r"[^A-Za-z0-9]+", "", word) for word in word_list]
    return [word for word in clean_tokens if word]


def remove_stopwords(word_list):
    filtered_tokens = [word for word in word_list if word not in stop_words]
    return filtered_tokens


def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


# ==========================================
# THEME EXTRACTION LOGIC
# ==========================================


def extract_themes(reviews, top_n=20):
    tfidf = TfidfVectorizer(
        stop_words="english", ngram_range=(2, 3), min_df=2, max_df=0.8
    )

    X = tfidf.fit_transform(reviews)
    feature_names = tfidf.get_feature_names_out()
    scores = X.mean(axis=0).A1

    themes = pd.DataFrame({"theme": feature_names, "score": scores})
    themes = themes.sort_values(by="score", ascending=False)

    return themes.head(top_n)


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    # 1. LOAD AND CLEAN DATA
    print("--- Loading Dataset ---")
    df = pd.read_csv("./data/amazon_review_dataset.csv")

    print(f"Initial Shape: {df.shape}\n")
    df.info()
    print("\n")

    print(f"Duplicate count: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)

    print(f"Null values per column:\n{df.isna().sum()}\n")
    df.dropna(inplace=True)

    print(f"Sentiment counts:\n{df['sentiment'].value_counts()}\n")

    # 2. NLP TEXT PROCESSING
    print("--- Preprocessing text (this may take a minute) ---")
    df["review"] = df["review"].apply(normalize_text)
    df["review"] = df["review"].apply(word_tokenize)
    df["review"] = df["review"].apply(remove_noise)
    df["review"] = df["review"].apply(remove_stopwords)
    df["review"] = df["review"].apply(lemmatize_text)

    print("\nSample processed tokens:")
    print(df["review"].iloc[0:4])

    # Rejoin tokens into complete strings for vectorization
    df["review"] = df["review"].apply(lambda x: " ".join(x))

    # 3. FEATURE EXTRACTION (TF-IDF)
    x_text = df["review"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))
    x = vectorizer.fit_transform(x_text)

    print(f"\nShape of the vectorizer matrix: {x.shape}")
    print(f"Total number of features: {len(vectorizer.get_feature_names_out())}")

    # Save vectorizer transform asset
    joblib.dump(vectorizer, "./model/tfidf_vectorizer.pkl")

    # 4. TRAIN / TEST SPLIT
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set shape: {x_train.shape} | Test set shape: {x_test.shape}")

    # Class balance verifications
    print(
        f"Train Positive Ratio: {y_train.value_counts()['positive']/len(y_train):.2%}"
    )
    print(
        f"Train Negative Ratio: {y_train.value_counts()['negative']/len(y_train):.2%}"
    )
    print(
        f"Test Positive Ratio:  {y_test.value_counts()['positive']/len(y_test):.2%}"
    )
    print(
        f"Test Negative Ratio:  {y_test.value_counts()['negative']/len(y_test):.2%}\n"
    )

    # 5. MODEL TRAINING & CROSS-VALIDATION
    print("--- Training Logistic Regression Model ---")
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr_model.fit(x_train, y_train)

    cv_scores = cross_val_score(
        lr_model, x_train, y_train, cv=5, scoring="f1_weighted"
    )
    print("Fold Scores:", cv_scores)
    print("Average F1:", cv_scores.mean())
    print("Std Dev:", cv_scores.std())

    # Save model binary asset
    joblib.dump(lr_model, "./model/sentiment_model.pkl")

    # 6. EVALUATION METRICS
    y_pred = lr_model.predict(x_test)

    # Assigning predictions back to the dataframe safely for plotting later
    df.loc[y_test.index, "predicted_sentiment"] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cl_report = classification_report(y_test, y_pred)

    print(f"\nAccuracy :  {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall :    {recall:.4f}")
    print(f"F1 Score :  {f1:.4f}")
    print(f"\nClassification Report:\n{cl_report}")

    # 7. THEME EXTRACTION EXPORT DATASET SETUP
    # Filter reviews safely based on textual data columns
    positive_reviews = df[df["sentiment"] == "positive"]["review"]
    negative_reviews = df[df["sentiment"] == "negative"]["review"]

    print("\n--- Extracting Key Themes ---")
    positive_themes = extract_themes(positive_reviews, top_n=20)
    negative_themes = extract_themes(negative_reviews, top_n=20)

    print("\nTOP POSITIVE THEMES:\n", positive_themes.head())
    print("\nTOP NEGATIVE THEMES:\n", negative_themes.head())

    # Ensure output target directory exists or build locally
    positive_themes.to_csv("./data/positive_themes.csv", index=False)
    negative_themes.to_csv("./data/negative_themes.csv", index=False)
    print("Theme csv data files saved successfully.")

    results = {
        "Total Reviews": len(df),
        "Positive Reviews": len(positive_reviews),
        "Negative Reviews": len(negative_reviews),
    }
    print(f"\nDataset Overview: {results}")

    # 8. EXPORT GRAPHICAL PLOT IMAGES FOR STREAMLIT UI CONSUMPTION
    print("\n--- Generating Metric Distribution Plots ---")
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Sentiment Distribution
    plt.figure()
    df["predicted_sentiment"].value_counts().plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig("./plots/sentiment_distribution.png")
    plt.close()

    # Plot 2: Positive Themes
    plt.figure()
    positive_themes.head(10).plot(x="theme", y="score", kind="barh")
    plt.title("Top 10 Positive Themes")
    plt.tight_layout()
    plt.savefig("./plots/positive_themes.png")
    plt.close()

    # Plot 3: Negative Themes
    plt.figure()
    negative_themes.head(10).plot(x="theme", y="score", kind="barh")
    plt.title("Top 10 Negative Themes")
    plt.tight_layout()
    plt.savefig("./plots/negative_themes.png")
    plt.close()

    print("Dashboard visual images successfully compiled inside plots/ folder.")

    # 9. JSON EXPORT (Notice classification_report updated to dict representation)
    metrics_payload = {
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print("metrics.json file saved successfully. Execution complete!")