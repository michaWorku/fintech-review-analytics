# src/sentiment_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
#keyword Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

OUTPUT_PATH = "data/processed/review_sentiments.csv"
os.makedirs("data/processed", exist_ok=True)


# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text):
    """
    Preprocess review text: lowercasing, removing stopwords, lemmatization.
    """
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)


# -----------------------------
# Sentiment Analysis Methods
# -----------------------------
def get_textblob_sentiment(text):
    """ Use TextBlob sentiment analysis to classify text as positive, negative, or neutral."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def get_vader_sentiment(text):
    """ Use VADER sentiment analysis to classify text as positive, negative, or neutral."""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'


def get_transformer_sentiment(df, text_col="review"):
    """ Use a pre-trained transformer model for sentiment analysis.
    Adds 'transformer_sentiment' and 'transformer_score' columns to the DataFrame."""
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = classifier(df[text_col].tolist(), truncation=True, max_length=512)
    df["transformer_sentiment"] = [x["label"] for x in results]
    df["transformer_score"] = [x["score"] for x in results]
    return df


# -----------------------------
# Main Analysis Pipeline
# -----------------------------
def load_data(path):
    """ Load the cleaned reviews DataFrame from CSV."""
    return pd.read_csv(path)


def analyze_sentiment(df, text_col="review"):
    """ Analyze sentiment of reviews in the DataFrame."""
    df[text_col] = df[text_col].astype(str)
    df["cleaned_review"] = df[text_col].apply(preprocess_text)
    df["textblob_sentiment"] = df["cleaned_review"].apply(get_textblob_sentiment)
    df["vader_sentiment"] = df["cleaned_review"].apply(get_vader_sentiment)
    df = get_transformer_sentiment(df, text_col="cleaned_review")
    return df


def plot_sentiments(df):
    """
    Plot sentiment distributions for all sentiment analysis methods.
    """
    plt.figure(figsize=(18, 5))

    # TextBlob
    plt.subplot(1, 3, 1)
    sns.countplot(x='textblob_sentiment', data=df, palette='Set2')
    plt.title("TextBlob Sentiment Distribution")

    # VADER
    plt.subplot(1, 3, 2)
    sns.countplot(x='vader_sentiment', data=df, palette='Set1')
    plt.title("VADER Sentiment Distribution")

    # DistilBERT
    plt.subplot(1, 3, 3)
    sns.countplot(x='transformer_sentiment', data=df, palette='coolwarm')
    plt.title("DistilBERT Sentiment Distribution")

    plt.tight_layout()
    plt.show()


def save_results(df):
    """ Save the DataFrame with sentiment analysis results to CSV."""
    # Drop rows where cleaned_review is NaN or empty
    df = df[df['cleaned_review'].notna()]
    df = df[df['cleaned_review'].str.strip() != ""]
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Sentiment analysis results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    df = load_data("data/clean/cleaned_reviews.csv")
    df = analyze_sentiment(df)
    save_results(df)
