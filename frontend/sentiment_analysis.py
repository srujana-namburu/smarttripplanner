import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from django.conf import settings

# Download NLTK data
nltk.download('vader_lexicon')

# Load dataset (Ensure the correct path)
DATASET_PATH = os.path.join(settings.BASE_DIR, "tourist_review.csv")
df = pd.read_csv(DATASET_PATH)

# Fill missing values
df['reviews'] = df['review'].fillna("")

# Clean text function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.lower().strip()
    return ""

df['cleaned_reviews'] = df['reviews'].apply(clean_text)

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['cleaned_reviews'].apply(analyze_sentiment)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_reviews'])

# Mapping sentiment to numerical values
sentiment_map = {"Positive": 1, "Negative": 0, "Neutral": 2}
y = df['sentiment'].map(sentiment_map)

# Train ML Model
model = MultinomialNB()
model.fit(X, y)

# Function to analyze a specific location
def analyze_location(location):
    location_df = df[df['location'].str.lower() == location.lower()].copy()

    if location_df.empty:
        return {
            "message": f"No reviews available for {location}.",
            "summary": "No reviews available.",
            "wordcloud": None,
            "predictions": None
        }

    # Extract reviews
    reviews = location_df['cleaned_reviews']
    X_location = vectorizer.transform(reviews)

    # Predict sentiment
    predicted_sentiments = model.predict(X_location)
    reverse_sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"}
    location_df['predicted_sentiment'] = [reverse_sentiment_map[s] for s in predicted_sentiments]

    # Count sentiment distribution
    sentiment_counts = location_df['predicted_sentiment'].value_counts()

    # Summarize top words
    def get_top_words(text_series):
        words = " ".join(text_series).split()
        common_words = [word for word, _ in Counter(words).most_common(5)]
        return ", ".join(common_words) if common_words else "no specific keywords"

    positive_summary = get_top_words(location_df[location_df['predicted_sentiment'] == "Positive"]['cleaned_reviews'])
    negative_summary = get_top_words(location_df[location_df['predicted_sentiment'] == "Negative"]['cleaned_reviews'])

    summary = f"{location} is known for {positive_summary}. However, some visitors mentioned issues like {negative_summary}."

    # Generate Word Cloud
    wordcloud = None
    all_reviews_text = " ".join(reviews)
    if all_reviews_text:
        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(all_reviews_text)

    return {
        "message": f"Sentiment Analysis for {location}",
        "summary": summary,
        "wordcloud": wordcloud,
        "predictions": sentiment_counts
    }
