import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def load_data(file_path):
    """Loads JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def tokenize_text(text):
    """Tokenizes a given text into words."""
    return nltk.word_tokenize(text.lower())

def compute_lexical_richness(posts):
    """Computes lexical richness (TTR) and hapax legomena ratio."""
    words = [word for post in posts for word in tokenize_text(post["text"])]
    unique_words = set(words)
    hapax_legomena = [word for word, count in Counter(words).items() if count == 1]
    
    lexical_richness = len(unique_words) / len(words) if words else 0
    hapax_ratio = len(hapax_legomena) / len(words) if words else 0
    
    return lexical_richness, hapax_ratio

def compute_sentence_stats(posts):
    """Computes average sentence length."""
    sentence_lengths = [len(nltk.word_tokenize(post["text"])) for post in posts]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    return avg_sentence_length

def compute_post_variation(posts):
    """Computes cosine similarity between posts to check variation."""
    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [post["text"] for post in posts]
    
    if len(texts) < 2:
        return 1.0  # If only one post, assume full similarity (no variation)
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    return avg_similarity

def compute_posting_frequency(posts):
    """Computes the average time interval between posts."""
    timestamps = [datetime.fromisoformat(post["created_at"].replace("Z", "+00:00")) for post in posts]
    timestamps.sort()
    
    if len(timestamps) < 2:
        return 0  # No frequency data for single post
    
    intervals = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
    avg_posting_interval = np.mean(intervals)
    
    return avg_posting_interval

def compute_sentiment_variation(posts):
    """Computes average sentiment score of posts."""
    sentiments = [sia.polarity_scores(post["text"])["compound"] for post in posts]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    sentiment_variation = np.std(sentiments) if sentiments else 0
    return avg_sentiment, sentiment_variation

def compute_hashtag_repetition(posts):
    """Counts hashtag usage in posts."""
    hashtags = [word for post in posts for word in tokenize_text(post["text"]) if word.startswith("#")]
    most_common_hashtags = Counter(hashtags).most_common(5)
    return most_common_hashtags

def analyze_bots(bot_data):
    """Analyzes lexical richness, sentence structure, and variation for each bot."""
    bot_metrics = {}

    for bot_id, bot_info in bot_data.items():
        posts = bot_info.get("posts", [])
        lexical_richness, hapax_ratio = compute_lexical_richness(posts)
        avg_sentence_length = compute_sentence_stats(posts)
        avg_similarity = compute_post_variation(posts)
        avg_posting_interval = compute_posting_frequency(posts)
        avg_sentiment, sentiment_variation = compute_sentiment_variation(posts)
        common_hashtags = compute_hashtag_repetition(posts)

        bot_metrics[bot_id] = {
            "bot_team_name": bot_info["bot_team_name"],
            "lexical_richness": lexical_richness,
            "hapax_ratio": hapax_ratio,
            "avg_sentence_length": avg_sentence_length,
            "post_similarity": avg_similarity,
            "avg_posting_interval_sec": avg_posting_interval,
            "avg_sentiment": avg_sentiment,
            "sentiment_variation": sentiment_variation,
            "top_hashtags": common_hashtags
        }

    return bot_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze bot performance from posts.")
    parser.add_argument("input_file", type=str, help="Path to the bot posts JSON file.")

    args = parser.parse_args()

    # Load bot data
    bot_data = load_data(args.input_file)

    # Analyze bots
    bot_metrics = analyze_bots(bot_data)

    input_file_base = os.path.splitext(args.input_file)[0]
    output_file = f"{input_file_base}_metrics.json"

    with open(output_file, "w") as f:
        json.dump(bot_metrics, f, indent=4)
