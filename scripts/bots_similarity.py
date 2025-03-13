import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data(file_path):
    """Loads JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def group_users_by_team(bot_data):
    """Groups users by bot team name."""
    bot_teams = {}
    
    for user_id, user_info in bot_data.items():
        bot_team = user_info["bot_team_name"]
        if bot_team not in bot_teams:
            bot_teams[bot_team] = {}
        bot_teams[bot_team][user_id] = user_info["posts"]
    
    return bot_teams

def compute_cross_user_similarity(users_posts):
    """Computes similarity between different users' posts in the same bot team."""
    vectorizer = TfidfVectorizer(stop_words="english")
    
    user_ids = list(users_posts.keys())
    user_texts = [" ".join([post["text"] for post in users_posts[user]]) for user in user_ids]

    # If only one user, return full similarity
    if len(user_texts) < 2:
        return {user_ids[0]: {user_ids[0]: 1.0}}

    tfidf_matrix = vectorizer.fit_transform(user_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Store results in a dictionary
    similarity_dict = {user_ids[i]: {} for i in range(len(user_ids))}
    for i in range(len(user_ids)):
        for j in range(len(user_ids)):
            similarity_dict[user_ids[i]][user_ids[j]] = round(similarity_matrix[i, j], 4)

    return similarity_dict

def analyze_cross_user_similarity(bot_data):
    """Computes similarity between users in the same bot team."""
    bot_teams = group_users_by_team(bot_data)
    bot_team_similarities = {}

    for team_name, users_posts in bot_teams.items():
        bot_team_similarities[team_name] = compute_cross_user_similarity(users_posts)

    return bot_team_similarities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cross-user similarity within the same bot team.")
    parser.add_argument("input_file", type=str, help="Path to the bot posts JSON file.")

    args = parser.parse_args()

    # Load bot data
    bot_data = load_data(args.input_file)

    # Analyze cross-user similarity
    cross_user_similarity = analyze_cross_user_similarity(bot_data)

    input_file_base = os.path.splitext(args.input_file)[0]
    output_file = f"{input_file_base}_similarity.json"

    with open(output_file, "w") as f:
        json.dump(cross_user_similarity, f, indent=4)
