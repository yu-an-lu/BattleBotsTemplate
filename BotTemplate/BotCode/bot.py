import random
import datetime
import os
import re
import openai
import json
import math
import collections
import traceback
import emoji
import nltk
from nltk.corpus import stopwords

from pydantic import BaseModel
from dotenv import load_dotenv

from abc_classes import ABot
from teams_classes import NewUser, NewPost

class BotProfile(BaseModel):
    username: str
    name: str
    description: str
    location: str

class BotUsers(BaseModel):
    profiles: list[BotProfile]

class BotPost(BaseModel):
    text: str

class BotContent(BaseModel):
    posts: list[BotPost]

class Bot(ABot):
    def __init__(self, model="gpt-4o-2024-08-06"):
        self.bot_min_posts = 10
        self.bot_percentage = 0.02
        self.new_users = {}

        load_dotenv()
        self.api_key = os.getenv("ENV_VAR1")

        if self.api_key is None:
            raise ValueError("api key is not set")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def parse_user_profile_characteristics(self, users, top_n=100):
        nltk.download("stopwords")
        nltk.download('punkt_tab')
        stop_words = set(stopwords.words('english'))
        characteristics = {
            "username": {},
            "name": {},
            "description": {},
            "location": {}
        }

        for field in characteristics.keys():
            word_counts = []
            sentence_lengths = []
            punctuation_usage = collections.Counter()
            uppercase_ratio = []
            emoji_counts = collections.Counter()
            hashtag_counts = collections.Counter()
            word_freq = collections.Counter()

            for user in users:
                text = user.get(field)
                if not text:
                    continue
                    
                word_counts.append(len(text.split()))
                sentences = nltk.sent_tokenize(text)
                sentence_lengths.extend(len(sentence.split()) for sentence in sentences)
                punctuation_usage.update(re.findall(r"[.!?,;:]", text))
                uppercase_ratio.append(sum(1 for char in text if char.isupper()) / len(text) if len(text) > 0 else 0)
                emojis_in_text = [char for char in text if emoji.is_emoji(char)]
                emoji_counts.update(emojis_in_text)
                hashtags_in_text = [word for word in text.split() if word.startswith("#")]
                hashtag_counts.update(hashtags_in_text)
                tokens = nltk.word_tokenize(text)
                filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
                word_freq.update(filtered_words)

            def limit_to_top_n(counter, n=top_n):
                return {k: v for k, v in counter.most_common(n)}
            
            characteristics[field] = {
                "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
                "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
                "punctuation_usage": punctuation_usage,
                "avg_uppercase_ratio": sum(uppercase_ratio) / len(uppercase_ratio) if uppercase_ratio else 0,
                "top_emojis": limit_to_top_n(emoji_counts, top_n),
                "top_hashtags": limit_to_top_n(hashtag_counts, top_n),
                "word_freq": limit_to_top_n(word_freq, top_n)
            }

        # Save characteristics to a file for debugging
        #self.get_user_characteristics_json(characteristics)
        
        return characteristics


    def create_user(self, session_info):
        #self.get_session_info_json(session_info)
        self.session_info = session_info
        self.subsession_num = len(self.session_info.sub_sessions_id)

        # Determine num bots based on user count
        num_users = len(session_info.users)
        num_bots = math.ceil(num_users / (1 - self.bot_percentage) * self.bot_percentage)

        # Extract user profile characteristics
        user_characteristics = self.parse_user_profile_characteristics(session_info.users)

        user_profiles = self.generate_bot_profiles(num_bots, user_characteristics)
        print("Generated number of bots:", len(user_profiles))
        
        self.topics = self.parse_session_topics()

        users_average_posts = self.session_info.metadata.get("users_average_amount_posts")
        users_average_z_score = self.session_info.metadata.get("users_average_z_score")
        users_average_amount_words_in_post = self.session_info.metadata.get("users_average_amount_words_in_post")

        # Scale noise based on z-score: if variance is high, increase noise range
        z_score_adjustment = max(0.9, min(1.3, 1 + (abs(users_average_z_score) * 0.2)))

        for profile in user_profiles:
            noise_factor = random.uniform(0.9 * z_score_adjustment, 1.3 * z_score_adjustment)
            adjusted_posts = round(users_average_posts * noise_factor)
            adjusted_words = round(users_average_amount_words_in_post * noise_factor)
            self.new_users[profile.username] = {
                "user": NewUser(
                    username=profile.username,
                    name=profile.name,
                    description=profile.description,
                    location=profile.location
                ),
                "posts": [],
                "min_posts": max(adjusted_posts, self.bot_min_posts),
                "total_posts": 0,
                "posts_average_words": adjusted_words,
                "influence_target_used": False
            }
        
        #self.get_bot_users_info_json(new_users)
        return [user["user"] for user in self.new_users.values()]

    def generate_content(self, datasets_json, users_list):
        #self.get_sub_session_json(datasets_json)

        subsession_posts = []

        # Get subsession start and end time from session data
        subsession = next(
            (subsession for subsession in self.session_info.sub_sessions_info
             if subsession["sub_session_id"] == datasets_json.sub_session_id),
             None
        )

        start_time = datetime.datetime.fromisoformat(subsession["start_time"].replace("Z", "+00:00"))
        end_time = datetime.datetime.fromisoformat(subsession["end_time"].replace("Z", "+00:00"))
        
        remaining_subsessions = self.subsession_num - datasets_json.sub_session_id + 1

        for username, user_data in self.new_users.items():
            remaining_posts = user_data["min_posts"] - user_data["total_posts"]

            if remaining_subsessions > 1:
                max_posts = round(remaining_posts / remaining_subsessions * random.uniform(0.8, 1.2))
                num_posts = random.randint(0, min(max(1, max_posts), remaining_posts))
            else:
                num_posts = remaining_posts

            user_id = next(u for u in users_list if u.username == username).user_id

            # Select topics
            if datasets_json.sub_session_id == self.subsession_num and not user_data["influence_target_used"]:
                topics = [{"topic": self.session_info.influence_target["topic"], "keywords": self.session_info.influence_target["keywords"]}]
                user_data["influence_target_used"] = True
            else:
                topic_indices = random.sample(range(len(self.topics)), min(len(self.topics), random.randint(1, 3)))
                selected_topics = [list(self.topics.keys())[i] for i in topic_indices]
                topics = [{"topic": t, "keywords": self.topics[t]} for t in selected_topics]
                if any(t["topic"] == self.session_info.influence_target["topic"] for t in topics):
                    user_data["influence_target_used"] = True
            
            posts = self.generate_posts(username, topics, num_posts, user_data["posts_average_words"])
            user_data["total_posts"] += len(posts)

            for post in posts:
                time = start_time + (end_time - start_time) * random.random()
                text = post.text
                new_post = NewPost(
                    text=text,
                    author_id=user_id,
                    created_at=time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                )
                subsession_posts.append(new_post)
                user_data["posts"].append(new_post)

        return subsession_posts

    def generate_bot_profiles(self, num_bots, user_characteristics):
        prompt = json.dumps({
            "description": f"""
            You are trying to generate {num_bots} bot profiles to blend in as humans based on the profile characteristics.
            Try to generate profiles that use the same words and sentence structures as the characteristics, but with slight variations to avoid detection.
            Avoid using actual names and only cammel cases for the "username" and "name" fields, employ more twitter-like profiles.
            Be creative, do not use just give first and last names for the "name" field, use more creative names such as sentence-like names, underscores, weird capitalization, numbers, typos, memes etc.
            Restrict and limit the use of formal sentences and words to make the profiles look more natural.
            Try to add in typos, grammar errors, and other human-like mistakes to make the profiles look more human-like.
            When using emojis for name, can also put emojis in front of the name.
            """,
            "num_bots": num_bots, 
            "characteristics": user_characteristics
        })

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Generate {num_bots} bot profiles to blend in as humans based on the profile characteristics."
                }
            ],
            response_format=BotUsers
        )

        print("Generated bot profiles:\n", completion.choices[0].message.parsed.profiles)
        return completion.choices[0].message.parsed.profiles

    def generate_posts(self, username, topics, num_posts, post_average_words):
        prompt = json.dumps({
            "description": f"""
            Generate tweet-like posts on the given topics, each post is on one topic from the given topics. 
            Imitate human behavior by generating posts with medium variances of posts length to ultimately reach a given average of words per post.
            Vary the sentences structure and the choice of words to make the posts look more natural.
            Make sure that the generated posts are not too similar to the user's past posts to avoid detection.
            Try to add in typos, grammar errors, and other human-like mistakes to make the profiles look more human-like.
            """,
            "username": username,
            "topics": topics,
            "num_posts": num_posts,
            "average_words": post_average_words
        })
        
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Generate {num_posts} tweet-like posts on the given topics. Do not wrap the posts in quotes."
                }
            ],
            response_format=BotContent
        )

        print("Generated posts for user", username, ":\n", completion.choices[0].message.parsed.posts)
        return completion.choices[0].message.parsed.posts
    
    def parse_session_topics(self):
        return {t["topic"]: t["keywords"] for t in self.session_info.metadata.get("topics", [])}
    
    def get_bot_users_info_json(self, users_info):
        with open("bot_users_data.json", "w") as f:
            json.dump(
            [
                {
                    "username": user.username,
                    "name": user.name,
                    "description": user.description,
                    "location": user.location
                }
                for user in users_info
            ], f, indent=4)
    
    def get_session_info_json(self, session_info):
        with open("session_data.json", "w") as f:
            json.dump({
                "session_id": session_info.session_id,
                "lang": session_info.lang,
                "metadata": session_info.metadata,
                "influence_target": session_info.influence_target,
                "start_time": session_info.start_time,
                "end_time": session_info.end_time,
                "sub_sessions_info": session_info.sub_sessions_info,
                "sub_sessions_id": session_info.sub_sessions_id,
                "users": session_info.users,
                "usernames": list(session_info.usernames)
            }, f, indent=4)
        
    def get_sub_session_json(self, sub_session):
        with open("subsession_data.json", "w") as f:
            json.dump({
                "session_id": sub_session.session_id,
                "sub_session_id": sub_session.sub_session_id,
                "posts": sub_session.posts,
                "users": sub_session.users
            }, f, indent=4)

    def get_user_characteristics_json(self, user_characteristics):
        with open("user_characteristics_data.json", "w") as f:
            json.dump(user_characteristics, f, indent=4)
