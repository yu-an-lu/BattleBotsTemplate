import random
import datetime
import os
import openai
import json
import pandas as pd
import numpy as np

from pydantic import BaseModel
from dotenv import load_dotenv

from abc_classes import ABot
from teams_classes import NewUser, NewPost

class BotProfile(BaseModel):
    username: str
    name: str
    description: str
    location: str
    tweet_count: int
    distribution: str

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
        self.bots = {
            "low": {},
            "middle": {},
            "high": {}
        }

        load_dotenv()
        self.api_key = os.getenv("ENV_VAR1")

        if self.api_key is None:
            raise ValueError("api key is not set")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def clean_text(self, text):
        if text is None:
            return ""
        text = text.replace("\x00", "")  # remove null bytes
        return text.strip()

    def sample_user_profiles(self, users):
        df = pd.DataFrame(users)
        df_sorted = df.sort_values(by="z_score")

        # 15 examples: 5 lowest, 5 highest, 5 middle
        low_sample = df_sorted.head(5)
        high_sample = df_sorted.tail(5)
        q25, q75 = np.percentile(df_sorted["z_score"], [25, 75])
        middle = df_sorted[(df_sorted["z_score"] >= q25) & (df_sorted["z_score"] <= q75)]
        middle_sample = middle.sample(10, random_state=42)\
        
        cols = ["username", "name", "description", "location", "tweet_count"]

        def group_data(df):
            return {col: df[col].tolist() for col in cols}

        final_sample = {
            "low": group_data(low_sample),
            "middle": group_data(middle_sample),
            "high": group_data(high_sample)
        }

        return final_sample

    def create_user(self, session_info):
        self.session_info = session_info
        #self.get_session_info_json(session_info)

        self.subsession_num = len(self.session_info.sub_sessions_id)

        num_bots = 6

        user_sample = self.sample_user_profiles(session_info.users)

        user_profiles = self.generate_bot_profiles(num_bots, user_sample)
        print("Generated number of bots:", len(user_profiles))
        
        self.topics = self.parse_session_topics()

        new_users = []

        for profile in user_profiles:
            user = NewUser(
                    username=profile.username,
                    name=self.clean_text(profile.name),
                    description=self.clean_text(profile.description),
                    location=self.clean_text(profile.location)
                )
            
            new_users.append(user)

            user_data = {
                "user": user,
                "posts": [],
                "min_posts": min(random.randint(30, 40), max(profile.tweet_count, random.randint(self.bot_min_posts, self.bot_min_posts + 10))),
                "total_posts": 0,
                "influence_target_used": False
            }

            if profile.distribution in self.bots:
                self.bots[profile.distribution][profile.username] = user_data
            else:
                raise ValueError(f"Unexpected distribution category: {profile.distribution}")
        
        #self.get_bot_users_info_json(self.bots, user_sample)
        return new_users

    def generate_content(self, datasets_json, users_list):
        #self.get_sub_session_json(datasets_json)

        subsession_posts = []
        generated_posts = {}

        # subsession start and end time from session data
        subsession = next(
            (subsession for subsession in self.session_info.sub_sessions_info
             if subsession["sub_session_id"] == datasets_json.sub_session_id),
             None
        )

        start_time = datetime.datetime.fromisoformat(subsession["start_time"].replace("Z", "+00:00"))
        end_time = datetime.datetime.fromisoformat(subsession["end_time"].replace("Z", "+00:00"))

        remaining_subsessions = self.subsession_num - datasets_json.sub_session_id + 1

        for distribution, users in self.bots.items():

            for username, user_data in users.items():

                remaining_posts = user_data["min_posts"] - user_data["total_posts"]
                num_posts = remaining_posts

                if remaining_subsessions > 1:
                    max_posts = round(remaining_posts / remaining_subsessions * random.uniform(0.8, 1.2))
                    num_posts = random.randint(1, min(max(1, max_posts), remaining_posts))

                user_id = next(u for u in users_list if u.username == username).user_id

                # select topics
                if datasets_json.sub_session_id == self.subsession_num and not user_data["influence_target_used"]:
                    topics = [{"topic": self.session_info.influence_target["topic"], "keywords": self.session_info.influence_target["keywords"]}]
                    user_data["influence_target_used"] = True
                else:
                    topic_indices = random.sample(range(len(self.topics)), min(len(self.topics), random.randint(1, 3)))
                    selected_topics = [list(self.topics.keys())[i] for i in topic_indices]
                    topics = [{"topic": t, "keywords": self.topics[t]} for t in selected_topics]
                    if any(t["topic"] == self.session_info.influence_target["topic"] for t in topics):
                        user_data["influence_target_used"] = True

                timestamps = []

                for _ in range(num_posts):
                    time = start_time + datetime.timedelta(seconds=random.uniform(0, (end_time - start_time).total_seconds()))
                    timestamps.append(time.strftime('%Y-%m-%dT%H:%M:%S.000Z'))

                post_samples = []
                total_posts = len(datasets_json.posts)

                for time in timestamps:
                    # find the closest index in subsession data
                    closest_index = min(
                        range(total_posts), 
                        key=lambda i: abs(
                            datetime.datetime.strptime(datasets_json.posts[i]["created_at"], "%Y-%m-%dT%H:%M:%S.000Z") - 
                            datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.000Z")))
                    
                    # 3 posts before and 3 after
                    start_index = max(0, closest_index - 3)
                    end_index = min(total_posts, closest_index + 4)
                    
                    extracted_posts = [post["text"] for post in datasets_json.posts[start_index:end_index]]

                    post_samples.append(extracted_posts)

                posts = []
                for i in range(0, num_posts, 5):
                    batch_size = min(5, num_posts - i)

                    batch_generated_posts = self.generate_posts(
                        user_data["user"],
                        topics,
                        batch_size,
                        post_samples[i : i + batch_size]
                        )
                
                    user_data["total_posts"] += len(batch_generated_posts)
                    posts.extend(batch_generated_posts)
                
                user_generated_posts = []

                # if (len(posts) != num_posts):
                #     raise ValueError(f"Number of generated posts ({len(posts)}) does not match the expected number of posts ({num_posts}) for user {username} in subsession {datasets_json.sub_session_id}")

                for i in range(len(posts)):
                    if i < len(timestamps):
                        time = timestamps[i]
                    else:
                        time = start_time + datetime.timedelta(seconds=random.uniform(0, (end_time - start_time).total_seconds()))
                        time = time.strftime('%Y-%m-%dT%H:%M:%S.000Z') 

                    text = self.clean_text(posts[i].text)

                    new_post = NewPost(
                        text=text,
                        author_id=user_id,
                        created_at=time
                    )
                    
                    subsession_posts.append(new_post)
                    user_data["posts"].append(new_post)

                    user_generated_posts.append({
                        "text": text,
                        "created_at": time,
                        "sample_posts": post_samples[i]
                    })
                
                generated_posts[username] = user_generated_posts

        #self.get_generated_posts_json(generated_posts, datasets_json.sub_session_id)

        return subsession_posts

    def generate_bot_profiles(self, num_bots, user_sample):
        prompt = json.dumps({
            "instructions": f"""
            You are trying to generate {num_bots} twitter bot profiles to blend in as humans based on the user sample.
            You are provided with a list of sample user profiles for each distribution of users: low, middle, and high z-scores.
            Generate a balanced number of bot profiles for each distribution.
            All generated profiles should be unique and blend in the human profiles of the corresponding distribution.
            Important:
            - Username, name, description, and location do not have to make sense.
            - Name does not have to be a real name with first and last name. It can be a single word or a combination of words.
            - Description can be a single word or a combination of words.
            - Location can be a single word or a combination of words.
            Similar means:
            - Similar to the majority of the sample profiles.
            - Similar vocabulary, and tone.
            - Similar emojis and links if present.
            For each user profile:
            - Create profiles by using the same words or synonyms from the sample profiles provided for the distribution.
            Username:
            - Should be of the same word structure as the majority of the sample usernames provided for the distribution.
            Name: 
            - Should be in the same language as the majority of the sample names provided for the distribution.
            - Should be of the same word structure, i.e. camel case, capitalization, underscore, space, emojis, as the sample names provided for the distribution.
            Description:
            - Should be in the same language as the majority of the sample descriptions provided for the distribution.
            - Should be of a similar way of saying things, i.e. sentence structure, writing style, and length to the sample descriptions provided for the distribution.
            Location:
            - Should be null if the majority of the sample locations provided for the distribution are null.
            - Should be of the same word structure, i.e. camel case, capitalization, underscore, space, emojis, as the sample locations provided for the distribution.
            Tweet count:
            - Should be similar to the sample tweet counts provided for the distribution.
            Distribution:
            - Should be indicated as low, middle, or high.
            """,
            "num_bots": num_bots,
            "sample": user_sample
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
                    "content": f"Generate {num_bots} bot profiles to blend in as humans based on the user sample."
                }
            ],
            response_format=BotUsers
        )

        #print("Generated bot profiles:\n", completion.choices[0].message.parsed.profiles)
        return completion.choices[0].message.parsed.profiles

    def generate_posts(self, user, topics, num_posts, sample_posts):
        prompt = json.dumps({
            "instructions": f"""
            You are trying to generate {num_posts} tweet-like posts for the twitter user {user.username}.
            You are provided with a list of sample posts for each post to generate.
            All generated posts should be unique and blend in the human tweets.
            All generated posts should be in the language of the session {self.session_info.lang}.
            Text-wise:
            - Each post should have a similar tone, writing style, sentence structure as the list of sample posts provided for it.
            - Each post should start capitalized or uncapitalized if the majority of the sample posts provided for it start capitalized or uncapitalized.
            - Each post should only use emojis if the majority of the sample posts provided for it have emojis.
            - Each post should only refer to links (https://t.co/twitter_link) if the majority of the sample posts provided for it have links.
            Content-wise:
            - 90% of the posts should have the same content as a random sample post provided for it, but adjusted to the personality of the user.
            - 10% of the posts should be on a topic from the provided topics.
            Similar means:
            - Similar to the majority of the sample posts.
            - Similar words, vocabulary, and tone.
            - Similar way of saying things: sentence structure, writing style, and length.
            - Similar emojis and links if present.
            - Similar start of the post. Capitalized or not, punctuation, etc.
            - Similar places of capitalization, no capitalization, and punctuation.
            Do not wrap the posts in quotes.
            """,
            "language": self.session_info.lang,
            "user": user.username,
            "user_name": user.name,
            "user_description": user.description,
            "user_location": user.location, 
            "num_posts": num_posts,
            "topics": topics,
            "sample_posts": sample_posts
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
                    "content": f"""
                    Generate {num_posts} tweet-like posts for the twitter user {user.username}.
                    """
                }
            ],
            response_format=BotContent
        )

        #print("Generated posts for user", username, ":\n", completion.choices[0].message.parsed.posts)
        return completion.choices[0].message.parsed.posts
    
    def parse_session_topics(self):
        return {t["topic"]: t["keywords"] for t in self.session_info.metadata.get("topics", [])}
    
    def get_bot_users_info_json(self, users_data, user_sample):
        def serialize_user_data(user_data):
            return {
                "user": {
                    "username": user_data["user"].username,
                    "name": user_data["user"].name,
                    "description": user_data["user"].description,
                    "location": user_data["user"].location
                },
                "posts": user_data["posts"],
                "min_posts": user_data["min_posts"],
                "total_posts": user_data["total_posts"],
                "influence_target_used": user_data["influence_target_used"]
            }
        
        distribution_data = {}
    
        for distribution, users in users_data.items():
            distribution_data[distribution] = {
                "users_data": [serialize_user_data(user_data) for user_data in users.values()],
                "users_sample": user_sample.get(distribution, {})
            }
        
        with open(f"data/session{self.session_info.session_id}_bot_users_data.json", "w", encoding="utf-8") as f:
            json.dump(distribution_data, f, indent=4, ensure_ascii=False)
    
    def get_session_info_json(self, session_info):
        with open(f"data/session{self.session_info.session_id}_session_data.json", "w") as f:
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
            }, f, indent=4, ensure_ascii=False)
        
    def get_sub_session_json(self, sub_session):
        with open(f"data/session{self.session_info.session_id}_subsession{sub_session.sub_session_id}_data.json", "w") as f:
            json.dump({
                "session_id": sub_session.session_id,
                "sub_session_id": sub_session.sub_session_id,
                "posts": sub_session.posts,
                "users": sub_session.users
            }, f, indent=4, ensure_ascii=False)

    def get_user_characteristics_json(self, user_characteristics):
        with open(f"data/session{self.session_info.session_id}_user_characteristics_data.json", "w") as f:
            json.dump(user_characteristics, f, indent=4, ensure_ascii=False)

    def get_generated_posts_json(self, generated_posts, subsession_num):
        with open(f"data/session{self.session_info.session_id}_subsession{subsession_num}_posts_data.json", "w") as f:
            json.dump(generated_posts, f, indent=4, ensure_ascii=False)
