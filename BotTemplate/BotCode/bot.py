import random
import datetime
import os
import re
import openai
import json
import math
import traceback

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

class Bot(ABot):
    def __init__(self, model="gpt-4o-2024-08-06"):
        self.bot_min_posts = 10
        self.total_posts = 0
        self.influence_target_used = False
        self.bot_percentage = 0.05
        self.bot_num = 1

        load_dotenv()
        self.api_key = os.getenv("ENV_VAR1")

        if self.api_key is None:
            raise ValueError("api key is not set")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)

    def create_user(self, session_info):
        #self.get_session_info_json(session_info)
        self.session_info = session_info
        self.subsession_num = len(self.session_info.sub_sessions_id)

        # Determine num bots based on user count
        #num_users = len(session_info.users)
        #num_bots = math.ceil(num_users / (1 - self.bot_percentage) * self.bot_percentage)

        bots = self.generate_bot_profiles(self.bot_num, session_info.users)

        self.topics = self.parse_session_topics()

        users_average_posts = self.session_info.metadata.get("users_average_amount_posts")
        users_average_z_score = self.session_info.metadata.get("users_average_z_score")

        # Scale noise based on z-score: if variance is high, increase noise range
        z_score_adjustment = max(0.9, min(1.3, 1 + (abs(users_average_z_score) * 0.2)))
        noise_factor = random.uniform(0.9 * z_score_adjustment, 1.3 * z_score_adjustment)
        adjusted_posts = round(users_average_posts * noise_factor)
       
        self.bot_min_posts = max(adjusted_posts, self.bot_min_posts)
        self.post_average_words = self.session_info.metadata.get("users_average_amount_words_in_post")

        new_users = [
            NewUser(
                username=profile.username, 
                name=profile.name, 
                description=profile.description, 
                location=profile.location
            ) for profile in bots.profiles
        ]
        
        #self.get_bot_users_info_json(new_users)
        return new_users

    def generate_content(self, datasets_json, users_list):
        #self.get_sub_session_json(datasets_json)

        posts = []

        # Get subsession start and end time from session data
        subsession = next(
            (subsession for subsession in self.session_info.sub_sessions_info
             if subsession["sub_session_id"] == datasets_json.sub_session_id),
             None
        )

        start_time = datetime.datetime.fromisoformat(subsession["start_time"].replace("Z", "+00:00"))
        end_time = datetime.datetime.fromisoformat(subsession["end_time"].replace("Z", "+00:00"))
        
        remaining_posts = max(0, self.bot_min_posts - self.total_posts)
        remaining_subsessions = self.subsession_num - datasets_json.sub_session_id + 1
        if remaining_subsessions > 1:
            max_posts = round(remaining_posts / remaining_subsessions * random.uniform(0.8, 1.2))
            num_posts = min(max(1, max_posts), remaining_posts)
        else:
            num_posts = remaining_posts

        for _ in range(num_posts):
            user = random.choice(users_list)

            # Select topics
            if datasets_json.sub_session_id == self.subsession_num and not self.influence_target_used:
                # Force influence target if it hasn't been used
                topic = self.session_info.influence_target["topic"]
                keywords = self.session_info.influence_target["keywords"]
                self.influence_target_used = True
            else:
                topic, keywords = random.choice(list(self.topics.items()))
                if (topic == self.session_info.influence_target["topic"]):
                    self.influence_target_used = True
            
            text = self.generate_text(topic, keywords)
            time = start_time + (end_time - start_time) * random.random()

            posts.append(NewPost(
                    text=text, 
                    author_id=user.user_id, 
                    created_at=time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    user=user
                ))
        
        self.total_posts += num_posts

        return posts

    def generate_bot_profiles(self, num_bots, user_data):
        prompt = json.dumps({
        "description": f"You are trying to generate {num_bots} bot profiles to blend in with the following user data.",
        "num_bots": num_bots, 
        "users": user_data
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
                    "content": f"Generate {num_bots} bot profiles to blend in with the user data."
                }
            ],
            response_format=BotUsers
        )

        return completion.choices[0].message.parsed

    def generate_text(self, topic, keywords):
        keywords_str = ",".join(keywords) if keywords else ""
        prompt = f"Generate a short tweet-like post about {topic} using about {self.post_average_words} words. Use relevant keywords from {keywords_str} if appropriate."
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            top_p=0.95,
        )

        return re.sub(r'^"|"$', "", completion.choices[0].message.content.strip())
    
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
