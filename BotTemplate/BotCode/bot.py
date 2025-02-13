import random
import datetime
import os
import re

from groq import Groq
from dotenv import load_dotenv

from abc_classes import ABot
from teams_classes import NewUser, NewPost

class Bot(ABot):
    def __init__(self, groq_model="gemma2-9b-it", groq_api_key="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"):
        self.min_posts = 10
        self.total_posts = 0

        self.groq_api_key = groq_api_key
        #load_dotenv()
        #self.groq_api_key = os.getenv("GROQ_API_KEY")

        if self.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set")
        self.groq_model = groq_model
        self.client = Groq(api_key=self.groq_api_key)

    def create_user(self, session_info):
        self.session_info = session_info
        #self.print_session_info()
        self.topics = self.parse_session_topics()
        self.post_average_words = self.session_info.metadata.get("users_average_amount_words_in_post")

        new_users = [
            NewUser(username="yuanbot", name="yuan", description="swe student")
        ]
        
        return new_users

    def generate_content(self, datasets_json, users_list):
        posts = []
        start_time = datetime.datetime.fromisoformat(self.session_info.start_time.replace("Z", "+00:00"))
        end_time = datetime.datetime.fromisoformat(self.session_info.end_time.replace("Z", "+00:00"))
        remaining_posts = max(0, self.min_posts - self.total_posts)
        num_posts = random.randint(1, remaining_posts) if remaining_posts > 0 else 1

        for _ in range(num_posts):
            user = random.choice(users_list)
            topic, keywords = random.choice(list(self.topics.items()))
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

    def generate_text(self, topic, keywords):
        keywords_str = ",".join(keywords) if keywords else ""
        prompt = f"Generate a short tweet-like post about {topic} using about {self.post_average_words} words. Use relevant keywords from {keywords_str} if appropriate."
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.groq_model,
            temperature=1,
            top_p=0.95,
            #reasoning_format="hidden"
        )

        return re.sub(r'[^\x00-\x7F]+', "", re.sub(r'^"|"$', "", chat_completion.choices[0].message.content.strip()))
    
    def parse_session_topics(self):
        return {t["topic"]: t["keywords"] for t in self.session_info.metadata.get("topics", [])}
    
    def print_session_info(self):
        print("Session info id:", self.session_info.session_id)
        print("Session info lang:", self.session_info.lang)
        print("Session info metadata:", self.session_info.metadata)
        print("Session info influence target:", self.session_info.influence_target)
        print("Session info start time:", self.session_info.start_time)
        print("Session info end time:", self.session_info.end_time)
