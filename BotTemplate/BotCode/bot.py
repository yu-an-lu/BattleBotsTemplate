import random
import datetime

from abc_classes import ABot
from teams_classes import NewUser, NewPost

class Bot(ABot):
    def create_user(self, session_info):
        # todo logic
        # Example:
        new_users = [
            NewUser(username="yuanbot", name="yuan", description="swe student")
        ]
        # print("Session info id:", session_info.session_id)
        # print("Session info lang:", session_info.lang)
        # print("Session info metadata:", session_info.metadata)
        # print("Session info influence target:", session_info.influence_target)
        # print("Session info start time:", session_info.start_time)
        # print("Session info end time:", session_info.end_time)
        return new_users

    def generate_content(self, datasets_json, users_list):
        # todo logic
        # It needs to return json with the users and their description and the posts to be inserted.
        # Example:
        templates = [
            "Pandas are amazing!",
            "I love data science!",
            "I am not a bot",
            "I am a human",
            "Coffee and coding go hand in hand.",
            "Hi, swe student here."
        ]

        posts = []
        for j in range(len(users_list)):
            text = random.choice(templates)
            time = datetime.datetime.now(datetime.timezone.utc).replace(hour=random.randint(0, 23), minute=random.randint(0, 59), second=random.randint(0, 59), microsecond=0)
            posts.append(NewPost(text=text, author_id=users_list[j].user_id, created_at=time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),user=users_list[j]))
        return posts
