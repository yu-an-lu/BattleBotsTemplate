import json
import argparse
import os

def extract_bot_posts(input_file, bot_team_name):
    with open(input_file, "r") as f:
        data = json.load(f)

    posts = data["posts"]
    users = data["users"]

    bot_posts_dict = {}

    # Identify bot users
    bot_users = {user["user_id"]: user for user in users if user["bot_team_name"] == bot_team_name}

    # Iterate through posts and match them with bot users
    for post in posts:
        author_id = post["author_id"]
        if author_id in bot_users:
            bot_info = bot_users[author_id]
            
            if author_id not in bot_posts_dict:
                bot_posts_dict[author_id] = {
                    "bot_team_name": bot_team_name,
                    "user_id": author_id,
                    "username": bot_info["username"],
                    "name": bot_info["name"],
                    "description": bot_info["description"],
                    "location": bot_info["location"],
                    "detectors": bot_info["detectors"],
                    "posts": []
                }
            
            bot_posts_dict[author_id]["posts"].append(post)

    return bot_posts_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("--bot_team_name", type=str, default="Not A Bot", help="Name of the bot team")
   
    args = parser.parse_args()

    bot_posts = extract_bot_posts(args.input_file, args.bot_team_name)

    input_file_base = os.path.splitext(args.input_file)[0]
    output_file = f"{input_file_base}_{args.bot_team_name}.json"

    with open(output_file, "w") as f:
        json.dump(bot_posts, f, indent=4)
