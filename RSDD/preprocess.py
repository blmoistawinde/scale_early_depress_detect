import re
import os
import html
import pickle
import json
from tqdm import tqdm

URL_REGEX = re.compile(
                r'(?i)http[s]?://(?:[a-zA-Z]|[0-9]|[#$%*-_;=?&@~.&+]|[!*,])+',
                re.IGNORECASE)
remove_url = lambda x: re.sub(URL_REGEX, "[URL]", x).strip()

def all_preprocess(text):
    text = text.strip()
    if text == "[removed]":
        return ""
    text = remove_url(text)
    if text == "[URL]":
        return ""
    text = html.unescape(text)
    return text

if __name__ == "__main__":
    print(all_preprocess("This is my all time favorite Monty python sketch. https://youtu.be/2K8_jgiNqUc"))
    print(all_preprocess("https://youtu.be/2K8_jgiNqUc"))
    print(all_preprocess("[removed]"))
    print(all_preprocess("&gt;Why did you move?"))
    os.makedirs("./processed/", exist_ok=True)
    for split in ["training", "testing", "validation"]:
        print(split)
        user_ids = []
        user_posts = []
        user_labels = []
        with open(split, encoding="utf-8") as f:
            for line in tqdm(f):
                json0 = json.loads(line)[0]
                id0 = json0["id"]
                label0 = int(json0["label"] == "depression")
                posts = []
                for pid, post in json0["posts"]:
                    post = all_preprocess(post)
                    if post != "":
                        posts.append(post)
                if len(posts) < 32:
                    continue
                user_ids.append(id0)
                user_labels.append(label0)
                user_posts.append(posts)
        print(len(user_posts), len(user_posts[0]))
        with open(f"./processed/{split}.pkl", "wb") as fo:
            pickle.dump([user_ids, user_posts, user_labels], fo)