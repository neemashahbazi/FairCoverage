import pandas as pd
import re


def time_to_seconds(time_str):
    time_str = str(time_str).replace(" ", "")
    match = re.match(r"^(?:(\d+):)?(\d+):(\d+)$", time_str)
    if match:
        groups = match.groups("0")
        hours = int(groups[0]) if groups[0] else 0
        minutes = int(groups[1])
        seconds = int(groups[2])
        return hours * 3600 + minutes * 60 + seconds
    match = re.match(r"^(\d+):(\d+)$", time_str)
    if match:
        minutes, seconds = match.groups()
        return int(minutes) * 60 + int(seconds)
    return None


df = pd.read_csv("data/amazon.csv")
df_genre = df["Genre"]
all_genres = set()
df_genre.apply(lambda x: [all_genres.add(g.strip()) for g in str(x).split(",")])
for genre in all_genres:
    df[genre] = df_genre.apply(lambda x: int(genre in [g.strip() for g in str(x).split(",")]))

df["weight"] = df["Time"].apply(time_to_seconds)
df["weight"] = df["weight"].fillna(1000).astype(int)
df.to_csv("data/amazon_with_genres.csv", index=False)
