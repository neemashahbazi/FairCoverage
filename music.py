import pandas as pd
from collections import Counter

path = "data/itunes.csv"
# path = "data/amazon.csv"
df = pd.read_csv(path)
unique_genres = set(genre.strip() for genres in df["Genre"].dropna() for genre in genres.split(","))
print(unique_genres, len(unique_genres))
genre_counts = Counter(genre.strip() for genres in df["Genre"].dropna() for genre in genres.split(","))
sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
print(sorted_genres)