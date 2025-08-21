import pandas as pd

# df = pd.read_json('data/yelp_academic_dataset_business.json', lines=True)[["categories","review_count","stars"]]

# df['categories'] = df['categories'].fillna('')
# all_categories = set()
# df['categories'].str.split(',').apply(lambda x: [cat.strip() for cat in x if cat.strip()]).apply(all_categories.update)

# for category in all_categories:
#     df[category] = df['categories'].apply(lambda x: int(category in [cat.strip() for cat in x.split(',')]))

# df = df.drop(columns=['categories'])

# print(df.info())
# df.to_csv('data/yelp_business.csv', index=False)

df=pd.read_csv('data/yelp_business.csv').info()
df = pd.read_csv('data/yelp_business.csv')
column_sums = df.sum().sort_values(ascending=False)
print(column_sums)
sorted_columns = column_sums.index
df = df[sorted_columns]
df.to_csv('data/yelp_business_sorted.csv', index=False)
