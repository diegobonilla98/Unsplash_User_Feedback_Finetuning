import pandas as pd
import numpy as np

df = pd.read_csv("unsplash_images_cleaned_user_caption_top_kw.csv")
split_values = np.random.choice(['train', 'test', 'val'], size=len(df), p=[0.7, 0.2, 0.1])
df['split'] = split_values
columns_to_keep = ['photo_id', 'keyword', 'user_keyword', 'saved_path', 'keyword_cleaned', 'user_keyword_cleaned',
                   'user_tags_caption', 'user_top_keywords', 'split']
df.drop(df.columns.difference(columns_to_keep), 1, inplace=True)

df.to_csv("unsplash_images_cleaned_user_caption_top_kw_split_redux.csv", index=False)
