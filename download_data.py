import os
import urllib.request
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor


df = pd.read_csv("merged.csv")


def download_image(row):
    """
    Download an image from the URL specified in the 'photo_url' column of the row,
    save it to the 'images' directory, and return the save path.
    """
    url = row['photo_image_url'] + "?w=640"
    image_id = row['photo_id']
    try:
        save_path = "Unsplash_lite\\images\\" + image_id + ".jpg"
        urllib.request.urlretrieve(url, save_path)
        return save_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return pd.NA  # Use pd.NA for missing values in pandas


with ThreadPoolExecutor(max_workers=8) as executor:
    progress_bar = tqdm(total=len(df), desc="Downloading images")
    futures = [executor.submit(download_image, row) for _, row in df.iterrows()]
    saved_paths = []
    for future in futures:
        saved_paths.append(future.result())
        progress_bar.update(1)
    progress_bar.close()

df['saved_path'] = saved_paths
df.to_csv("merged_paths.csv", index=False)

