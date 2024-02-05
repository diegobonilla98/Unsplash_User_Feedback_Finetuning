import glob
import pandas as pd
from functools import reduce


def read_and_concat_datasets(path, documents):
    """
    Read multiple dataset files, concatenate them into dataframes, and return a dictionary of these dataframes.

    :param path: The directory path where the files are located.
    :param documents: A list of document names to be read.
    :return: A dictionary with document names as keys and concatenated DataFrames as values.
    """
    datasets = {}
    for doc in documents:
        files = glob.glob(f"{path}{doc}.tsv*")
        subsets = [pd.read_csv(filename, sep='\t', header=0) for filename in files]
        datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)
    return datasets


def process_datasets(datasets):
    """
    Process datasets by grouping keywords and ensuring all datasets have common photo IDs.

    :param datasets: A dictionary with document names as keys and DataFrames as values.
    :return: A list of processed DataFrames ready for merging.
    """
    datasets_list = list(datasets.values())
    # Group by 'photo_id' and aggregate 'keyword'
    datasets_list[1] = datasets_list[1].groupby('photo_id')['keyword'].agg(list).reset_index()
    datasets_list[2] = datasets_list[2].groupby('photo_id')['keyword'].agg(list).reset_index()
    datasets_list[2] = datasets_list[2].rename(columns={'keyword': 'user_keyword'})

    # Filter DataFrames to contain only common photo IDs
    common_ids = set(datasets_list[0]['photo_id'])
    for df in datasets_list[1:]:
        common_ids &= set(df['photo_id'])
    filtered_dfs = [df[df['photo_id'].isin(common_ids)] for df in datasets_list]

    return filtered_dfs


def merge_datasets(datasets_list):
    """
    Merge multiple datasets into a single DataFrame based on a common key ('photo_id').

    :param datasets_list: A list of DataFrames to be merged.
    :return: A single DataFrame resulting from the merging process.
    """
    return reduce(lambda left, right: pd.merge(left, right, on='photo_id'), datasets_list)


# Main execution
if __name__ == "__main__":
    path = 'Unsplash_lite\\'
    documents = ['photos', 'keywords', 'conversions']

    datasets = read_and_concat_datasets(path, documents)
    processed_datasets = process_datasets(datasets)
    merged_df = merge_datasets(processed_datasets)

    merged_df.to_csv("merged.csv", index=False)
