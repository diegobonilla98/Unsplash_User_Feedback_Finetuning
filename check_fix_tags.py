import pandas as pd
import tqdm
import Levenshtein as lev


def calculate_similarity(string1, string2):
    """
    Calculate the similarity between two strings using Levenshtein distance.

    :param string1: The first string for comparison.
    :param string2: The second string for comparison.
    :return: The similarity percentage between the two strings.
    """
    distance = lev.distance(string1, string2)
    max_length = max(len(string1), len(string2))
    return (1 - distance / max_length) * 100


def filter_similar_strings(strings, threshold=70):
    """
    Filters out strings that are similar above a specified threshold.

    :param strings: List of strings to filter.
    :param threshold: Similarity threshold above which strings are considered duplicates.
    :return: A filtered list of strings.
    """
    filtered_strings = []
    for current_string in strings:
        if not filtered_strings:
            filtered_strings.append(current_string)
            continue
        is_duplicate = False
        for existing_string in filtered_strings:
            if calculate_similarity(current_string, existing_string) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_strings.append(current_string)
    return filtered_strings


def clean_keywords_in_dataframe(df, keyword_column="keyword", user_keyword_column="user_keyword", cleaned_keyword_column="keyword_cleaned", cleaned_user_keyword_column="user_keyword_cleaned"):
    """
    Cleans keywords in a pandas DataFrame by removing similar strings.

    :param df: The DataFrame containing keywords to clean.
    :param keyword_column: The name of the column containing keywords.
    :param user_keyword_column: The name of the column containing user keywords.
    :param cleaned_keyword_column: The name of the column to store cleaned keywords.
    :param cleaned_user_keyword_column: The name of the column to store cleaned user keywords.
    """
    df[cleaned_keyword_column] = None
    df[cleaned_user_keyword_column] = None
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        keywords = eval(row[keyword_column])
        user_keywords = eval(row[user_keyword_column])
        cleaned_keywords = filter_similar_strings(keywords)
        cleaned_user_keywords = filter_similar_strings(user_keywords)
        df.at[idx, cleaned_keyword_column] = cleaned_keywords
        df.at[idx, cleaned_user_keyword_column] = cleaned_user_keywords


if __name__ == "__main__":
    # load, clean, and save keyword data
    df = pd.read_csv("merged_paths.csv")
    clean_keywords_in_dataframe(df)
    df.to_csv("unsplash_images_cleaned.csv", index=False)
