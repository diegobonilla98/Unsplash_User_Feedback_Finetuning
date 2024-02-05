import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(embeddings_path, user_embeddings_path, csv_path):
    """
    Load embeddings and CSV data from specified paths.

    :param embeddings_path: Path to the web keyword embeddings file.
    :param user_embeddings_path: Path to the user tag embeddings file.
    :param csv_path: Path to the CSV file containing image data.
    :return: Tuple of web keyword embeddings, user keyword embeddings, and the dataframe.
    """
    web_keyword_embeddings = np.load(embeddings_path)
    user_keyword_embeddings = np.load(user_embeddings_path)
    df = pd.read_csv(csv_path)
    return web_keyword_embeddings, user_keyword_embeddings, df


def normalize_embeddings(embeddings):
    """
    Normalize the embeddings.

    :param embeddings: Embeddings to be normalized.
    :return: Normalized embeddings.
    """
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def calculate_similarities(web_embeddings, user_embeddings):
    """
    Calculate the cosine similarities between web and user embeddings.

    :param web_embeddings: Normalized web keyword embeddings.
    :param user_embeddings: Normalized user keyword embeddings.
    :return: Similarities as a numpy array.
    """
    return np.sum(web_embeddings * user_embeddings, axis=1)


def plot_results(similarities):
    """
    Plot the similarities and their distribution.

    :param similarities: Similarities between web and user embeddings.
    """
    plt.title("Distribution of web and user keywords similarity")
    plt.hist(similarities, bins=25)
    plt.savefig("distribution_keyword_web_user_similarity.png")
    plt.show()


# Main script
if __name__ == "__main__":
    web_keyword_embeddings, user_keyword_embeddings, df = load_data(
        "bert_combined_keywords_embeddings.npy",
        "bert_combined_user_keywords_embeddings.npy",
        "unsplash_images_cleaned_user_caption_top_kw.csv"
    )

    A_norm = normalize_embeddings(web_keyword_embeddings)
    B_norm = normalize_embeddings(user_keyword_embeddings)

    similarities = calculate_similarities(A_norm, B_norm)

    plot_results(similarities)
