import os
import numpy as np
import pandas as pd
from this_utils import load_model_and_device, encode_text, find_closest_images, show_images_grid, batch_process_images


def load_embeddings(image_paths, model, preprocess, device, batch_size=64, embeddings_file="embeddings_clip_finetune_user_captions.npy"):
    """
    Load or generate embeddings for the given image paths.

    :param image_paths: List of paths to images.
    :param model: The model used for generating embeddings.
    :param preprocess: The preprocessing function to apply to images before embedding.
    :param device: The device to run the model on.
    :return: A numpy array of image embeddings.
    """
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    else:
        embeddings = batch_process_images(image_paths, batch_size, model, preprocess, device)
        np.save(embeddings_file, embeddings)
    return embeddings


def calculate_variance(embeddings):
    """
    Calculate and print the variance of normalized embeddings.

    :param embeddings: Numpy array of image embeddings.
    """
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    variance = np.var(normalized_embeddings, axis=0).mean()
    return variance


def find_and_show_closest_images(text_embedding, image_embeddings, image_paths, top_n=25, result_image_path="NEW_clip_finetune_user_captions_top_25_retrieval.png"):
    """
    Find the closest images to the given text embedding and display them.

    :param text_embedding: Embedding of the input text.
    :param image_embeddings: Embeddings of the images.
    :param image_paths: List of image paths corresponding to the embeddings.
    :param top_n: Number of closest images to return and display.
    """
    closest_paths = find_closest_images(text_embedding, image_embeddings, image_paths, top_n=top_n)
    show_images_grid(closest_paths, text, result_image_path)


# Main process
if __name__ == "__main__":
    # Load model, preprocessing, and device
    model, preprocess, device = load_model_and_device("NEW_clip_user_finetuned.pth")

    # Prepare text embedding
    text = "a plane in the blue sky"
    text_embedding = encode_text(text, model, device)

    # Load image paths from CSV
    df = pd.read_csv("unsplash_images_cleaned_user_caption_top_kw_split_redux.csv")
    df = df[df["split"] == "val"]
    image_paths = df['saved_path'].tolist()

    # Load or generate image embeddings
    image_embeddings = load_embeddings(image_paths, model, preprocess, device)

    # Calculate and print variance of embeddings
    print(calculate_variance(image_embeddings))

    # Find and show closest images to the text
    find_and_show_closest_images(text_embedding, image_embeddings, image_paths)
