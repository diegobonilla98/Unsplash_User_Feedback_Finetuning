import torch
from PIL import Image
import clip
from typing import List
import numpy as np
from this_utils import load_model_and_device
import pandas as pd

# Load the model
model, preprocess, device = load_model_and_device("NEW_clip_user_finetuned.pth")


def preprocess_images(image_paths: List[str], preprocess) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    return torch.stack([preprocess(image) for image in images]).to(device)


def preprocess_texts(texts: List[str], model) -> torch.Tensor:
    return clip.tokenize(texts).to(device)


def calculate_similarity(images: torch.Tensor, texts: torch.Tensor, model) -> np.ndarray:
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as dot product
        similarity = (image_features @ text_features.T).cpu().numpy()
    return similarity


def evaluate_accuracy(similarity_matrix: np.ndarray) -> float:
    # For each image, check if the highest similarity score is for its corresponding text
    top_image_matches = np.argmax(similarity_matrix, axis=1)
    top_text_matches = np.argmax(similarity_matrix, axis=0)

    correct_image_matches = sum(i == top_image_matches[i] for i in range(len(top_image_matches)))
    correct_text_matches = sum(i == top_text_matches[i] for i in range(len(top_text_matches)))

    total_correct_matches = correct_image_matches + correct_text_matches
    total_possible_matches = similarity_matrix.shape[0] + similarity_matrix.shape[1]

    accuracy = total_correct_matches / total_possible_matches
    return accuracy


# Example usage
df = pd.read_csv("unsplash_images_cleaned_user_caption_top_kw_split_redux.csv")
df = df[df["split"] == "val"]
image_paths = df["saved_path"].tolist()
texts = df["user_tags_caption"].tolist()

images = preprocess_images(image_paths, preprocess)
texts = preprocess_texts(texts, model)
similarity_scores = calculate_similarity(images, texts, model)
accuracy = evaluate_accuracy(similarity_scores)

print(f"Model accuracy: {accuracy}")
