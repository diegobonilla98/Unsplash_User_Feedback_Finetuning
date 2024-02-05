import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import torch
import clip
from PIL import Image, ImageFile
import tqdm
import numpy as np

# Allow loading truncated images with PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


def show_images_grid(image_paths, title, save_fig_path):
    """
    Display a grid of images.

    :param image_paths: List of paths to the images.
    :param title: Title of the plot.
    :param save_fig_path: Path to save the figure.
    """
    n_images = len(image_paths)
    ncols = int(math.ceil(math.sqrt(n_images)))
    nrows = int(math.ceil(n_images / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    fig.suptitle(title)
    axs = axs.flatten()

    for img_path, ax in zip(image_paths, axs):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Remove empty subplots
    for i in range(len(image_paths), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.show()


def load_model_and_device(model_path):
    """
    Load the model and specify the device.

    :param model_path: Path to the model, if None, load default CLIP model.
    :return: Tuple of (model, preprocess, device).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    if model_path:
        model = torch.load(model_path).to(device)
    model.eval()
    return model, preprocess, device


def batch_process_images(image_paths, batch_size, model, preprocess, device):
    """
    Process images in batches.

    :param image_paths: List of image paths.
    :param batch_size: Size of each batch.
    :param model: Loaded model for image encoding.
    :param preprocess: Preprocessing function for the images.
    :param device: Computation device.
    :return: Numpy array of all features.
    """
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    all_features = []

    with torch.no_grad():
        for batch in tqdm.tqdm(batches):
            images = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in batch]
            images = torch.cat(images)
            features = model.encode_image(images)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features).numpy()
    return all_features


def encode_text(text, model, device):
    """
    Encode text into features.

    :param text: Text to encode.
    :param model: Loaded model for text encoding.
    :param device: Computation device.
    :return: Numpy array of text features.
    """
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
    return text_features.cpu().numpy()


def find_closest_images(text_embedding, image_embeddings, image_paths, top_n=5):
    """
    Find the closest images to the given text description.

    :param text_embedding: Embedding of the input text.
    :param image_embeddings: Embeddings of the images.
    :param image_paths: List of image paths corresponding to the embeddings.
    :param top_n: Number of closest images to return.
    :return: List of paths to the closest images.
    """
    # Normalize the embeddings
    text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    # Calculate cosine similarity
    similarities = np.dot(image_embeddings, text_embedding.T).flatten()

    # Get the top N indices of images with the highest similarity
    indices = np.argsort(similarities)[-top_n:][::-1]

    # Fetch the paths of these images
    closest_paths = [image_paths[idx] for idx in indices]
    return closest_paths
