import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image, ImageFile
from ram import get_transform, inference_tag2text
from ram.models import tag2text

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def initialize_model(checkpoint_path, image_size, device):
    """
    Initialize the tag2text model with the given parameters.

    :param checkpoint_path: Path to the model's checkpoint.
    :param image_size: The size of the images for the model.
    :param device: The device to run the model on.
    :return: The initialized model.
    """
    transform = get_transform(image_size=image_size)
    model = tag2text(pretrained=checkpoint_path, image_size=image_size, vit='swin_b').eval().to(device)
    return model, transform


def process_images(dataframe, model, transform, device):
    """
    Process images in the dataframe, generating captions and top keywords using the model.

    :param dataframe: The dataframe containing image paths and user keywords.
    :param model: The initialized tag2text model.
    :param transform: Image transformation function.
    :param device: The device to run the model on.
    :return: The dataframe with added captions and top keywords.
    """
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        user_keywords = ', '.join(eval(row["user_keyword_cleaned"]))
        image = Image.open(row["saved_path"])
        image_tensor = transform(image).unsqueeze(0).to(device)
        result = inference_tag2text(image_tensor, model, user_keywords)
        tags, caption = result[0].strip().replace('  ', ' ').split(" | "), result[2]
        dataframe.at[idx, "user_tags_caption"] = caption
        dataframe.at[idx, "user_top_keywords"] = tags
        if idx % 10 == 0:
            dataframe.to_csv("unsplash_images_cleaned_user_caption_top_kw.csv", index=False)
    return dataframe


if __name__ == "__main__":
    # Create captions of the images using user keyword feedback
    checkpoint_path = "./tag2text_swin_14m.pth"
    image_size = 384
    device = "cuda"

    model, transform = initialize_model(checkpoint_path, image_size, device)
    dataframe = pd.read_csv("unsplash_images_cleaned.csv")
    dataframe["user_tags_caption"] = None
    dataframe["user_top_keywords"] = None

    processed_df = process_images(dataframe, model, transform, device)
    processed_df.to_csv("unsplash_images_cleaned_user_caption_top_kw.csv", index=False)