import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFile
import clip
from torch import optim
from tqdm import tqdm

# Enable loading truncated images with PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device for model training
device = "cuda"

# Load the CLIP model and preprocessing function
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


class CustomDataset(Dataset):
    """
    Custom dataset class for loading Unsplash images and their user captions.
    """

    def __init__(self):
        """
        Initialize the dataset by loading the CSV file containing image paths and captions.
        """
        self.df = pd.read_csv("unsplash_images_cleaned_user_caption_top_kw.csv")

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetch the image and its associated caption by index.

        :param idx: Index of the dataset item.
        :return: A tuple of the processed image and tokenized caption text.
        """
        row = self.df.iloc[idx]
        image_path = row["saved_path"]
        user_keywords = row["user_tags_caption"]

        image = Image.open(image_path)
        image = preprocess(image)
        text = clip.tokenize([user_keywords], truncate=True)[0]

        return image, text


def convert_models_to_fp32(model):
    """
    Convert model parameters and gradients to float32.

    :param model: The model to convert.
    """
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


# Initialize dataset and dataloader
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training parameters
EPOCHS = 5
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader) * EPOCHS)

# Training loop
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch in pbar:
        optimizer.zero_grad()
        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        total_loss.backward()
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss.item():.4f}")

# Save the fine-tuned model
torch.save(model, "clip_user_finetuned.pth")
