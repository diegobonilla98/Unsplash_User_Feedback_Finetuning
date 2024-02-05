import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from this_utils import show_images_grid
import tqdm
import os


# Initialize BERT model and tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to("cuda")

# Read dataset
df = pd.read_csv("unsplash_images_cleaned_user_caption_top_kw.csv")


def encode_tags(tags):
    """
    Encode tags into embeddings using BERT.

    :param tags: A list of tags to be encoded.
    :return: The mean embedding of the tags as a numpy array.
    """
    inputs = tokenizer(tags, return_tensors='pt', padding=True, truncation=True, max_length=512, return_attention_mask=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    embeddings_mean = embeddings.mean(dim=1).mean(dim=0).cpu().numpy()
    return embeddings_mean


def find_closest_embedding(search_keyword, list_of_embeddings, k=5):
    """
    Find the top k closest embeddings to the search keyword.

    :param search_keyword: The keyword to search for.
    :param list_of_embeddings: A tensor of embeddings.
    :param k: The number of closest embeddings to return.
    :return: A list of indices of the top k closest embeddings.
    """
    inputs = tokenizer(search_keyword, return_tensors='pt', padding=True, truncation=True, max_length=512,
                       return_attention_mask=True).to("cuda")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    search_embedding = outputs.last_hidden_state.mean(dim=1)
    search_embedding_norm = search_embedding / search_embedding.norm(dim=1)[:, None]
    list_of_embeddings_norm = list_of_embeddings / list_of_embeddings.norm(dim=1)[:, None]
    similarities = torch.mm(search_embedding_norm, list_of_embeddings_norm.transpose(0, 1))
    top_k_values, top_k_indices = torch.topk(similarities, k, dim=1)
    return top_k_indices[0].tolist()


# Load or compute embeddings
embeddings_file = "bert_combined_keywords_embeddings.npy"
if os.path.exists(embeddings_file):
    list_of_embeddings = np.load(embeddings_file)
else:
    list_of_embeddings = []
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        keyword_cleaned = eval(row["keyword_cleaned"])
        combined_embedding = encode_tags(keyword_cleaned)
        list_of_embeddings.append(combined_embedding)
    list_of_embeddings = np.array(list_of_embeddings)
    np.save(embeddings_file, list_of_embeddings)

# Convert list of embeddings to tensor and perform search
list_of_embeddings_tensor = torch.tensor(list_of_embeddings).float().to("cuda")
search_keyword = "a plane in the blue sky"
closest_indexes = find_closest_embedding(search_keyword, list_of_embeddings_tensor, k=25)
print(closest_indexes)

# Calculate and print the variance of normalized embeddings
normalized_embeddings = list_of_embeddings / np.linalg.norm(list_of_embeddings, axis=1, keepdims=True)
variance = np.var(normalized_embeddings, axis=0).mean()
print(variance)

# Retrieve and display closest images
closest_paths = df.iloc[closest_indexes]["saved_path"].tolist()
show_images_grid(closest_paths, search_keyword, "bert_combined_keywords_match_top_25_retrieval.png")
