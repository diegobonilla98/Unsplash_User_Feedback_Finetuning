import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class TranslationNet(nn.Module):
    """
    Defines the translation network architecture for mapping embedding spaces.

    :param input_dim: Dimension of the input embedding.
    :param output_dim: Dimension of the output embedding.
    """

    def __init__(self, input_dim=768, output_dim=768):
        super(TranslationNet, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input embeddings.
        :return: Translated embeddings.
        """
        return self.sequence(x)


def load_datasets():
    """
    Loads web and user keyword embeddings and creates training and testing datasets.

    :return: A tuple of DataLoader instances for train and test datasets.
    """
    web_keyword_embeddings = np.load("bert_combined_keywords_embeddings.npy")
    user_keyword_embeddings = np.load("bert_combined_user_keywords_embeddings.npy")

    web_keywords_tensor = torch.tensor(web_keyword_embeddings, dtype=torch.float32)
    user_keywords_tensor = torch.tensor(user_keyword_embeddings, dtype=torch.float32)

    train_num = int(web_keyword_embeddings.shape[0] * 0.8)

    train_dataset = TensorDataset(web_keywords_tensor[:train_num], user_keywords_tensor[:train_num])
    test_dataset = TensorDataset(web_keywords_tensor[train_num:], user_keywords_tensor[train_num:])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def train_model(train_loader, test_loader, device):
    """
    Trains the model using the given DataLoader instances for training and validation.

    :param train_loader: DataLoader for the training data.
    :param test_loader: DataLoader for the test data.
    :param device: Device to train the model on ('cuda' or 'cpu').
    """
    model = TranslationNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    patience = 10
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for web_batch, user_batch in train_loader:
            web_batch, user_batch = web_batch.to(device), user_batch.to(device)
            optimizer.zero_grad()
            outputs = model(web_batch)
            loss = criterion(outputs, user_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * web_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for web_batch, user_batch in test_loader:
                web_batch, user_batch = web_batch.to(device), user_batch.to(device)
                outputs = model(web_batch)
                loss = criterion(outputs, user_batch)
                val_loss += loss.item() * web_batch.size(0)
            val_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    torch.save(model, "tag_translation_model.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_datasets()
    train_model(train_loader, test_loader, device)
