import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from io import BytesIO, StringIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizeSpectrogram(transforms.Normalize):
    def __init__(self):
        super(NormalizeSpectrogram, self).__init__((0.5,), (0.5,))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    NormalizeSpectrogram(),
])

def load_spectrogram(track_id, spectrogram_bucket):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(spectrogram_bucket)
        blob = storage.Blob(f"{track_id}.png", bucket)
        blob_bytes = blob.download_as_bytes()

        img_data = BytesIO(blob_bytes)
        img = Image.open(img_data)

        return transform(img)
    except NotFound:
        return None
    except GoogleCloudError as e:
        print(f"Error loading spectrogram for track_id {track_id}: {str(e)}")
        return None

def load_data(spotify_data_file, spectrogram_bucket):
    try:
        print("Loading data...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(spectrogram_bucket)
        blob = storage.Blob(f"{spotify_data_file}", bucket)
        spotify_data_string = blob.download_as_text()
        spotify_df = pd.read_csv(StringIO(spotify_data_string))

        audio_features_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                                  'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

        print("Loading spectrograms...")
        with ThreadPoolExecutor() as executor:
            spectrograms = list(executor.map(
                load_spectrogram,
                spotify_df["track_id"],
                repeat(spectrogram_bucket)
            ))

        spectrograms = [spec for spec in spectrograms if spec is not None]

        track_ids_spectrogram_map = {track_id: spec for track_id, spec in zip(spotify_df["track_id"], spectrograms)}

        filtered_df = spotify_df[spotify_df["track_id"].isin(track_ids_spectrogram_map.keys())]
        audio_features = filtered_df[audio_features_columns].to_numpy()
        remaining_spectrograms = [track_ids_spectrogram_map[track_id] for track_id in filtered_df["track_id"]]

        return remaining_spectrograms, audio_features
    except GoogleCloudError as e:
        print(f"Error loading data: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading data: {str(e)}")
        return None, None

class CNNModel(nn.Module):
    def __init__(self, num_features):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_features)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train_model(train_loader, val_loader, num_epochs=10, batch_size=32, learning_rate=0.001):
    try:
        num_features = train_loader.dataset.tensors[1].shape[1]
        model = CNNModel(num_features).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        patience = 5

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step()
            print(f'Epoch {epoch + 1}, Training Loss: {running_loss}, Validation Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == patience:
                    print("Early stopping")
                    break

        model.load_state_dict(best_model_weights)
        return model
    except Exception as e:
        print(f"Unexpected error during training: {str(e)}")
        return None

def save_model_to_bucket(model, save_path, bucket_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        torch.save(model.state_dict(), save_path)

        blob = storage.Blob(save_path, bucket)
        blob.upload_from_filename(save_path)

        os.remove(save_path)
    except GoogleCloudError as e:
        print(f"Error saving model to bucket: {str(e)}")
    except Exception as e:
        print(f"Unexpected error saving model to bucket: {str(e)}")

spectrogram_bucket = "spectrogram-botify/Sample_Spectrogram"
sample_set_file_path = "spectrogram-botify/sample_feats.csv"

spectrograms, audio_features = load_data(sample_set_file_path, spectrogram_bucket)
if spectrograms is None or audio_features is None:
    print("Error loading data. Exiting.")
    exit()

spectrograms = torch.stack(spectrograms).float().to(device)
audio_features = torch.from_numpy(audio_features).float().to(device)

num_epochs = 10
batch_size = 32
learning_rate = 0.001

k_fold = KFold(n_splits=5)

val_losses = []

for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(spectrograms)):
    print(f"Fold {fold_idx + 1}")

    X_train, X_val = spectrograms[train_indices], spectrograms[val_indices]
    y_train, y_val = audio_features[train_indices], audio_features[val_indices]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = train_model(train_loader, val_loader, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    if model is None:
        print(f"Error training model for fold {fold_idx + 1}.")
        continue

    save_path = f"model_weights_fold_{fold_idx + 1}.pt"
    save_model_to_bucket(model, save_path, spectrogram_bucket)
    val_losses.append(best_val_loss)

avg_val_loss = sum(val_losses) / len(val_losses)
print(f"Average Validation Loss: {avg_val_loss}")

final_train_dataset = TensorDataset(spectrograms, audio_features)
final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)

final_model = train_model(final_train_loader, final_train_loader, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
if final_model is None:
    print("Error training final model.")

final_save_path = "final_model_weights.pt"
save_model_to_bucket(final_model, final_save_path, spectrogram_bucket)