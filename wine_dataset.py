import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class WineQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_csv(filepath):
    return pd.read_csv(filepath)


def preprocess_data(wine_data):
    X = wine_data[["alcohol", "sulphates"]]
    y = (wine_data["quality"] >= 6).astype(int)
    scaler = StandardScaler()
    return scaler.fit_transform(X), y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def to_tensor(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train.astype("float32"))
    y_train_tensor = torch.tensor(y_train.values).long()
    X_test_tensor = torch.tensor(X_test.astype("float32"))
    y_test_tensor = torch.tensor(y_test.values).long()
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def get_dataloaders(
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size
):
    train_dataset = WineQualityDataset(X_train_tensor, y_train_tensor)
    test_dataset = WineQualityDataset(X_test_tensor, y_test_tensor)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def get_class_weights(y_train):
    classes = np.unique(y_train)
    return compute_class_weight("balanced", classes=classes, y=y_train)


def get_wine_dataloaders(batch_size=32, test_size=0.2, filepath="winequality-red.csv"):
    wine_data = load_csv(filepath)
    X, y = preprocess_data(wine_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = to_tensor(
        X_train, X_test, y_train, y_test
    )
    train_loader, test_loader = get_dataloaders(
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size
    )
    class_weights = get_class_weights(y_train)
    return train_loader, test_loader, class_weights
