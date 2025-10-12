import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ChurnDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the preprocessed churn data.
    """
    def __init__(self, features_path: str, labels_path: str, preprocessor):
        """
        Args:
            features_path (str): Path to the preprocessed features CSV (e.g., X_train.csv).
            labels_path (str): Path to the labels CSV (e.g., y_train.csv).
            preprocessor (ChurnDataPreprocessor): The fitted preprocessor object.
        """
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)
        
        # Separate numeric and categorical features based on the preprocessor
        self.numeric_features = torch.tensor(X[preprocessor.numeric_cols].values, dtype=torch.float32)
        self.categorical_features = torch.tensor(X[preprocessor.categorical_cols].values, dtype=torch.int64)
        self.labels = torch.tensor(y.values, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'numeric': self.numeric_features[idx],
            'categorical': self.categorical_features[idx]
        }, self.labels[idx]
