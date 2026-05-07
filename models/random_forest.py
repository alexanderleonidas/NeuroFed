import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """
        A wrapper for the scikit-learn RandomForestClassifier that is compatible with PyTorch tensors.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available CPU cores
        )

    def fit(self, x, y):
        # Convert tensors to numpy arrays if necessary
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y

        self.model.fit(x_np, y_np)

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        predictions = self.model.predict(x_np)
        return torch.from_numpy(predictions).long()

    def predict_proba(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        probabilities = self.model.predict_proba(x_np)
        return torch.from_numpy(probabilities).float()