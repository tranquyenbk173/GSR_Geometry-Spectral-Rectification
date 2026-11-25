"""Nearest class mean on frozen features (SimpleCIL)."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class NCM:
    def __init__(self, device: torch.device):
        self.device = device
        self.class_sums = None
        self.class_counts = defaultdict(int)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.long, device=self.device)
        if self.class_sums is None:
            dim = X.shape[1]
            self.class_sums = defaultdict(lambda: torch.zeros(dim, device=self.device))
        for c in torch.unique(Y).tolist():
            X_c = X[Y == c]
            self.class_sums[c] += X_c.sum(dim=0)
            self.class_counts[c] += X_c.size(0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.device), dim=1)
        class_ids = list(self.class_sums.keys())
        means = torch.stack([self.class_sums[c] / self.class_counts[c] for c in class_ids])
        preds = (X @ F.normalize(means, dim=1).T).argmax(dim=1).cpu().numpy()
        return np.array([class_ids[p] for p in preds])
