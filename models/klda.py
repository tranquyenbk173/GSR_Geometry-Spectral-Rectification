"""Kernel LDA with random Fourier features (Momeni et al.)."""

from collections import defaultdict

import numpy as np
import torch


class KLDA:
    def __init__(self, D: int, gamma: float, reg: float, seed: int, device: torch.device):
        self.D = D
        self.gamma = gamma
        self.reg = reg
        self.seed = seed
        self.device = device
        self.W = None
        self.b = None
        self.class_means = {}
        self.class_counts = defaultdict(int)
        self.sigma = None
        self.sigma_inv = None
        self.is_updated = False

    def _rff(self, X: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            torch.manual_seed(self.seed)
            self.W = torch.randn(X.shape[1], self.D, device=self.device) * np.sqrt(2 * self.gamma)
            self.b = 2 * np.pi * torch.rand(self.D, device=self.device)
        return np.sqrt(2.0 / self.D) * torch.cos(X @ self.W + self.b)

    def update(self, X: np.ndarray, Y: np.ndarray):
        Z = self._rff(torch.as_tensor(X, dtype=torch.float32, device=self.device))
        Y = torch.as_tensor(Y, dtype=torch.long, device=self.device)
        if self.sigma is None:
            self.sigma = torch.zeros(self.D, self.D, device=self.device)

        for c in torch.unique(Y).tolist():
            Z_c = Z[Y == c]
            n_new, n_old = Z_c.size(0), self.class_counts[c]
            mean_new = Z_c.mean(dim=0)
            if n_old == 0:
                self.class_means[c] = mean_new
            else:
                self.class_means[c] = (n_old * self.class_means[c] + n_new * mean_new) / (n_old + n_new)
            self.class_counts[c] += n_new

            centered = Z_c - self.class_means[c]
            self.sigma += centered.T @ centered
        self.is_updated = False

    def predict(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        if not self.is_updated:
            self.sigma_inv = torch.inverse(self.sigma + self.reg * torch.eye(self.D, device=self.device))
            self.is_updated = True
        labels = list(self.class_means.keys())
        means = torch.stack([self.class_means[c] for c in labels])
        M = means @ self.sigma_inv
        bias = (M * means).sum(dim=1)

        preds = []
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        for start in range(0, X.shape[0], batch_size):
            Z = self._rff(X[start:start + batch_size])
            # Mahalanobis distance (z - mu)^T S^-1 (z - mu), up to the term in z only.
            preds.append((bias - 2.0 * Z @ M.T).argmin(dim=1))
        preds = torch.cat(preds).cpu().numpy()
        return np.array([labels[i] for i in preds])
