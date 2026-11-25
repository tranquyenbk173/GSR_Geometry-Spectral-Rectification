"""Streaming LDA (Hayes & Kanan) with a shared covariance matrix."""

from collections import defaultdict

import numpy as np
import torch


class SLDA:
    def __init__(self, reg: float, device: torch.device):
        self.reg = reg
        self.device = device
        self.class_means = {}
        self.class_counts = defaultdict(int)
        self.cov = None
        self.total_count = 0
        self.cov_inv = None
        self.is_updated = False

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.long, device=self.device)
        if self.cov is None:
            self.cov = torch.zeros(X.shape[1], X.shape[1], device=self.device)

        for c in torch.unique(Y).tolist():
            X_c = X[Y == c]
            n_new, n_old = X_c.size(0), self.class_counts[c]
            mean_new = X_c.mean(dim=0)
            if n_old == 0:
                self.class_means[c] = mean_new
            else:
                self.class_means[c] = (n_old * self.class_means[c] + n_new * mean_new) / (n_old + n_new)
            self.class_counts[c] += n_new

            centered = X_c - self.class_means[c]
            self.cov += centered.T @ centered
            self.total_count += n_new
        self.is_updated = False

    def _update_cov_inv(self):
        cov = self.cov / (self.total_count - len(self.class_means))
        cov = cov + self.reg * torch.eye(cov.shape[0], device=self.device)
        self.cov_inv = torch.inverse(cov)
        self.is_updated = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if not self.is_updated:
            self._update_cov_inv()
        labels = list(self.class_means.keys())
        means = torch.stack([self.class_means[c] for c in labels])
        # LDA discriminant: x^T S^-1 mu - 0.5 mu^T S^-1 mu
        scores = X @ self.cov_inv @ means.T - 0.5 * ((means @ self.cov_inv) * means).sum(dim=1)
        preds = scores.argmax(dim=1).cpu().numpy()
        return np.array([labels[i] for i in preds])
