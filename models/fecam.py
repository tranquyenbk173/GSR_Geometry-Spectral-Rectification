"""FeCAM (Goswami et al.): per-class Mahalanobis classifier with Tukey
transform, covariance shrinkage and correlation normalisation."""

import numpy as np
import torch


class FeCAM:
    def __init__(self, device: torch.device, use_tukey: bool = True,
                 gamma1: float = 1.0, gamma2: float = 1.0):
        self.device = device
        self.use_tukey = use_tukey
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.class_means = {}
        self.class_precisions = {}

    def _tukey(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).pow(0.5) if self.use_tukey else x

    def _shrink(self, cov: torch.Tensor) -> torch.Tensor:
        d = cov.shape[0]
        eye = torch.eye(d, device=self.device)
        v1 = cov.diag().mean()
        v2 = cov[~eye.bool()].mean() if d > 1 else 0.0
        return cov + self.gamma1 * v1 * eye + self.gamma2 * v2 * (1.0 - eye)

    @staticmethod
    def _to_correlation(cov: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(cov.diag())
        return cov / torch.outer(std, std).clamp(min=1e-8)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = self._tukey(torch.as_tensor(X, dtype=torch.float32, device=self.device))
        Y = torch.as_tensor(Y, dtype=torch.long, device=self.device)
        d = X.shape[1]
        for c in torch.unique(Y).tolist():
            feats = X[Y == c]
            mean = feats.mean(dim=0)
            if feats.size(0) > 1:
                centered = feats - mean
                cov = centered.T @ centered / (feats.size(0) - 1)
            else:
                cov = torch.zeros(d, d, device=self.device)
            cov = self._to_correlation(self._shrink(cov))
            try:
                precision = torch.linalg.inv(cov)
            except RuntimeError:
                precision = torch.linalg.pinv(cov)
            self.class_means[c] = mean
            self.class_precisions[c] = precision

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._tukey(torch.as_tensor(X, dtype=torch.float32, device=self.device))
        classes = sorted(self.class_means)
        dists = []
        for c in classes:
            diff = X - self.class_means[c]
            dists.append(((diff @ self.class_precisions[c]) * diff).sum(dim=1))
        preds = torch.stack(dists, dim=1).argmin(dim=1).cpu().numpy()
        return np.array([classes[i] for i in preds])
