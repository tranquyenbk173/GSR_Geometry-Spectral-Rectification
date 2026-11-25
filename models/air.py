"""AIR: Analytic Imbalance Rectifier (Fang et al.).

Per-class auto-/cross-correlation statistics are aggregated with the
re-weighting factor pi_y = 1 / N_y before solving the ridge regression.
"""

import numpy as np
import torch


class AIR:
    def __init__(self, D: int, reg: float, seed: int, device: torch.device):
        self.D = D
        self.reg = reg
        self.seed = seed
        self.device = device
        self.W = None
        self.class_stats = {}    # class index -> {"A": (D, D), "c": (D,), "count": int}
        self.Theta = None
        self.is_updated = False
        self.num_classes = 0
        self.label_to_index = {}
        self.index_to_label = {}

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.long, device=self.device)
        if self.W is None:
            torch.manual_seed(self.seed)
            self.W = torch.randn(self.D, X.shape[1], device=self.device)

        H = torch.relu(X @ self.W.T)
        for label in torch.unique(Y).tolist():
            if label not in self.label_to_index:
                self.label_to_index[label] = self.num_classes
                self.index_to_label[self.num_classes] = label
                self.num_classes += 1
            idx = self.label_to_index[label]
            if idx not in self.class_stats:
                self.class_stats[idx] = {
                    "A": torch.zeros(self.D, self.D, dtype=torch.float64, device=self.device),
                    "c": torch.zeros(self.D, dtype=torch.float64, device=self.device),
                    "count": 0,
                }
            H_c = H[Y == label].double()
            stats = self.class_stats[idx]
            stats["A"] += H_c.T @ H_c
            stats["c"] += H_c.sum(dim=0)
            stats["count"] += H_c.shape[0]
        self.is_updated = False

    def _solve(self):
        A = self.reg * torch.eye(self.D, dtype=torch.float64, device=self.device)
        C = torch.zeros(self.D, self.num_classes, dtype=torch.float64, device=self.device)
        for idx, stats in self.class_stats.items():
            pi = 1.0 / stats["count"]
            A += pi * stats["A"]
            C[:, idx] = pi * stats["c"]
        self.Theta = torch.linalg.solve(A, C).float()
        self.is_updated = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if not self.is_updated:
            self._solve()
        preds = (torch.relu(X @ self.W.T) @ self.Theta).argmax(dim=1).cpu().numpy()
        return np.array([self.index_to_label[i] for i in preds])
