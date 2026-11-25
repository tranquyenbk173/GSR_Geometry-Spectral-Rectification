"""GACL-style RLS classifier on frozen features, with optional GSR.

    H   = ReLU(X W^T)                    fixed random projection (buffer layer)
    G_t = G_{t-1} + H^T H + beta * H_aug^T H_aug
    Q_t = Q_{t-1} + H^T Y + beta * H_aug^T Y_aug
    W_t = (G_t + reg * I)^{-1} Q_t

With ``gsr=None`` this is the plain ridge baseline. G and Q are accumulated and
solved in double precision: with a large rectification weight beta the system
is too ill-conditioned for single precision.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .gsr import GSRConfig, gsr_augment, stable_rank


class GACL:
    def __init__(self, D: int, reg: float, seed: int, device: torch.device,
                 gsr: Optional[GSRConfig] = None):
        self.D = D
        self.reg = reg
        self.seed = seed
        self.device = device
        self.gsr = gsr
        # Features are projected onto the unit sphere whenever spherical GSR is
        # used; the plain baseline and linear mixup work on raw features.
        self.normalize_input = gsr is not None and gsr.spherical

        self.W = None        # (D, d) random projection
        self.G = None        # (D, D) Gram matrix
        self.Q = None        # (D, C) cross-correlation
        self.Theta = None    # (D, C) classifier
        self.is_updated = False

        self.num_classes = 0
        self.label_to_index = {}
        self.index_to_label = {}
        self.class_counts = {}   # N_c: real samples seen per class index

    def _prepare(self, X: np.ndarray) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return F.normalize(X, p=2, dim=1) if self.normalize_input else X

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        return torch.relu(X @ self.W.T)

    def _one_hot(self, y: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(y.numel(), self.num_classes, device=self.device, dtype=torch.float64)
        out.scatter_(1, y.unsqueeze(1), 1.0)
        return out

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = self._prepare(X)

        if self.W is None:
            torch.manual_seed(self.seed)
            self.W = torch.randn(self.D, X.shape[1], device=self.device)
            self.G = torch.zeros(self.D, self.D, device=self.device, dtype=torch.float64)
            self.Q = torch.zeros(self.D, 0, device=self.device, dtype=torch.float64)

        for label in np.unique(Y).tolist():
            if label not in self.label_to_index:
                self.label_to_index[label] = self.num_classes
                self.index_to_label[self.num_classes] = label
                self.num_classes += 1
        y = torch.tensor([self.label_to_index[v] for v in np.asarray(Y).tolist()],
                         device=self.device, dtype=torch.long)
        for c, n in zip(*[t.tolist() for t in torch.unique(y, return_counts=True)]):
            self.class_counts[c] = self.class_counts.get(c, 0) + n

        if self.Q.size(1) < self.num_classes:
            pad = torch.zeros(self.D, self.num_classes - self.Q.size(1), device=self.device,
                              dtype=torch.float64)
            self.Q = torch.cat([self.Q, pad], dim=1)

        # Fidelity term: real features.
        H = self._project(X).double()
        self.G += H.T @ H
        self.Q += H.T @ self._one_hot(y)

        # Rectification term: intra-class mixup features.
        if self.gsr is not None:
            X_aug, y_aug = gsr_augment(X, y, self.gsr, self.class_counts)
            if X_aug is not None:
                H_aug = self._project(X_aug).double()
                self.G += self.gsr.beta * (H_aug.T @ H_aug)
                self.Q += self.gsr.beta * (H_aug.T @ self._one_hot(y_aug))

        self.is_updated = False

    def _solve(self):
        A = self.G + self.reg * torch.eye(self.D, device=self.device, dtype=torch.float64)
        self.Theta = torch.linalg.solve(A, self.Q).float()
        self.is_updated = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_updated:
            self._solve()
        logits = self._project(self._prepare(X)) @ self.Theta
        preds = logits.argmax(dim=1).cpu().numpy()
        return np.array([self.index_to_label[i] for i in preds])

    def stable_rank(self) -> float:
        """Stable rank of the regularised Gram matrix, sr(G + reg * I)."""
        return stable_rank(self.G, self.reg)
