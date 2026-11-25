"""AnaCP baseline: analytic contrastive projection (Momeni et al.).

A stack of ``num_heads`` contrastive-projection (CP) layers maps frozen
features towards spread-out class targets; a final RLS classifier is fitted on
Gaussian pseudo-replay drawn from the stored class statistics.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class CPLayer:
    def __init__(self, D: int, input_dim: int, device: torch.device, reg: float):
        self.D = D
        self.device = device
        self.reg = reg
        self.W = torch.randn(D, input_dim, device=device)
        self.A = torch.zeros(D, D, device=device)
        self.Z_means = defaultdict(lambda: torch.zeros(D, device=device))
        self.counts = defaultdict(int)
        self.theta = None

    def update_stats(self, X: torch.Tensor, Y: torch.Tensor):
        Z = F.gelu(X @ self.W.T)
        self.A += Z.T @ Z
        for c in torch.unique(Y).tolist():
            mask = Y == c
            n = int(mask.sum())
            total = self.counts[c] + n
            self.Z_means[c] = (self.Z_means[c] * self.counts[c] + Z[mask].mean(dim=0) * n) / total
            self.counts[c] = total

    def solve_theta(self, target_means: torch.Tensor):
        B = torch.zeros(self.D, target_means.shape[1], device=self.device)
        for c, z_mean in self.Z_means.items():
            B += self.counts[c] * torch.outer(z_mean, target_means[c])
        A = self.A + self.reg * torch.eye(self.D, device=self.device)
        self.theta = torch.linalg.solve(A, B)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = F.gelu(X @ self.W.T)
        return F.normalize(Z @ self.theta, p=2, dim=1)


class AnaCP:
    def __init__(self, D: int, reg: float, num_heads: int, seed: int, device: torch.device,
                 samples_per_class: int, shared_cov: bool):
        self.D = D
        self.reg = reg
        self.num_heads = num_heads
        self.seed = seed
        self.device = device
        self.samples_per_class = samples_per_class
        self.shared_cov_flag = shared_cov
        torch.manual_seed(seed)

        self.input_dim = None
        self.num_classes = 0
        self.class_map = {}
        self.inverse_map = {}
        self.class_counts = defaultdict(int)
        self.class_means = None
        self.class_covs = {}
        self.shared_cov = None

        self.W = None
        self.layers = None
        self.theta = None

    # ------------------------------------------------------------------
    # class statistics
    # ------------------------------------------------------------------
    def _eye(self) -> torch.Tensor:
        return torch.eye(self.input_dim, device=self.device)

    def _map_labels(self, Y: np.ndarray) -> torch.Tensor:
        for label in sorted(set(np.asarray(Y).tolist())):
            if label not in self.class_map:
                self.class_map[label] = self.num_classes
                self.inverse_map[self.num_classes] = label
                self.num_classes += 1
        return torch.tensor([self.class_map[v] for v in np.asarray(Y).tolist()],
                            device=self.device, dtype=torch.long)

    def _update_means(self, X: torch.Tensor, y: torch.Tensor):
        for c in torch.unique(y).tolist():
            mask = y == c
            n = int(mask.sum())
            total = self.class_counts[c] + n
            self.class_means[c] = (self.class_means[c] * self.class_counts[c]
                                   + X[mask].mean(dim=0) * n) / total
            self.class_counts[c] = total

    def _update_shared_cov(self, X: torch.Tensor, y: torch.Tensor):
        centered = X - torch.stack([self.class_means[c] for c in y.tolist()])
        cov_new = centered.T @ centered / X.shape[0]
        if self.shared_cov is None:
            self.shared_cov = cov_new + 1e-4 * self._eye()
        else:
            n = sum(self.class_counts.values())
            self.shared_cov = (self.shared_cov * n + cov_new * X.shape[0]) / (n + X.shape[0])

    def _update_class_cov(self, X: torch.Tensor, y: torch.Tensor):
        for c in torch.unique(y).tolist():
            centered = X[y == c] - self.class_means[c]
            n_new = centered.shape[0]
            cov = centered.T @ centered / n_new
            if c in self.class_covs:
                n = self.class_counts[c]
                self.class_covs[c] = (self.class_covs[c] * (n - n_new) + cov * n_new) / n
            else:
                self.class_covs[c] = cov + 1e-4 * self._eye()

    def _spread_means(self, class_means: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Push class means apart in the whitened space to obtain CP targets."""
        eigvals, eigvecs = torch.linalg.eigh(cov)
        whiten = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + 1e-8)) @ eigvecs.T
        center = class_means.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd((class_means - center) @ whiten.T, full_matrices=False)
        spread = U @ torch.diag(S + 1.0) @ Vh
        return F.normalize(spread @ whiten + center, p=2, dim=1)

    # ------------------------------------------------------------------
    # pseudo-replay for the final classifier
    # ------------------------------------------------------------------
    def _generate_replay_data(self):
        n = self.samples_per_class
        replay_Y = torch.arange(self.num_classes, device=self.device).repeat_interleave(n)
        means = torch.stack([self.class_means[c] for c in range(self.num_classes)])
        zero = torch.zeros(self.input_dim, device=self.device)
        if self.shared_cov_flag:
            noise = torch.distributions.MultivariateNormal(zero, self.shared_cov)
            replay_X = noise.sample((replay_Y.numel(),)) + means[replay_Y]
        else:
            chunks = []
            for c in range(self.num_classes):
                noise = torch.distributions.MultivariateNormal(zero, self.class_covs[c])
                chunks.append(noise.sample((n,)) + means[c])
            replay_X = torch.cat(chunks, dim=0)
        return replay_X, replay_Y

    # ------------------------------------------------------------------
    # training / inference
    # ------------------------------------------------------------------
    def _init(self, input_dim: int):
        self.input_dim = input_dim
        self.W = torch.randn(self.D, input_dim, device=self.device)
        self.layers = [CPLayer(self.D, input_dim, self.device, self.reg)
                       for _ in range(self.num_heads)]
        self.class_means = defaultdict(lambda: torch.zeros(input_dim, device=self.device))

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        outs = torch.stack([layer.forward(X) for layer in self.layers]).mean(dim=0)
        return F.gelu(outs @ self.W.T)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.device), p=2, dim=1)
        if self.input_dim is None:
            self._init(X.shape[1])
        y = self._map_labels(Y)

        self._update_means(X, y)
        self._update_shared_cov(X, y)
        if not self.shared_cov_flag:
            self._update_class_cov(X, y)

        means = torch.stack([self.class_means[c] for c in range(self.num_classes)])
        target_means = self._spread_means(means, self.shared_cov)
        for layer in self.layers:
            layer.update_stats(X, y)
            layer.solve_theta(target_means)

        replay_X, replay_Y = self._generate_replay_data()
        H = self._embed(replay_X)
        Y_onehot = torch.zeros(replay_Y.numel(), self.num_classes, device=self.device)
        Y_onehot.scatter_(1, replay_Y.unsqueeze(1), 1.0)
        A = H.T @ H + self.reg * torch.eye(self.D, device=self.device)
        self.theta = torch.linalg.solve(A, H.T @ Y_onehot)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.device), p=2, dim=1)
        preds = (self._embed(X) @ self.theta).argmax(dim=1).cpu().numpy()
        return np.array([self.inverse_map[p] for p in preds])
