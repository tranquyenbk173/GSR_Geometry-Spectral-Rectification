"""AnaCP + GSR.

GSR enters through the Gram matrices of the contrastive-projection (CP)
layers: each layer accumulates real features (fidelity term) and intra-class
spherical-mixup features (rectification term).

Besides GSR, this long-tailed variant of AnaCP differs from the baseline in
``anacp.py`` in the following ways, all of which are part of the "AnaCP + Ours"
results:

* CP layers re-weight their random features by a per-dimension
  signal-to-noise ratio (between-/within-class variance), estimated on the
  first task, and use the inverse SNR as a per-dimension ridge penalty.
* Pseudo-replay for the final classifier is pulled towards the class mean
  (centroid mixup with Beta(5, 1)), re-projected to the unit sphere, drawn from
  a shrunk covariance, and over-sampled ``tail_replay_factor`` times for
  classes with fewer than ``samples_per_class`` real samples.
* The final classifier uses an SNR-based per-dimension ridge penalty.
* Logits are re-weighted by the inverse square-root class frequency.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .anacp import AnaCP
from .gsr import GSRConfig, gsr_augment


def _class_snr(means: torch.Tensor, sq_sums: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Per-dimension between-class / within-class variance ratio.

    means, sq_sums: (C, D) per-class mean and sum of squares; counts: (C,).
    """
    w = (counts / counts.sum()).unsqueeze(1)
    global_mean = (w * means).sum(dim=0)
    var_c = (sq_sums / counts.unsqueeze(1) - means ** 2).clamp(min=1e-6)
    sigma_w = (w * var_c).sum(dim=0).clamp(min=1e-6)
    sigma_b = (w * (means - global_mean) ** 2).sum(dim=0)
    return sigma_b / sigma_w


def _woodbury_solve(H: torch.Tensor, Y: torch.Tensor, r_inv: torch.Tensor) -> torch.Tensor:
    """Solve (H^T H + R) theta = H^T Y for diagonal R when N < D."""
    RinvHt = r_inv.unsqueeze(1) * H.T
    K = H @ RinvHt + torch.eye(H.shape[0], device=H.device)
    return RinvHt @ torch.linalg.solve(K, Y)


class SNRCPLayer:
    """CP layer with SNR-weighted random features."""

    def __init__(self, D: int, input_dim: int, device: torch.device, reg: float,
                 snr_power: float = 0.3):
        self.D = D
        self.device = device
        self.reg = reg
        self.snr_power = snr_power
        self.W = torch.randn(D, input_dim, device=device)
        self.A = torch.zeros(D, D, device=device)
        self.Z_means = {}
        self.Z_sq_sums = {}
        self.counts = {}
        self.scale = None    # sqrt of the SNR weights, fixed after the first task
        self.ridge = None
        self.theta = None

    def update_stats(self, X: torch.Tensor, Y: torch.Tensor,
                     weight: Optional[torch.Tensor] = None):
        """Accumulate statistics; ``weight`` (N,) holds per-sample weights."""
        Z = F.gelu(X @ self.W.T)
        if weight is None:
            self.A += Z.T @ Z
        else:
            self.A += (Z * weight.unsqueeze(1)).T @ Z

        for c in torch.unique(Y).tolist():
            mask = Y == c
            Z_c = Z[mask]
            if weight is None:
                n, mean, sq_sum = Z_c.shape[0], Z_c.mean(dim=0), (Z_c ** 2).sum(dim=0)
            else:
                w_c = weight[mask].unsqueeze(1)
                n = float(w_c.sum())
                mean, sq_sum = (w_c * Z_c).sum(dim=0) / n, (w_c * Z_c ** 2).sum(dim=0)

            if c not in self.counts:
                self.Z_means[c], self.Z_sq_sums[c], self.counts[c] = mean, sq_sum, n
            else:
                total = self.counts[c] + n
                self.Z_means[c] = (self.Z_means[c] * self.counts[c] + mean * n) / total
                self.Z_sq_sums[c] += sq_sum
                self.counts[c] = total

    def _build_snr(self):
        classes = list(self.counts)
        counts = torch.tensor([float(self.counts[c]) for c in classes], device=self.device)
        snr = _class_snr(torch.stack([self.Z_means[c] for c in classes]),
                         torch.stack([self.Z_sq_sums[c] for c in classes]), counts)
        g = (snr / (1.0 + snr)).clamp(min=0.0).pow(self.snr_power)
        g = g / g.mean().clamp(min=1e-6)
        inv = 1.0 / (g.clamp(min=1e-6) + 1e-6)
        self.scale = torch.sqrt(g)
        self.ridge = inv / inv.mean().clamp(min=1e-6)

    def solve_theta(self, target_means: torch.Tensor):
        B = torch.zeros(self.D, target_means.shape[1], device=self.device)
        for c, z_mean in self.Z_means.items():
            B += self.counts[c] * torch.outer(z_mean, target_means[c])
        if self.scale is None:
            # The SNR weights are estimated once, on the first task, and kept
            # fixed afterwards.
            self._build_snr()
        A = self.scale.unsqueeze(1) * self.A * self.scale.unsqueeze(0)
        A.diagonal().add_(self.reg * self.ridge + 1e-5)
        self.theta = torch.linalg.solve(A, self.scale.unsqueeze(1) * B)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = F.gelu(X @ self.W.T) * self.scale
        return F.normalize(Z @ self.theta, p=2, dim=1)


# ---------------------------------------------------------------------------------------
# Hyper-parameters of AnaCP + GSR.
#
# The AnaCP variant has three hyper-parameters of its own: the rectification weight beta,
# the exponent p of the logit re-weighting N_c^(-p) and the tail over-sampling factor r of
# the pseudo-replay. DEFAULT_HPARAMS is the un-tuned configuration. SELECTED_HPARAMS holds,
# for every backbone and dataset of the paper, the configuration chosen from
#     beta in {0.1, 0.25, 0.5, 1},  p in {0, 0.25, 0.5},  r in {1, 4}
# on a validation split of the training images (never the test set); see Appendix C.3 of
# the paper. scripts/sweep_anacp.sh re-runs the selection. run.py uses SELECTED_HPARAMS
# unless --anacp_hparams default is given; the ridge parameter and the projection
# dimension are shared with all other methods and are not part of the selection.
# ---------------------------------------------------------------------------------------
DEFAULT_HPARAMS = dict(gsr_beta=1.0, anacp_logit_power=0.5, anacp_tail_replay=4)
SELECTED_HPARAMS = {
    # (backbone, dataset):       beta                 p                        r
    ("dinov2", "cifar100"):   dict(gsr_beta=0.25, anacp_logit_power=0.5,  anacp_tail_replay=1),
    ("dinov2", "imagenet-r"): dict(gsr_beta=0.1,  anacp_logit_power=0.5,  anacp_tail_replay=1),
    ("dinov2", "t-imagenet"): dict(gsr_beta=0.5,  anacp_logit_power=0.5,  anacp_tail_replay=1),
    ("dinov2", "cub"):        dict(gsr_beta=0.5,  anacp_logit_power=0.5,  anacp_tail_replay=1),
    ("mocov3", "cifar100"):   dict(gsr_beta=0.1,  anacp_logit_power=0.25, anacp_tail_replay=4),
    ("mocov3", "imagenet-r"): dict(gsr_beta=0.1,  anacp_logit_power=0.25, anacp_tail_replay=1),
    ("mocov3", "t-imagenet"): dict(gsr_beta=0.25, anacp_logit_power=0.25, anacp_tail_replay=4),
    ("mocov3", "cub"):        dict(gsr_beta=0.1,  anacp_logit_power=0.5,  anacp_tail_replay=4),
}


def anacp_gsr_hparams(backbone: str, dataset: str, selected: bool = True) -> dict:
    """Hyper-parameters for a backbone / dataset; the defaults if none were selected."""
    if selected and (backbone, dataset) in SELECTED_HPARAMS:
        return dict(SELECTED_HPARAMS[(backbone, dataset)])
    return dict(DEFAULT_HPARAMS)


class AnaCPGSR(AnaCP):
    def __init__(self, D: int, reg: float, num_heads: int, seed: int, device: torch.device,
                 samples_per_class: int, shared_cov: bool, gsr: GSRConfig,
                 tail_replay_factor: int = 4, replay_shrinkage: float = 0.1,
                 logit_power: float = 0.5):
        if not gsr.spherical:
            raise ValueError("AnaCP + GSR operates on the unit sphere (geometry='spherical').")
        super().__init__(D, reg, num_heads, seed, device, samples_per_class, shared_cov)
        self.gsr = gsr
        self.tail_replay_factor = tail_replay_factor
        self.replay_shrinkage = replay_shrinkage
        self.logit_power = logit_power      # logits are scaled by N_c^(-logit_power)
        self.class_weights = None

    def _init(self, input_dim: int):
        self.input_dim = input_dim
        self.W = torch.randn(self.D, input_dim, device=self.device)
        self.layers = [SNRCPLayer(self.D, input_dim, self.device, self.reg)
                       for _ in range(self.num_heads)]
        self.class_means = defaultdict(lambda: torch.zeros(input_dim, device=self.device))

    def _generate_replay_data(self):
        """Gaussian pseudo-replay with spherical centroid mixup."""
        beta = torch.distributions.Beta(torch.tensor([5.0]), torch.tensor([1.0]))
        eye = self._eye()
        shared_tril = None
        xs, ys = [], []
        for c in range(self.num_classes):
            n_draw = self.samples_per_class
            if self.class_counts[c] < self.samples_per_class:
                n_draw *= self.tail_replay_factor

            if self.shared_cov_flag and shared_tril is not None:
                tril = shared_tril
            else:
                cov = self.shared_cov if self.shared_cov_flag else self.class_covs[c]
                s = self.replay_shrinkage
                cov = (1 - s) * cov + s * (cov.trace() / self.input_dim) * eye
                tril = torch.linalg.cholesky(cov)
                if self.shared_cov_flag:
                    shared_tril = tril

            mean = self.class_means[c]
            x = torch.distributions.MultivariateNormal(mean, scale_tril=tril).sample((n_draw,))
            lam = beta.sample((n_draw,)).to(self.device)
            xs.append(F.normalize(lam * x + (1.0 - lam) * mean, p=2, dim=1))
            ys.append(torch.full((n_draw,), c, dtype=torch.long, device=self.device))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.device), p=2, dim=1)
        if self.input_dim is None:
            self._init(X.shape[1])
        y = self._map_labels(Y)

        # Class statistics use real data only.
        self._update_means(X, y)
        self._update_shared_cov(X, y)
        if not self.shared_cov_flag:
            self._update_class_cov(X, y)

        freq = torch.tensor([float(self.class_counts[c]) for c in range(self.num_classes)],
                            device=self.device)
        scale = freq.sqrt() if self.logit_power == 0.5 else freq.pow(self.logit_power)
        w = 1.0 / (scale + 1e-6)
        self.class_weights = w / w.mean()

        means = torch.stack([self.class_means[c] for c in range(self.num_classes)])
        target_means = self._spread_means(means, self.shared_cov)

        # CP layers: fidelity term (real) + rectification term (GSR mixup).
        X_aug, y_aug = gsr_augment(X, y, self.gsr, self.class_counts)
        if X_aug is None:
            X_all, y_all, weight = X, y, None
        else:
            X_all, y_all = torch.cat([X, X_aug], dim=0), torch.cat([y, y_aug], dim=0)
            weight = None
            if self.gsr.beta != 1.0:
                weight = torch.ones(X_all.shape[0], device=self.device)
                weight[X.shape[0]:] = self.gsr.beta
        for layer in self.layers:
            layer.update_stats(X_all, y_all, weight)
            layer.solve_theta(target_means)

        # Final classifier on pseudo-replay.
        replay_X, replay_Y = self._generate_replay_data()
        H = self._embed(replay_X)
        Y_onehot = torch.zeros(replay_Y.numel(), self.num_classes, device=self.device)
        Y_onehot.scatter_(1, replay_Y.unsqueeze(1), 1.0)

        counts = Y_onehot.sum(dim=0)
        snr = _class_snr((Y_onehot.T @ H) / counts.unsqueeze(1), Y_onehot.T @ H ** 2, counts)
        g = snr / (1.0 + snr)
        g = g / g.mean().clamp(min=1e-6)
        ridge = self.reg / (g + 1e-6)

        if H.shape[0] < self.D:
            self.theta = _woodbury_solve(H, Y_onehot, 1.0 / (ridge + 1e-8))
        else:
            A = H.T @ H
            A.diagonal().add_(ridge)
            self.theta = torch.linalg.solve(A, H.T @ Y_onehot)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=self.device), p=2, dim=1)
        logits = (self._embed(X) @ self.theta) * self.class_weights.unsqueeze(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        return np.array([self.inverse_map[p] for p in preds])
