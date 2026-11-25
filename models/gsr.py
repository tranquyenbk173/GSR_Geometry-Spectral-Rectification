"""Geometry-Spectral Rectification (GSR).

GSR rectifies the Gram matrix of an RLS-based analytic learner by adding
synthetic features obtained with intra-class mixup in the frozen feature space
(Sec. 3.3 of the paper).

Two things are configurable and correspond to the rows of the ablation table:

* geometry
    ``spherical``  z~ = normalize(g * z_i + (1 - g) * z_j)
    ``linear``     z~ = g * z_i + (1 - g) * z_j                   (plain mixup)

* mixing coefficient, g ~ Beta(alpha_c, alpha_c)
    ``fixed``      alpha_c = alpha for every class (default, alpha = 2)
    ``adaptive``   alpha_c = alpha_base
                             + (alpha_max - alpha_base) * exp(-xi * N_c)
                   where N_c is the number of real samples of class c seen so
                   far. In our experiments the shape of the Beta distribution
                   has no measurable effect on accuracy.

* rectification budget M_c, the number of mixed samples of a class with N_c
  real samples in the current task, with T = min(largest class, max_target)
    ``balance``       M_c = max(T - N_c, 0)        fill every class up to T
    ``exp``           M_c = T * exp(-kappa * N_c / T)   smooth count-adaptive decay
    ``uniform``       M_c = M for every class      (class-agnostic)
    ``proportional``  M_c = r * N_c                (class-agnostic, standard mixup)
  ``balance`` is the rule used in all our experiments. The two class-agnostic
  budgets spend the same total number of mixed samples as ``balance`` and only
  differ in how it is allocated across classes.

* sampler, how the M_c rectification features of a class are generated
    ``mixup``            intra-class mixup (GSR, default)
    ``inter_mixup``      mixup with a sample of another class of the task,
                         g ~ Beta(1, 1) folded to [0.5, 1], label of the class
    ``gauss_shrinkage``  N(mu_c, Sigma_c) with a shrunk empirical covariance
    ``gauss_transfer``   N(mu_c, Sigma_head), covariance averaged over the
                         classes of the task with more than 5 samples
  The last three are the feature-augmentation alternatives of Table 6; they use
  the same budget M_c and differ only in the sampling distribution.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


SAMPLERS = ("mixup", "inter_mixup", "gauss_shrinkage", "gauss_transfer")
GAUSS_SHRINKAGE = 1e-4      # weight of the isotropic target in the shrunk covariance
GAUSS_HEAD_MIN = 5          # classes with more samples than this define the transferred covariance
GAUSS_FALLBACK_STD = 0.1    # isotropic noise used when the covariance is not positive definite


@dataclass
class GSRConfig:
    geometry: str = "spherical"      # "spherical" | "linear"
    alpha_mode: str = "fixed"        # "fixed" | "adaptive"
    alpha: float = 2.0               # Beta parameter when alpha_mode == "fixed"
    alpha_base: float = 0.6          # adaptive: alpha_c of data-rich classes
    alpha_max: float = 1.0           # adaptive: alpha_c as N_c -> 0
    xi: float = 5e-3                 # adaptive: decay rate
    beta: float = 1.0                # weight of the rectification term
    max_target: int = 200            # cap on the per-class balancing target
    budget: str = "balance"          # "balance" | "exp" | "uniform" | "proportional"
    budget_kappa: float = 3.0        # decay of the "exp" budget
    distinct_pairs: bool = False     # forbid mixing a sample with itself
    sampler: str = "mixup"           # "mixup" | "inter_mixup" | "gauss_shrinkage" | "gauss_transfer"

    def __post_init__(self):
        if self.geometry not in ("spherical", "linear"):
            raise ValueError(f"Unknown GSR geometry: {self.geometry}")
        if self.alpha_mode not in ("adaptive", "fixed"):
            raise ValueError(f"Unknown GSR alpha mode: {self.alpha_mode}")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"Unknown GSR sampler: {self.sampler}")
        if self.budget not in ("balance", "exp", "uniform", "proportional"):
            raise ValueError(f"Unknown GSR budget: {self.budget}")
        if min(self.alpha, self.alpha_base, self.alpha_max) <= 0:
            raise ValueError("Beta parameters must be positive.")

    @property
    def spherical(self) -> bool:
        return self.geometry == "spherical"


def mixing_intensity(n_c: int, cfg: GSRConfig) -> float:
    """Beta parameter alpha_c of a class with ``n_c`` real samples."""
    if cfg.alpha_mode == "fixed":
        return cfg.alpha
    return cfg.alpha_base + (cfg.alpha_max - cfg.alpha_base) * math.exp(-cfg.xi * n_c)


def rectification_budget(counts: Sequence[int], cfg: GSRConfig) -> List[int]:
    """Number of mixed samples per class, given the real counts in a task."""
    target = min(max(counts), cfg.max_target)
    balance = [max(target - n, 0) for n in counts]
    if cfg.budget == "balance":
        return balance
    if cfg.budget == "exp":
        return [int(round(target * math.exp(-cfg.budget_kappa * n / target))) for n in counts]
    total = sum(balance)
    if cfg.budget == "uniform":
        return [int(round(total / len(counts)))] * len(counts)
    return [int(round(total * n / sum(counts))) for n in counts]      # proportional


def _sample_gaussian(mean: torch.Tensor, cov: torch.Tensor, n: int) -> torch.Tensor:
    """n samples of N(mean, cov); isotropic noise if cov is not positive definite."""
    L, info = torch.linalg.cholesky_ex(cov)
    noise = torch.randn(n, mean.numel(), device=mean.device)
    if int(info) != 0:
        return mean + GAUSS_FALLBACK_STD * noise
    return mean + noise @ L.T


def _class_covariance(z: torch.Tensor) -> torch.Tensor:
    centred = z - z.mean(dim=0)
    return centred.T @ centred / (z.size(0) - 1)


def _head_covariance(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average covariance of the data-rich classes of the task (covariance transfer)."""
    d = features.size(1)
    eye = torch.eye(d, device=features.device)
    covs = [_class_covariance(features[labels == c]) for c, n in
            zip(*[t.tolist() for t in torch.unique(labels, return_counts=True)]) if n > GAUSS_HEAD_MIN]
    if not covs:
        return 0.01 * eye
    return torch.stack(covs).mean(dim=0) + 1e-5 * eye


def gsr_augment(
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg: GSRConfig,
    class_counts: Optional[Dict[int, int]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Generate the rectification set Z_aug for one task.

    Args:
        features: (N, d) real features of the current task. For the spherical
            geometry they are expected to be L2-normalised.
        labels: (N,) integer class ids.
        cfg: GSR configuration.
        class_counts: cumulative number of real samples per class id, used as
            N_c by the adaptive alpha mode. Defaults to the counts inside ``features``.

    Returns:
        (Z_aug, Y_aug), or (None, None) if no class needs rectification.
    """
    device = features.device
    classes, counts = torch.unique(labels, return_counts=True)
    if classes.numel() == 0:
        return None, None
    budget = rectification_budget(counts.tolist(), cfg)

    head_cov = _head_covariance(features, labels) if cfg.sampler == "gauss_transfer" else None
    if cfg.sampler == "inter_mixup" and classes.numel() < 2:
        return None, None

    aug_x, aug_y = [], []
    for c, n_real, n_aug in zip(classes.tolist(), counts.tolist(), budget):
        if n_aug <= 0:
            continue
        z = features[labels == c]

        if cfg.sampler == "mixup":
            i = torch.randint(0, n_real, (n_aug,), device=device)
            if cfg.distinct_pairs and n_real > 1:
                j = (i + torch.randint(1, n_real, (n_aug,), device=device)) % n_real
            else:
                j = torch.randint(0, n_real, (n_aug,), device=device)
            n_c = n_real if class_counts is None else class_counts.get(c, n_real)
            alpha_c = mixing_intensity(n_c, cfg)
            gamma = torch.distributions.Beta(alpha_c, alpha_c).sample((n_aug, 1)).to(device)
            z_new = gamma * z[i] + (1.0 - gamma) * z[j]

        elif cfg.sampler == "inter_mixup":
            others = features[labels != c]
            i = torch.randint(0, n_real, (n_aug,), device=device)
            j = torch.randint(0, others.size(0), (n_aug,), device=device)
            gamma = torch.distributions.Beta(1.0, 1.0).sample((n_aug, 1)).to(device)
            gamma = torch.max(gamma, 1.0 - gamma)          # the class keeps the larger share
            z_new = gamma * z[i] + (1.0 - gamma) * others[j]

        elif cfg.sampler == "gauss_shrinkage":
            d = z.size(1)
            eye = torch.eye(d, device=device)
            if n_real > 1:
                cov = _class_covariance(z)
                cov = (1.0 - GAUSS_SHRINKAGE) * cov + GAUSS_SHRINKAGE * eye * cov.trace() / d
            else:
                cov = 0.01 * eye
            z_new = _sample_gaussian(z.mean(dim=0), cov, n_aug)

        else:                                               # gauss_transfer
            z_new = _sample_gaussian(z.mean(dim=0), head_cov, n_aug)

        if cfg.spherical:
            z_new = F.normalize(z_new, p=2, dim=1)

        aug_x.append(z_new)
        aug_y.append(torch.full((n_aug,), c, device=device, dtype=torch.long))

    if not aug_x:
        return None, None
    return torch.cat(aug_x, dim=0), torch.cat(aug_y, dim=0)


def stable_rank(gram: torch.Tensor, reg: float = 0.0) -> float:
    """sr(G + reg * I) = ||G + reg I||_F^2 / lambda_max(G + reg I)^2."""
    evals = torch.linalg.eigvalsh(gram.double()) + reg
    return float((evals ** 2).sum() / evals.max() ** 2)
