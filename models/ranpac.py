"""RanPAC classifier on frozen features (McDonnell et al., NeurIPS 2023), with optional GSR.

This is the second phase of RanPAC: a frozen random projection with a ReLU
non-linearity followed by a ridge classifier on accumulated second-order
statistics,

    H   = ReLU(X W^T)
    G_t = G_{t-1} + H^T H,    Q_t = Q_{t-1} + H^T Y,
    W_t = (G_t + lambda * I)^{-1} Q_t.

The first-session parameter-efficient adaptation of the backbone is not used:
as for every other method in this repository the backbone stays frozen.

By default lambda is the shared ``--reg``, the protocol used for all RLS-based
methods in the paper. With ``ridge_search=True`` lambda is re-selected after
every task from 10^-8 ... 10^8 on a held-out 20% of the task, as in the original
implementation.

Under the shared protocol the update is the same ridge system as the GACL
baseline of this repository; the two differ only in the draw of the random
projection.
"""

from typing import Optional

import numpy as np
import torch

from .gacl import GACL
from .gsr import GSRConfig

RIDGES = [10.0 ** k for k in range(-8, 9)]
PROJECTION_SEED_OFFSET = 1000   # RanPAC draws its own projection, independent of GACL's


class RanPAC(GACL):
    def __init__(self, D: int, reg: float, seed: int, device: torch.device,
                 gsr: Optional[GSRConfig] = None, ridge_search: bool = False):
        super().__init__(D=D, reg=reg, seed=seed + PROJECTION_SEED_OFFSET, device=device, gsr=gsr)
        self.ridge_search = ridge_search

    def update(self, X: np.ndarray, Y: np.ndarray):
        super().update(X, Y)
        if self.ridge_search:
            self.reg = self._select_ridge(X, Y)

    def _select_ridge(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Ridge with the lowest squared error on the last 20% of the task."""
        perm = np.random.RandomState(self.seed).permutation(len(Y))
        H = self._project(self._prepare(X)[perm]).double()
        y = torch.tensor([self.label_to_index[v] for v in np.asarray(Y)[perm].tolist()],
                         device=self.device, dtype=torch.long)
        T = self._one_hot(y)
        n_fit = int(0.8 * H.size(0))
        G, Q = H[:n_fit].T @ H[:n_fit], H[:n_fit].T @ T[:n_fit]
        eye = torch.eye(self.D, device=self.device, dtype=torch.float64)
        losses = []
        for ridge in RIDGES:
            W = torch.linalg.solve(G + ridge * eye, Q)
            losses.append(float(((H[n_fit:] @ W - T[n_fit:]) ** 2).mean()))
        return RIDGES[int(np.argmin(losses))]
