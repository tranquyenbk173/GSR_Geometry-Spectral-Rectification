"""Sanity tests on synthetic features: `python -m pytest tests`."""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anacp_gsr import AnaCPGSR                                    # noqa: E402
from models.gacl import GACL                                             # noqa: E402
from models.gsr import GSRConfig, gsr_augment, mixing_intensity, stable_rank  # noqa: E402


def toy_task(seed=0, d=32, counts=(60, 5, 5, 20)):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c, n in enumerate(counts):
        xs.append(torch.randn(n, d, generator=g) + 3.0 * torch.randn(1, d, generator=g))
        ys += [c] * n
    return torch.nn.functional.normalize(torch.cat(xs), dim=1), torch.tensor(ys)


def test_mixing_intensity_matches_eq8():
    cfg = GSRConfig(alpha_mode="adaptive")
    for n in (1, 5, 50, 500):
        assert math.isclose(mixing_intensity(n, cfg), 0.6 + 0.4 * math.exp(-5e-3 * n))
    # tail -> alpha_max, head -> alpha_base
    assert mixing_intensity(0, cfg) == 1.0
    assert abs(mixing_intensity(10 ** 6, cfg) - 0.6) < 1e-9
    assert mixing_intensity(5, GSRConfig(alpha_mode="fixed", alpha=2.0)) == 2.0


def test_augment_balances_classes_on_the_sphere():
    X, y = toy_task()
    X_aug, y_aug = gsr_augment(X, y, GSRConfig())
    counts = torch.bincount(torch.cat([y, y_aug]))
    assert counts.tolist() == [60, 60, 60, 60]
    assert torch.allclose(X_aug.norm(dim=1), torch.ones(len(X_aug)), atol=1e-5)
    # max_target caps the balancing target
    _, y_aug = gsr_augment(X, y, GSRConfig(max_target=10))
    assert torch.bincount(y_aug, minlength=4).tolist() == [0, 5, 5, 0]
    # linear mixup shrinks the norm instead
    X_lin, _ = gsr_augment(X, y, GSRConfig(geometry="linear"))
    assert X_lin.norm(dim=1).max() <= 1 + 1e-5 and X_lin.norm(dim=1).min() < 0.999


def test_augmented_samples_stay_in_class_span():
    X, y = toy_task()
    X_aug, y_aug = gsr_augment(X, y, GSRConfig(distinct_pairs=True))
    Z = X[y == 1]
    proj = Z.T @ torch.linalg.pinv(Z.T)
    A = X_aug[y_aug == 1]
    assert torch.allclose(A @ proj.T, A, atol=1e-4)
    # with distinct pairs no synthetic sample duplicates a real one
    assert (A @ Z.T).max() < 1 - 1e-6


def test_adaptive_reduces_to_fixed_when_alpha_is_constant():
    X, y = toy_task()
    torch.manual_seed(0)
    a, _ = gsr_augment(X, y, GSRConfig(alpha_mode="fixed", alpha=2.0))
    torch.manual_seed(0)
    b, _ = gsr_augment(X, y, GSRConfig(alpha_mode="adaptive", alpha_base=2.0, alpha_max=2.0))
    assert torch.equal(a, b)


def test_gsr_rectifies_gram_and_helps_tail():
    rng = np.random.RandomState(0)
    d, n_cls = 64, 20
    means = rng.randn(n_cls, d)
    def sample(c, n):
        return (means[c] + 1.5 * rng.randn(n, d)).astype(np.float32)
    tasks, Xte, yte = [], [], []
    for t in range(4):
        X, Y = [], []
        for c in range(5 * t, 5 * t + 5):
            n = 200 if c % 5 < 2 else 5
            X.append(sample(c, n)); Y += [c] * n
            Xte.append(sample(c, 100)); yte += [c] * 100
        tasks.append((np.concatenate(X), np.array(Y)))
    Xte, yte = np.concatenate(Xte), np.array(yte)
    tail = np.array([c % 5 >= 2 for c in yte])

    def fit(gsr):
        model = GACL(D=400, reg=1.0, seed=0, device=torch.device("cpu"), gsr=gsr)
        for X, Y in tasks:
            model.update(X, Y)
        pred = model.predict(Xte)
        return model, float(np.mean(pred[tail] == yte[tail]))

    base, acc_base = fit(None)
    for cfg in (GSRConfig(), GSRConfig(alpha_mode="adaptive")):
        model, acc = fit(cfg)
        assert acc > acc_base + 0.1
        assert model.class_counts == base.class_counts      # N_c counts real samples only
    # beta = 0 switches the rectification term off (features are still normalised)
    off, _ = fit(GSRConfig(beta=0.0))
    ref = GACL(D=400, reg=1.0, seed=0, device=torch.device("cpu"), gsr=None)
    ref.normalize_input = True
    for X, Y in tasks:
        ref.update(X, Y)
    assert torch.allclose(off.G, ref.G) and torch.allclose(off.Q, ref.Q)
    assert stable_rank(model.G, 1.0) >= 1.0


def test_anacp_gsr_runs_and_beta_weighting_is_consistent():
    rng = np.random.RandomState(0)
    X = rng.randn(140, 24).astype(np.float32) + 2.0
    Y = np.array([0] * 60 + [1] * 5 + [2] * 60 + [3] * 15)
    preds = []
    for beta in (1.0, 1.0 + 1e-7):   # weighted code path must agree with the unweighted one
        model = AnaCPGSR(D=64, reg=1.0, num_heads=2, seed=0, device=torch.device("cpu"),
                         samples_per_class=20, shared_cov=False, gsr=GSRConfig(beta=beta))
        model.update(X, Y)
        preds.append(model.predict(X))
        assert set(preds[-1].tolist()) <= {0, 1, 2, 3}
    assert np.mean(preds[0] == preds[1]) > 0.97


def test_rectification_budgets():
    from models.gsr import rectification_budget
    counts = [500, 5, 5, 50]
    balance = rectification_budget(counts, GSRConfig())
    assert balance == [0, 195, 195, 150]
    # class-agnostic budgets spend the same total as the balancing rule
    uniform = rectification_budget(counts, GSRConfig(budget="uniform"))
    proportional = rectification_budget(counts, GSRConfig(budget="proportional"))
    assert len(set(uniform)) == 1 and abs(sum(uniform) - sum(balance)) <= len(counts)
    assert abs(sum(proportional) - sum(balance)) <= len(counts)
    assert proportional[0] > 50 * proportional[1]
    # the exponential budget decreases monotonically with the class size
    exp = rectification_budget(counts, GSRConfig(budget="exp"))
    assert exp[1] == exp[2] > exp[3] > exp[0] == 0
    X, y = toy_task()
    _, y_aug = gsr_augment(X, y, GSRConfig(budget="uniform"))
    assert len(set(torch.bincount(y_aug).tolist())) == 1


def test_alternative_samplers_use_the_same_budget():
    x, y = toy_task()
    for sampler in ("inter_mixup", "gauss_shrinkage", "gauss_transfer"):
        torch.manual_seed(0)
        x_aug, y_aug = gsr_augment(x, y, GSRConfig(sampler=sampler))
        assert torch.bincount(y_aug, minlength=4).tolist() == [0, 55, 55, 40]
        assert torch.allclose(x_aug.norm(dim=1), torch.ones(len(x_aug)), atol=1e-5)


def test_ranpac_runs_with_and_without_gsr():
    from models.ranpac import RanPAC
    x, y = toy_task()
    for gsr, search in ((None, False), (GSRConfig(), False), (None, True)):
        model = RanPAC(D=64, reg=1.0, seed=0, device=torch.device("cpu"), gsr=gsr, ridge_search=search)
        model.update(x.numpy(), y.numpy())
        assert (model.predict(x.numpy()) == y.numpy()).mean() > 0.9
