"""Spectral probe: stable rank of several matrices, with and without GSR.

    python tools/probe_stable_rank.py --dataset_name t-imagenet --backbone dinov2

For every variant (baseline, GSR) the script accumulates, task by task, the
second-moment matrix of
    backbone   the frozen backbone features (d x d)
    relu       the random ReLU projection used by GACL (D x D)
    cp         the Gram matrix of the first AnaCP contrastive-projection layer
each one uncentred (the matrix the solver inverts) and centred (mean removed),
and reports
    sr2   sum(lambda^2) / lambda_max^2     stable rank, ||A||_F^2 / ||A||_2^2
    sr2+  same, on A + reg * I
    sr1   sum(lambda)   / lambda_max       intrinsic dimension, tr(A) / ||A||_2
    erank exp(entropy of lambda / sum(lambda))
Accepts the same options as run.py (--gsr_beta, --gsr_geometry, --reg, --D, ...).
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models import get_model, gsr_config_from_args  # noqa: E402
from models.gsr import gsr_augment  # noqa: E402
from trainer import build_tasks, load_features  # noqa: E402


def spectrum_stats(mat: torch.Tensor, reg: float):
    ev = torch.linalg.eigvalsh(mat.double()).clamp(min=0)
    top = ev.max()
    p = ev / ev.sum()
    erank = torch.exp(-(p[p > 0] * p[p > 0].log()).sum())
    ev_reg = ev + reg
    return (float((ev ** 2).sum() / top ** 2), float((ev_reg ** 2).sum() / ev_reg.max() ** 2),
            float(ev.sum() / top), float(erank))


def legacy_stats(G: torch.Tensor, reg: float):
    """Every spectral quantity found in the original research code, on the uncentred G."""
    ev = torch.linalg.eigvalsh(G.double()).clamp(min=0)
    evr = ev + reg
    return {
        "sr(G)": float((ev ** 2).sum() / ev.max() ** 2),                 # compute_gram_condition_number, gacl*.py
        "sr(G+tI)": float((evr ** 2).sum() / evr.max() ** 2),
        "ln_cond(G+tI)": float(torch.log(evr.max() / evr.min())),        # anacp*.py version (max / min), in log
        "V(G+tI)": float((1.0 / (evr + 0.1)).sum()),                     # measure_trace_inverse(G + reg I), tau=0.1
        "V(G)": float((1.0 / (ev + 0.1)).sum()),                         # measure_trace_inverse(G), tau=0.1
        "rank": int((ev > 1e-6 * ev.max()).sum()),
    }


class Moments:
    """Weighted second moment, first moment and total weight of a feature stream."""

    def __init__(self):
        self.S2, self.s1, self.n = None, None, 0.0

    def add(self, H: torch.Tensor, weight: float = 1.0):
        H = H.double()
        if self.S2 is None:
            self.S2 = torch.zeros(H.shape[1], H.shape[1], dtype=torch.float64, device=H.device)
            self.s1 = torch.zeros(H.shape[1], dtype=torch.float64, device=H.device)
        self.S2 += weight * (H.T @ H)
        self.s1 += weight * H.sum(dim=0)
        self.n += weight * H.shape[0]

    def centred(self) -> torch.Tensor:
        return self.S2 - torch.outer(self.s1, self.s1) / self.n

    def since(self, prev):
        """Moments of the samples added after the snapshot ``prev`` (None = all)."""
        out = Moments()
        if prev is None:
            out.S2, out.s1, out.n = self.S2, self.s1, self.n
        else:
            out.S2, out.s1, out.n = self.S2 - prev[0], self.s1 - prev[1], self.n - prev[2]
        return out

    def snapshot(self):
        return self.S2.clone(), self.s1.clone(), self.n


def probe_gacl(args, train_tasks, normalize: bool, gsr, report_tasks, legacy=False, per_task=False):
    """Replays the statistics accumulated by models/gacl.py."""
    device = args.device
    torch.manual_seed(args.seed)
    W = None
    backbone, relu = Moments(), Moments()
    label_to_index, class_counts, rows = {}, {}, []
    for t, (X, Y) in enumerate(train_tasks):
        prev = {id(m): (m.snapshot() if m.S2 is not None else None) for m in (backbone, relu)}
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        if normalize:
            X = F.normalize(X, p=2, dim=1)
        if W is None:
            W = torch.randn(args.D, X.shape[1], device=device)
        for label in np.unique(Y).tolist():
            label_to_index.setdefault(label, len(label_to_index))
        y = torch.tensor([label_to_index[v] for v in np.asarray(Y).tolist()], device=device)
        for c, n in zip(*[v.tolist() for v in torch.unique(y, return_counts=True)]):
            class_counts[c] = class_counts.get(c, 0) + n

        backbone.add(X)
        relu.add(torch.relu(X @ W.T))
        if gsr is not None:
            X_aug, _ = gsr_augment(X, y, gsr, class_counts)
            if X_aug is not None:
                backbone.add(X_aug, gsr.beta)
                relu.add(torch.relu(X_aug @ W.T), gsr.beta)

        if legacy and t + 1 in report_tasks:
            rows.append(("relu", "legacy", t + 1, legacy_stats(relu.S2, args.reg)))
        elif t + 1 in report_tasks:
            for name, m in (("backbone", backbone), ("relu", relu)):
                if per_task:
                    m = m.since(prev[id(m)])
                rows.append((name, "uncentred", t + 1, spectrum_stats(m.S2, args.reg)))
                rows.append((name, "centred", t + 1, spectrum_stats(m.centred(), args.reg)))
    return rows


def probe_anacp(args, train_tasks, model_name: str, report_tasks, per_task=False):
    args = copy.copy(args)
    args.model_name = model_name
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = get_model(args)
    rows, prev = [], None
    for t, (X, Y) in enumerate(train_tasks):
        model.update(X, Y)
        layer = model.layers[0]
        cur = Moments()
        cur.S2 = layer.A.double().clone()
        cur.n = float(sum(layer.counts.values()))
        cur.s1 = sum(float(layer.counts[c]) * layer.Z_means[c].double() for c in layer.counts)
        m = cur.since(prev) if per_task else cur
        prev = (cur.S2, cur.s1, cur.n)
        if t + 1 in report_tasks:
            rows.append(("cp", "uncentred", t + 1, spectrum_stats(m.S2, args.reg)))
            rows.append(("cp", "centred", t + 1, spectrum_stats(m.centred(), args.reg)))
    return rows


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--report_tasks", type=int, nargs="+", default=None)
    extra.add_argument("--skip_anacp", action="store_true")
    extra.add_argument("--legacy", action="store_true",
                       help="print the spectral quantities of the original research code instead")
    extra.add_argument("--matrix", choices=["all", "relu", "cp"], default="all",
                       help="restrict the printed rows to one matrix")
    opts, rest = extra.parse_known_args()
    if "--model_name" not in rest:
        rest += ["--model_name", "gacl_gsr"]
    sys.argv = [sys.argv[0]] + rest
    from run import parse_args
    args = parse_args()
    args.device = torch.device(args.device)
    report = set(opts.report_tasks or [1, args.n_tasks // 2, args.n_tasks])

    X_train, Y_train = load_features(args, "train")
    X_test, Y_test = load_features(args, "test")
    train_tasks, _, _ = build_tasks(args, X_train, Y_train, X_test, Y_test)
    gsr = gsr_config_from_args(args)

    if opts.legacy:
        print(f"# {args.dataset_name} {args.backbone} seed={args.seed} D={args.D} reg={args.reg:g}  (legacy metrics)")
        for label, norm, cfg in (("GACL", False, None), (f"GACL+GSR beta={gsr.beta:g}", gsr.spherical, gsr)):
            for _, _, task, st in probe_gacl(args, train_tasks, norm, cfg, report, legacy=True):
                if task == min(report):
                    print(f"{'variant':<20}{'task':>4}" + "".join(f"{k:>15}" for k in st))
                print(f"{label:<20}{task:>4}" + "".join(
                    f"{v:>15d}" if isinstance(v, int) else f"{v:>15.4f}" for v in st.values()), flush=True)
        return

    variants = [
        ("baseline (raw feat.)", lambda: probe_gacl(args, train_tasks, False, None, report)),
        ("baseline (L2-norm.)", lambda: probe_gacl(args, train_tasks, True, None, report)),
        (f"GSR beta={gsr.beta:g} {gsr.geometry}",
         lambda: probe_gacl(args, train_tasks, gsr.spherical, gsr, report)),
    ]
    if not opts.skip_anacp:
        variants += [
            ("AnaCP", lambda: probe_anacp(args, train_tasks, "anacp", report)),
            (f"AnaCP+GSR beta={gsr.beta:g}", lambda: probe_anacp(args, train_tasks, "anacp_gsr", report)),
        ]

    print(f"# {args.dataset_name} {args.backbone} seed={args.seed} D={args.D} reg={args.reg:g}")
    print(f"{'variant':<26}{'matrix':<10}{'centring':<11}{'task':>4}{'sr2':>9}{'sr2+':>9}{'sr1':>9}{'erank':>9}")
    for label, fn in variants:
        for name, centring, task, (sr2, sr2r, sr1, er) in sorted(fn(), key=lambda r: (r[0], r[1], r[2])):
            if opts.matrix != "all" and name != opts.matrix:
                continue
            print(f"{label:<26}{name:<10}{centring:<11}{task:>4}{sr2:>9.3f}{sr2r:>9.3f}{sr1:>9.2f}{er:>9.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
