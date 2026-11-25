"""Figure: stable rank of the mean-centred feature covariance, with / without GSR.

    python tools/plot_stable_rank.py --datasets imagenet-r t-imagenet cifar100 --seeds 0 1 2

For every task t the script accumulates the weighted second moment of the
features seen so far (real samples, plus GSR samples weighted by beta), removes
the mean,
    C_t = sum_i w_i (h_i - mu)(h_i - mu)^T,   mu = sum_i w_i h_i / sum_i w_i,
and reports sr(C_t) = ||C_t||_F^2 / ||C_t||_2^2. Both variants use L2-normalised
backbone features, so the only difference between the two curves is the
rectification set. ``h`` is the random ReLU projection (GACL) or the first
contrastive-projection layer (AnaCP). Raw numbers go to <out>.json.
Other options are forwarded to run.py (--backbone, --gsr_beta, --reg, ...).
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLUE, ORANGE, INK, MUTED = "#4878a0", "#e05058", "#0b0b0b", "#52514e"
TITLES = {"cifar100": "CIFAR-100", "imagenet-r": "ImageNet-R", "t-imagenet": "Tiny-ImageNet", "cub": "CUB-200"}


def measure(args, datasets, seeds, families, per_task=False, centring="centred"):
    import torch
    from models import gsr_config_from_args
    from probe_stable_rank import probe_anacp, probe_gacl
    from trainer import build_tasks, load_features

    tasks_all = set(range(1, args.n_tasks + 1))
    out = {}
    for dataset in datasets:
        for seed in seeds:
            a = copy.copy(args)
            a.dataset_name, a.seed, a.device = dataset, seed, torch.device(args.device)
            train_tasks, _, _ = build_tasks(a, *load_features(a, "train"), *load_features(a, "test"))
            gsr = gsr_config_from_args(a)
            runs = {}
            if "gacl" in families:
                runs["GACL"] = [r for r in probe_gacl(a, train_tasks, True, None, tasks_all, per_task=per_task) if r[0] == "relu"]
                runs["GACL + GSR"] = [r for r in probe_gacl(a, train_tasks, gsr.spherical, gsr, tasks_all, per_task=per_task)
                                      if r[0] == "relu"]
            if "anacp" in families:
                runs["AnaCP"] = probe_anacp(a, train_tasks, "anacp", tasks_all, per_task)
                runs["AnaCP + GSR"] = probe_anacp(a, train_tasks, "anacp_gsr", tasks_all, per_task)
            for name, rows in runs.items():
                curve = [st[1 if centring == "uncentred" else 0]      # uncentred: sr(G + reg I)
                         for _, c, t, st in sorted(rows, key=lambda r: r[2]) if c == centring]
                out.setdefault(dataset, {}).setdefault(name, []).append(curve)
                print(f"{dataset:<12} seed {seed} {name:<12} " + " ".join(f"{v:6.2f}" for v in curve), flush=True)
    return out


def plot(data, path, n_tasks, ylabel="Stable rank (centred)", keep=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    families = [f for f in ("GACL", "AnaCP") if any(f in d for d in data.values())
                and (keep is None or f.lower() in keep)]
    datasets = list(data)
    fig, axes = plt.subplots(len(families), len(datasets), squeeze=False,
                             figsize=(2.6 * len(datasets), 2.1 * len(families) + 0.5))
    x = np.arange(1, n_tasks + 1)
    for i, fam in enumerate(families):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            for name, color, ls, marker in ((fam, BLUE, "--", "o"), (fam + " + GSR", ORANGE, "-", "s")):
                y = np.asarray(data[ds][name], dtype=float)
                mean, std = y.mean(axis=0), y.std(axis=0)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, mean, ls, color=color, marker=marker, markersize=3.5, linewidth=1.6,
                        markeredgecolor="white", markeredgewidth=0.6, label=name)
            ax.set_ylim(bottom=0)
            ax.set_xticks(x[1::2] if n_tasks > 6 else x)
            ax.grid(True, color="#e6e5e1", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(TITLES.get(ds, ds), fontsize=9.5, color=INK)
            if i == len(families) - 1:
                ax.set_xlabel("Task")
            if j == 0:
                ax.set_ylabel(ylabel)
                ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=2.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {path}.pdf / .png")


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--datasets", nargs="+", default=["imagenet-r", "t-imagenet", "cifar100"])
    extra.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    extra.add_argument("--families", nargs="+", choices=["gacl", "anacp"], default=["gacl", "anacp"])
    extra.add_argument("--per_task", action="store_true",
                       help="statistics of each task's own increment instead of the cumulative matrix")
    extra.add_argument("--centring", choices=["centred", "uncentred"], default="centred",
                       help="uncentred reports sr(G + reg I), the matrix inverted by the solver")
    extra.add_argument("--out", default="assets/stable_rank")
    extra.add_argument("--ylabel", default=None)
    extra.add_argument("--from_json", default=None, help="re-draw from saved numbers, no measurement")
    opts, rest = extra.parse_known_args()

    if opts.from_json:
        data = json.load(open(opts.from_json))
        plot(data, opts.out, len(next(iter(next(iter(data.values())).values()))[0]),
             opts.ylabel or "Stable rank", opts.families)
        return

    sys.argv = [sys.argv[0], "--dataset_name", opts.datasets[0], "--model_name", "gacl_gsr"] + rest
    from run import parse_args
    args = parse_args()
    data = measure(args, opts.datasets, opts.seeds, opts.families, opts.per_task, opts.centring)
    os.makedirs(os.path.dirname(opts.out) or ".", exist_ok=True)
    with open(opts.out + ".json", "w") as f:
        json.dump(data, f, indent=1)
    label = opts.ylabel or ("Per-task stable rank" if opts.per_task else "Cumulative stable rank")
    plot(data, opts.out, args.n_tasks, label, opts.families)


if __name__ == "__main__":
    main()
