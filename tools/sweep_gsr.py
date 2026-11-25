"""Hyper-parameter sweep for GSR on cached features.

Every option of the search space takes a list of values; the full grid is
evaluated. Configurations are ranked on a validation set held out from the
*training* images (``--val_fraction`` of every class), so the test set is never
used for model selection. Configurations are then grouped by ``--group_by``
(e.g. the rectification budget or the alpha mode); the best configuration of
each group on validation is re-trained with the full protocol and reported on
the test set. Every group therefore gets the same tuning budget. The ridge
parameter ``--reg`` is part of the search space; the baseline without GSR (and
any ``--extra_baselines``) gets its own ``reg`` selected on the same validation
set.

Examples:
    # class-agnostic vs count-adaptive rectification budget
    python tools/sweep_gsr.py --dataset_name cub --group_by budget \\
        --budget proportional uniform balance exp --beta 0.5 1 2 4 8 --max_target 200 500

    # GSR against other samplers, each with its own beta selected on validation
    python tools/sweep_gsr.py --dataset_name imagenet-r --group_by sampler \\
        --sampler mixup inter_mixup gauss_transfer --beta 0.25 1 4 16

    # fixed vs count-dependent Beta parameter
    python tools/sweep_gsr.py --dataset_name cub --group_by alpha_mode \\
        --alpha_mode fixed adaptive --alpha_base 0.3 0.6 1.0 --xi 5e-3 5e-2
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model                              # noqa: E402
from run import add_gsr_args, str2bool                    # noqa: E402
from trainer import build_tasks, load_features, run_cil   # noqa: E402

GSR_KEYS = ["geometry", "alpha_mode", "alpha", "alpha_base", "alpha_max", "xi", "beta",
            "max_target", "budget", "budget_kappa", "distinct_pairs", "sampler"]
ANACP_KEYS = ["anacp_logit_power", "anacp_tail_replay"]   # components of AnaCP + GSR only
KEYS = GSR_KEYS + ANACP_KEYS + ["reg"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--model_name", choices=["gacl_gsr", "anacp_gsr"], default="gacl_gsr")
    p.add_argument("--backbone", default="dinov2")
    p.add_argument("--mocov3_ckpt", default="mocov3-vit-base-300ep.pth")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--feature_dir", default="./features")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--head_ratio", type=float, default=0.3)
    p.add_argument("--tail_shot", type=int, default=5)
    p.add_argument("--D", type=int, default=5000)
    p.add_argument("--num_heads", type=int, default=3)
    p.add_argument("--samples_per_class", type=int, default=100)
    p.add_argument("--shared_cov", type=str2bool, default=True)
    p.add_argument("--baseline_shared_cov", type=str2bool, default=True,
                   help="AnaCP: shared replay covariance of the baseline without GSR")
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="fraction of every class's training images held out for model selection")
    p.add_argument("--extra_baselines", nargs="*", default=[],
                   help="other models (e.g. air) evaluated with reg selected on validation")
    p.add_argument("--group_by", choices=KEYS, default="budget",
                   help="report the best configuration (on validation) of each value of this option")
    p.add_argument("--output", default=None, help="write all results to this JSON file")

    g = p.add_argument_group("search space (lists of values)")
    g.add_argument("--reg", type=float, nargs="+", default=[1e2])
    g.add_argument("--anacp_logit_power", type=float, nargs="+", default=[0.5],
                   help="AnaCP + GSR: exponent of the N_c^(-p) logit re-weighting")
    g.add_argument("--anacp_tail_replay", type=int, nargs="+", default=[4],
                   help="AnaCP + GSR: over-sampling factor of tail classes in the pseudo-replay")
    g.add_argument("--geometry", nargs="+", default=["spherical"])
    g.add_argument("--alpha_mode", nargs="+", default=["fixed"])
    g.add_argument("--alpha", type=float, nargs="+", default=[2.0])
    g.add_argument("--alpha_base", type=float, nargs="+", default=[0.6])
    g.add_argument("--alpha_max", type=float, nargs="+", default=[1.0])
    g.add_argument("--xi", type=float, nargs="+", default=[5e-3])
    g.add_argument("--beta", type=float, nargs="+", default=[1.0])
    g.add_argument("--max_target", type=int, nargs="+", default=[200])
    g.add_argument("--budget", nargs="+", default=["balance"])
    g.add_argument("--budget_kappa", type=float, nargs="+", default=[3.0])
    g.add_argument("--distinct_pairs", type=str2bool, nargs="+", default=[False])
    g.add_argument("--sampler", nargs="+", default=["mixup"],
                   help="mixup (GSR), inter_mixup, gauss_shrinkage, gauss_transfer; use with --group_by sampler")
    return p.parse_args()


def build_grid(args):
    """Cartesian product of the search space, without redundant combinations."""
    configs = []
    for values in itertools.product(*[getattr(args, k) for k in KEYS]):
        cfg = dict(zip(KEYS, values))
        # Collapse options that have no effect for this configuration.
        if cfg["alpha_mode"] == "fixed":
            cfg.update(alpha_base=args.alpha_base[0], alpha_max=args.alpha_max[0], xi=args.xi[0])
        else:
            cfg.update(alpha=args.alpha[0])
        if cfg["sampler"] != "mixup":      # Beta parameters only apply to intra-class mixup
            cfg.update(alpha_mode=args.alpha_mode[0], alpha=args.alpha[0], alpha_base=args.alpha_base[0],
                       alpha_max=args.alpha_max[0], xi=args.xi[0], distinct_pairs=args.distinct_pairs[0])
        if cfg["budget"] != "exp":
            cfg.update(budget_kappa=args.budget_kappa[0])
        if args.model_name != "anacp_gsr":
            cfg.update(anacp_logit_power=args.anacp_logit_power[0], anacp_tail_replay=args.anacp_tail_replay[0])
        if cfg not in configs:
            configs.append(cfg)
    return configs


def describe(cfg):
    if "geometry" not in cfg:
        return f"no GSR, reg={cfg['reg']:g}"
    if cfg["alpha_mode"] == "fixed":
        s = f"fixed Beta({cfg['alpha']:g},{cfg['alpha']:g})"
    else:
        s = f"adaptive base={cfg['alpha_base']:g} max={cfg['alpha_max']:g} xi={cfg['xi']:g}"
    budget = cfg["budget"] + (f"(k={cfg['budget_kappa']:g})" if cfg["budget"] == "exp" else "")
    extra = "" if cfg["geometry"] == "spherical" else " linear"
    if cfg["sampler"] != "mixup":
        s = cfg["sampler"]
    anacp = ""
    if (cfg["anacp_logit_power"], cfg["anacp_tail_replay"]) != (0.5, 4):
        anacp = f" lp={cfg['anacp_logit_power']:g} tr={cfg['anacp_tail_replay']}"
    return (f"reg={cfg['reg']:<7g} {budget:11s} beta={cfg['beta']:g} T={cfg['max_target']} | {s}"
            f"{' distinct' if cfg['distinct_pairs'] else ''}{extra}{anacp}")


def make_run_args(args, seed, model_name, cfg):
    run = argparse.Namespace(**vars(args))
    defaults = argparse.ArgumentParser()
    add_gsr_args(defaults)
    for k, v in vars(defaults.parse_args([])).items():
        setattr(run, k, v)
    run.seed, run.model_name, run.gamma, run.scenario = seed, model_name, 1e-4, "long_tail"
    if not model_name.endswith("_gsr"):
        run.shared_cov = args.baseline_shared_cov
    run.anacp_logit_power, run.anacp_tail_replay = 0.5, 4
    for k, v in (cfg or {}).items():
        setattr(run, k if k == "reg" or k in ANACP_KEYS else f"gsr_{k}", v)
    if cfg is None:
        run.reg = args.reg[0]
    return run


def validation_tasks(args, seed, X, Y):
    """Hold out ``val_fraction`` of each class; build the long-tailed tasks on the rest."""
    rng = np.random.RandomState(1000 + seed)
    val_mask = np.zeros(len(Y), dtype=bool)
    for c in np.unique(Y):
        idx = rng.permutation(np.flatnonzero(Y == c))
        val_mask[idx[: max(1, int(round(args.val_fraction * len(idx))))]] = True
    run = make_run_args(args, seed, args.model_name, None)
    return build_tasks(run, X[~val_mask], Y[~val_mask], X[val_mask], Y[val_mask])


def evaluate(args, seed, model_name, cfg, train_tasks, eval_tasks, split, final_only):
    torch.manual_seed(seed)
    model = get_model(make_run_args(args, seed, model_name, cfg))
    return run_cil(model, train_tasks, eval_tasks, split.head_classes, split.tail_classes,
                   verbose=False, final_only=final_only)


def main():
    args = parse_args()
    X_train, Y_train = load_features(args, "train")
    X_test, Y_test = load_features(args, "test")
    configs = build_grid(args)
    print(f"{len(configs)} configurations x {len(args.seeds)} seeds", flush=True)

    # 1) model selection on held-out training images
    baselines = [args.model_name.replace("_gsr", "")] + list(args.extra_baselines)
    base_val = {(m, r): [] for m in baselines for r in args.reg}
    val = {i: [] for i in range(len(configs))}
    for seed in args.seeds:
        tasks = validation_tasks(args, seed, X_train, Y_train)
        for m, r in base_val:
            base_val[(m, r)].append(evaluate(args, seed, m, {"reg": r}, *tasks, final_only=True))
            print(f"[val] seed {seed} | {m:10s} reg={r:<7g} last={100 * base_val[(m, r)][-1]['last_acc']:.2f}",
                  flush=True)
        for i, cfg in enumerate(configs):
            val[i].append(evaluate(args, seed, args.model_name, cfg, *tasks, final_only=True))
            print(f"[val] seed {seed} | {describe(cfg):75s} last={100 * val[i][-1]['last_acc']:.2f} "
                  f"tail={100 * val[i][-1]['tail_acc']:.2f}", flush=True)

    def mean(runs, key):
        return float(np.mean([r[key] for r in runs]))

    order = sorted(val, key=lambda i: -mean(val[i], "last_acc"))
    print("\nValidation ranking (last accuracy, mean over seeds)")
    for i in order:
        print(f"  {100 * mean(val[i], 'last_acc'):6.2f}  head={100 * mean(val[i], 'head_acc'):6.2f} "
              f"tail={100 * mean(val[i], 'tail_acc'):6.2f}  {describe(configs[i])}")

    # 2) best configuration of each group, re-trained with the full protocol
    finalists = []
    for m in baselines:
        best_reg = max(args.reg, key=lambda r: mean(base_val[(m, r)], "last_acc"))
        finalists.append((m, m, {"reg": best_reg}))
    for value in getattr(args, args.group_by):
        best = next(i for i in order if configs[i][args.group_by] == value)
        finalists.append((f"{args.group_by}={value}", args.model_name, configs[best]))

    test = {name: [] for name, _, _ in finalists}
    for seed in args.seeds:
        tasks = build_tasks(make_run_args(args, seed, args.model_name, None),
                            X_train, Y_train, X_test, Y_test)
        for name, model_name, cfg in finalists:
            test[name].append(evaluate(args, seed, model_name, cfg, *tasks, final_only=False))

    print("\nTest results, full protocol, best configuration of each group on validation "
          "(mean +- std over seeds)")
    for name, _, cfg in finalists:
        runs = test[name]
        stats = "  ".join(f"{k.split('_')[0]}={100 * np.mean([r[k] for r in runs]):.2f}"
                          f"+-{100 * np.std([r[k] for r in runs]):.2f}"
                          for k in ("last_acc", "avg_acc", "head_acc", "tail_acc"))
        print(f"  {name:22s} {stats}  | {describe(cfg)}")

    if args.output:
        out = {"args": vars(args),
               "validation": [{"config": configs[i], "runs": val[i]} for i in order],
               "test": {name: {"config": cfg, "runs": test[name]} for name, _, cfg in finalists}}
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
