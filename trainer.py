"""Class-incremental training / evaluation loop on frozen features."""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import build_cil_split, get_dataset
from models import get_model


def load_features(args, split: str):
    """Return (features, labels) of a full dataset split, extracting them once.

    The backbone is frozen, so features are cached in ``args.feature_dir`` and
    shared by every method, seed and scenario.
    """
    path = os.path.join(args.feature_dir, f"{args.dataset_name}_{args.backbone}_{split}.npz")
    if os.path.exists(path):
        data = np.load(path)
        return data["X"], data["Y"]

    from backbone import Backbone, get_processor
    print(f"Extracting {split} features of {args.dataset_name} with {args.backbone} ...")
    dataset = get_dataset(args.dataset_name, split, get_processor(args.backbone), args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    X, Y = Backbone(args.backbone, args.device, args.mocov3_ckpt).get_features(loader)
    os.makedirs(args.feature_dir, exist_ok=True)
    np.savez(path, X=X.astype(np.float32), Y=Y.astype(np.int64))
    return X, Y


def build_tasks(args, X_train, Y_train, X_test, Y_test):
    """Slice cached features into per-task train / test sets."""
    train_split = build_cil_split(Y_train, args.n_tasks, args.seed, args.scenario,
                                  args.head_ratio, args.tail_shot)
    test_split = build_cil_split(Y_test, args.n_tasks, args.seed, "standard")
    assert train_split.task_classes == test_split.task_classes

    # Shuffle samples inside each task; the order only affects mixup pairing.
    rng = np.random.RandomState(args.seed)
    train_tasks = []
    for idx in train_split.task_indices:
        idx = rng.permutation(idx)
        train_tasks.append((X_train[idx], Y_train[idx]))
    test_tasks = [(X_test[idx], Y_test[idx]) for idx in test_split.task_indices]
    return train_tasks, test_tasks, train_split


def run_cil(model, train_tasks, test_tasks, head_classes=None, tail_classes=None,
            log_stable_rank=False, verbose=True, final_only=False):
    """Train on the task sequence and evaluate on all seen tasks after each one.

    Returns a dict with the accuracy matrix ``A[t, i]`` (accuracy on task i
    after learning task t), ``last_acc`` and ``avg_acc`` (mean over seen tasks,
    after the last task / averaged over all steps), and head / tail accuracy
    after the last task when the class groups are given. With ``final_only``
    the model is evaluated after the last task only (no ``avg_acc``).
    """
    n_tasks = len(train_tasks)
    A = np.zeros((n_tasks, n_tasks))
    step_acc, stable_ranks = [], []
    preds = None

    for t, (X, Y) in enumerate(train_tasks):
        model.update(X, Y)
        if final_only and t < n_tasks - 1:
            continue
        preds = [model.predict(test_tasks[i][0]) for i in range(t + 1)]
        for i in range(t + 1):
            A[t, i] = np.mean(preds[i] == test_tasks[i][1])
        step_acc.append(float(A[t, : t + 1].mean()))
        msg = f"Task {t + 1:2d}/{n_tasks}: accuracy = {100 * step_acc[-1]:.2f}"
        if log_stable_rank and hasattr(model, "stable_rank"):
            stable_ranks.append(model.stable_rank())
            msg += f" | stable rank = {stable_ranks[-1]:.3f}"
        if verbose:
            print(msg, flush=True)

    results = {
        "acc_matrix": A.tolist(),
        "step_acc": step_acc,
        "last_acc": step_acc[-1],
        "avg_acc": float("nan") if final_only else float(np.mean(step_acc)),
    }
    if stable_ranks:
        results["stable_rank"] = stable_ranks
    if tail_classes:
        y_true = np.concatenate([y for _, y in test_tasks])
        y_pred = np.concatenate(preds)
        for name, group in (("head", head_classes), ("tail", tail_classes)):
            mask = np.isin(y_true, list(group))
            results[f"{name}_acc"] = float(np.mean(y_pred[mask] == y_true[mask]))
    return results


def experiment_name(args) -> str:
    parts = [args.model_name, args.scenario]
    if args.scenario == "long_tail":
        parts += [f"hr{args.head_ratio}", f"ts{args.tail_shot}"]
    parts += [f"D{args.D}", f"reg{args.reg}"]
    if args.model_name.startswith("ranpac") and getattr(args, "ranpac_ridge_search", False):
        parts.append("ridgesearch")
    if args.model_name == "anacp_gsr" and (args.anacp_logit_power != 0.5 or args.anacp_tail_replay != 4):
        parts.append(f"lp{args.anacp_logit_power}_tr{args.anacp_tail_replay}")
    if args.model_name.endswith("_gsr"):
        parts.append(args.gsr_geometry)
        if args.gsr_alpha_mode == "fixed":
            parts.append(f"a{args.gsr_alpha}")
        else:
            parts.append(f"ab{args.gsr_alpha_base}_am{args.gsr_alpha_max}_xi{args.gsr_xi}")
        parts += [f"beta{args.gsr_beta}", f"T{args.gsr_max_target}"]
        if args.gsr_budget != "balance":
            parts.append(args.gsr_budget + (f"{args.gsr_budget_kappa}" if args.gsr_budget == "exp" else ""))
        if args.gsr_distinct_pairs:
            parts.append("distinct")
        if getattr(args, "gsr_sampler", "mixup") != "mixup":
            parts.append(args.gsr_sampler)
    return "_".join(str(p) for p in parts)


def train_incremental(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train, Y_train = load_features(args, "train")
    X_test, Y_test = load_features(args, "test")
    train_tasks, test_tasks, split = build_tasks(args, X_train, Y_train, X_test, Y_test)

    model = get_model(args)
    start = time.time()
    results = run_cil(model, train_tasks, test_tasks, split.head_classes, split.tail_classes,
                      log_stable_rank=args.log_stable_rank)
    results["seconds"] = time.time() - start
    results["args"] = {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(args).items()}

    print(f"Last accuracy: {100 * results['last_acc']:.2f} | Average accuracy: {100 * results['avg_acc']:.2f}")
    if "tail_acc" in results:
        print(f"Head accuracy: {100 * results['head_acc']:.2f} | Tail accuracy: {100 * results['tail_acc']:.2f}")

    log_dir = os.path.join(args.log_dir, args.dataset_name, args.backbone,
                           experiment_name(args), f"seed{args.seed}")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
