"""Datasets and class-incremental splits (balanced and long-tailed)."""

import os
import random
import tarfile
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, ImageFolder

DATASETS = ["cifar100", "imagenet-r", "t-imagenet", "cub"]


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
class CUBDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform: Optional[Callable] = None):
        self.transform = transform

        def read(name, cast):
            with open(os.path.join(root_dir, name)) as f:
                return {int(line.split()[0]): cast(line.split()[1]) for line in f}

        paths = read("images.txt", str)
        labels = read("image_class_labels.txt", lambda v: int(v) - 1)
        is_train = read("train_test_split.txt", int)
        self.samples = [(os.path.join(root_dir, "images", paths[i]), labels[i])
                        for i in paths if is_train[i] == int(train)]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return (self.transform(image) if self.transform else image), label


class ImageNetR(ImageFolder):
    """ImageNet-R with a fixed, class-stratified 80/20 train/test split."""

    def find_classes(self, directory):
        classes = sorted(d.name for d in os.scandir(directory)
                         if d.is_dir() and not d.name.startswith("."))
        return classes, {name: i for i, name in enumerate(classes)}

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        per_class = defaultdict(list)
        for i, (_, label) in enumerate(self.samples):
            per_class[label].append(i)

        rng = random.Random(1234)
        selected = []
        for indices in per_class.values():
            rng.shuffle(indices)
            cut = int(0.8 * len(indices))
            selected.extend(indices[:cut] if split == "train" else indices[cut:])
        self.samples = [self.samples[i] for i in selected]
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]


class TinyImageNet(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform: Optional[Callable] = None):
        self.transform = transform
        with open(os.path.join(root_dir, "wnids.txt")) as f:
            class_to_idx = {line.strip(): i for i, line in enumerate(f)}

        self.samples = []
        if train:
            train_dir = os.path.join(root_dir, "train")
            for wnid in os.listdir(train_dir):
                if wnid not in class_to_idx:
                    continue
                folder = os.path.join(train_dir, wnid, "images")
                for name in os.listdir(folder):
                    self.samples.append((os.path.join(folder, name), class_to_idx[wnid]))
        else:
            val_dir = os.path.join(root_dir, "val")
            with open(os.path.join(val_dir, "val_annotations.txt")) as f:
                labels = {p[0]: class_to_idx[p[1]] for p in (line.strip().split("\t") for line in f)}
            for name in os.listdir(os.path.join(val_dir, "images")):
                if name in labels:
                    self.samples.append((os.path.join(val_dir, "images", name), labels[name]))
        # os.listdir order is file-system dependent; sort for reproducibility.
        self.samples.sort()
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return (self.transform(image) if self.transform else image), label


def _download_and_extract(url: str, data_dir: str, archive: str, extracted: str, mode: str):
    if os.path.exists(os.path.join(data_dir, extracted)):
        return
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, archive)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, target)
    with tarfile.open(target, mode) as tar:
        tar.extractall(path=data_dir)
    os.remove(target)


def _require(path: str, name: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found at {path}: download and extract it there (see README).")
    return path


def get_dataset(name: str, split: str, transform: Optional[Callable] = None,
                data_dir: str = "./data") -> Dataset:
    name, train = name.lower(), split.lower() == "train"
    if name == "cifar100":
        return CIFAR100(root=data_dir, train=train, download=True, transform=transform)
    if name == "imagenet-r":
        _download_and_extract("https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
                              data_dir, "imagenet-r.tar", "imagenet-r", "r:")
        return ImageNetR(os.path.join(data_dir, "imagenet-r"), split=split.lower(), transform=transform)
    if name == "cub":
        # https://data.caltech.edu/records/65de6-vp158 (CUB_200_2011.tgz)
        return CUBDataset(_require(os.path.join(data_dir, "CUB_200_2011"), "CUB-200-2011"),
                          train=train, transform=transform)
    if name == "t-imagenet":
        # http://cs231n.stanford.edu/tiny-imagenet-200.zip
        return TinyImageNet(_require(os.path.join(data_dir, "tiny-imagenet-200"), "Tiny-ImageNet"),
                            train=train, transform=transform)
    raise ValueError(f"Unsupported dataset: {name} (choose from {DATASETS})")


# ----------------------------------------------------------------------
# Class-incremental splits
# ----------------------------------------------------------------------
@dataclass
class CILSplit:
    task_classes: List[List[int]]            # classes of each task
    task_indices: List[np.ndarray]           # sample indices of each task
    head_classes: Set[int] = field(default_factory=set)
    tail_classes: Set[int] = field(default_factory=set)
    unused_indices: Dict[int, np.ndarray] = field(default_factory=dict)  # per tail class


def build_cil_split(targets: Sequence[int], n_tasks: int, seed: int, scenario: str = "standard",
                    head_ratio: float = 0.3, tail_shot: int = 5) -> CILSplit:
    """Split a dataset into ``n_tasks`` disjoint class-incremental tasks.

    ``standard``: every class keeps all of its samples.
    ``long_tail``: the first ``head_ratio`` fraction of the (shuffled) classes
    keep all samples (head); the remaining classes keep ``tail_shot`` random
    samples (tail). The class order only depends on ``seed``, so train and test splits built
    with the same seed are aligned.
    """
    targets = np.asarray(targets).astype(int)
    per_class = defaultdict(list)
    for idx, label in enumerate(targets.tolist()):
        per_class[label].append(idx)

    classes = sorted(per_class)
    if len(classes) % n_tasks != 0:
        raise ValueError(f"{len(classes)} classes cannot be split evenly into {n_tasks} tasks.")
    random.Random(seed).shuffle(classes)
    task_classes = [classes[i::n_tasks] for i in range(n_tasks)]
    n_head = int(head_ratio * len(classes))

    if scenario == "standard":
        head, keep = set(classes), {}
    elif scenario == "long_tail":
        head = set(classes[:n_head])
        keep = {c: tail_shot for c in classes[n_head:]}
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    split = CILSplit(task_classes=task_classes, task_indices=[], head_classes=head,
                     tail_classes=set(classes) - head)

    rng = random.Random(seed)
    for cls_list in task_classes:
        indices = []
        for c in cls_list:
            pool = per_class[c]
            if c not in keep or len(pool) <= keep[c]:
                indices.extend(pool)
            else:
                chosen = rng.sample(pool, keep[c])
                indices.extend(chosen)
                split.unused_indices[c] = np.setdiff1d(pool, chosen)
        split.task_indices.append(np.asarray(indices, dtype=np.int64))
    return split
