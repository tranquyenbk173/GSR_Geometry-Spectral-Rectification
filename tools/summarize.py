"""Aggregate logs/**/results.json into mean +- std tables.

    python tools/summarize.py --log_dir ./logs
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="./logs")
    args = p.parse_args()

    runs = defaultdict(list)
    for path in glob.glob(os.path.join(args.log_dir, "*", "*", "*", "seed*", "results.json")):
        dataset, backbone, exp, _ = os.path.relpath(path, args.log_dir).split(os.sep)[:4]
        with open(path) as f:
            runs[(dataset, backbone, exp)].append(json.load(f))

    def fmt(values):
        return f"{100 * np.mean(values):6.2f} +- {100 * np.std(values):4.2f}"

    print(f"{'dataset':12s} {'backbone':8s} {'seeds':>5s} {'A_last':>14s} {'A_avg':>14s} "
          f"{'head':>14s} {'tail':>14s}  experiment")
    for (dataset, backbone, exp), rs in sorted(runs.items()):
        cols = [fmt([r[k] for r in rs]) if all(k in r for r in rs) else " " * 14
                for k in ("last_acc", "avg_acc", "head_acc", "tail_acc")]
        print(f"{dataset:12s} {backbone:8s} {len(rs):5d} {'  '.join(cols)}  {exp}")


if __name__ == "__main__":
    main()
