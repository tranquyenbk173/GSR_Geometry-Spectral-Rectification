"""Print the configurations selected by tools/sweep_gsr.py as entries of SELECTED_HPARAMS.

    python tools/collect_selected.py --sweeps "sweeps/anacp_gsr_*.json"

Every sweep file contributes the configuration that ranked first on the
validation split (never the test set). Paste the output into
models/anacp_gsr.py.
"""

import argparse
import glob
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps", nargs="+", required=True)
    args = p.parse_args()
    for path in sorted(f for pattern in args.sweeps for f in glob.glob(pattern)):
        sweep = json.load(open(path))
        if "validation" not in sweep or "anacp_logit_power" not in sweep["validation"][0]["config"]:
            continue
        a, best = sweep["args"], sweep["validation"][0]      # ranked by validation accuracy
        cfg = best["config"]
        val = 100 * sum(r["last_acc"] for r in best["runs"]) / len(best["runs"])
        print(f'    ("{a["backbone"]}", "{a["dataset_name"]}"): dict(gsr_beta={cfg["beta"]:g}, '
              f'anacp_logit_power={cfg["anacp_logit_power"]:g}, '
              f'anacp_tail_replay={cfg["anacp_tail_replay"]}),   # validation {val:.2f}')


if __name__ == "__main__":
    main()
