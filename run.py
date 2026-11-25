import argparse

import torch

from backbone import BACKBONES
from data_loader import DATASETS
from models import MODEL_NAMES
from trainer import train_incremental


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes")


def add_gsr_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("GSR (used by *_gsr models)")
    g.add_argument("--gsr_geometry", choices=["spherical", "linear"], default="spherical")
    g.add_argument("--gsr_alpha_mode", choices=["fixed", "adaptive"], default="fixed",
                   help="fixed: Beta(alpha, alpha) for every class; adaptive: count-dependent alpha_c")
    g.add_argument("--gsr_alpha", type=float, default=2.0, help="Beta parameter for --gsr_alpha_mode fixed")
    g.add_argument("--gsr_alpha_base", type=float, default=0.6, help="adaptive alpha: value for data-rich classes")
    g.add_argument("--gsr_alpha_max", type=float, default=1.0,
                   help="adaptive alpha: value as N_c -> 0")
    g.add_argument("--gsr_xi", type=float, default=5e-3, help="adaptive alpha: decay rate")
    g.add_argument("--gsr_beta", type=float, default=1.0, help="weight of the rectification term")
    g.add_argument("--gsr_max_target", type=int, default=200,
                   help="cap on the per-class sample count that tail classes are filled up to")
    g.add_argument("--gsr_budget", choices=["balance", "exp", "uniform", "proportional"], default="balance",
                   help="how many mixed samples each class receives (see models/gsr.py)")
    g.add_argument("--gsr_budget_kappa", type=float, default=3.0, help="decay of the exp budget")
    g.add_argument("--gsr_sampler", choices=["mixup", "inter_mixup", "gauss_shrinkage", "gauss_transfer"],
                   default="mixup", help="how rectification features are generated; mixup is GSR, "
                                         "the others are the alternatives of Table 6")
    g.add_argument("--gsr_distinct_pairs", type=str2bool, default=False,
                   help="never mix a sample with itself")


def parse_args():
    p = argparse.ArgumentParser(description="Analytic class-incremental learning with GSR.")
    p.add_argument("--dataset_name", choices=DATASETS, required=True)
    p.add_argument("--model_name", choices=MODEL_NAMES, required=True)
    p.add_argument("--backbone", choices=BACKBONES, default="dinov2")
    p.add_argument("--mocov3_ckpt", type=str, default="mocov3-vit-base-300ep.pth")
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--feature_dir", type=str, default="./features")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)

    # Scenario
    p.add_argument("--scenario", choices=["standard", "long_tail"], default="long_tail")
    p.add_argument("--head_ratio", type=float, default=0.3, help="fraction of head classes")
    p.add_argument("--tail_shot", type=int, default=5, help="training samples per tail class")

    # Analytic learners
    p.add_argument("--D", type=int, default=5000, help="random projection dimension")
    p.add_argument("--reg", type=float, default=1e2, help="ridge regularisation")
    p.add_argument("--num_heads", type=int, default=3, help="AnaCP: number of CP layers")
    p.add_argument("--samples_per_class", type=int, default=100, help="AnaCP: replay samples per class")
    p.add_argument("--shared_cov", type=str2bool, default=True, help="AnaCP: shared replay covariance")
    p.add_argument("--anacp_logit_power", type=float, default=0.5,
                   help="AnaCP + GSR: logits are scaled by N_c^(-power); 0 disables the re-weighting")
    p.add_argument("--anacp_tail_replay", type=int, default=4,
                   help="AnaCP + GSR: over-sampling factor of tail classes in the pseudo-replay")
    p.add_argument("--ranpac_ridge_search", type=str2bool, default=False,
                   help="RanPAC: select the ridge parameter on a held-out split of every task, as in "
                        "the original implementation, instead of using --reg")
    p.add_argument("--gamma", type=float, default=1e-4, help="KLDA: RBF kernel width")

    add_gsr_args(p)
    p.add_argument("--anacp_hparams", choices=["selected", "default"], default="selected",
                   help="AnaCP + GSR: 'selected' uses the validation-selected beta / logit power / tail "
                        "replay of models/anacp_gsr.py for this backbone and dataset (Appendix C.3); "
                        "'default' uses beta=1, p=0.5, r=4. Explicit options take precedence")
    p.add_argument("--log_stable_rank", action="store_true",
                   help="log sr(G + reg I) of the uncentred Gram matrix after every task; the paper reports "
                        "the mean-centred per-task quantity, see tools/plot_stable_rank.py")
    args = p.parse_args()
    if args.model_name == "anacp_gsr":
        import sys
        from models.anacp_gsr import anacp_gsr_hparams
        chosen = anacp_gsr_hparams(args.backbone, args.dataset_name, args.anacp_hparams == "selected")
        for name, value in chosen.items():
            if f"--{name}" not in sys.argv:          # explicit options take precedence
                setattr(args, name, value)
        print(f"[anacp_gsr] {args.anacp_hparams} hyper-parameters for {args.backbone}/{args.dataset_name}: "
              f"beta={args.gsr_beta:g}, logit power={args.anacp_logit_power:g}, "
              f"tail replay={args.anacp_tail_replay}")
    return args


if __name__ == "__main__":
    train_incremental(parse_args())
