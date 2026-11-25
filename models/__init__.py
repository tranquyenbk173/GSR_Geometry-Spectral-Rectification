from .gsr import GSRConfig

MODEL_NAMES = ["gacl", "gacl_gsr", "ranpac", "ranpac_gsr", "anacp", "anacp_gsr", "air", "slda", "klda",
               "fecam", "ncm"]


def gsr_config_from_args(args) -> GSRConfig:
    return GSRConfig(
        geometry=args.gsr_geometry,
        alpha_mode=args.gsr_alpha_mode,
        alpha=args.gsr_alpha,
        alpha_base=args.gsr_alpha_base,
        alpha_max=args.gsr_alpha_max,
        xi=args.gsr_xi,
        beta=args.gsr_beta,
        max_target=args.gsr_max_target,
        distinct_pairs=args.gsr_distinct_pairs,
        budget=args.gsr_budget,
        budget_kappa=args.gsr_budget_kappa,
        sampler=getattr(args, "gsr_sampler", "mixup"),
    )


def get_model(args):
    name = args.model_name.lower()

    if name == "gacl":
        from .gacl import GACL
        return GACL(D=args.D, reg=args.reg, seed=args.seed, device=args.device)

    if name == "gacl_gsr":
        from .gacl import GACL
        return GACL(D=args.D, reg=args.reg, seed=args.seed, device=args.device,
                    gsr=gsr_config_from_args(args))

    if name in ("ranpac", "ranpac_gsr"):
        from .ranpac import RanPAC
        return RanPAC(D=args.D, reg=args.reg, seed=args.seed, device=args.device,
                      gsr=gsr_config_from_args(args) if name.endswith("_gsr") else None,
                      ridge_search=args.ranpac_ridge_search)

    if name == "anacp":
        from .anacp import AnaCP
        return AnaCP(D=args.D, reg=args.reg, num_heads=args.num_heads, seed=args.seed,
                     device=args.device, samples_per_class=args.samples_per_class,
                     shared_cov=args.shared_cov)

    if name == "anacp_gsr":
        from .anacp_gsr import AnaCPGSR
        return AnaCPGSR(D=args.D, reg=args.reg, num_heads=args.num_heads, seed=args.seed,
                        device=args.device, samples_per_class=args.samples_per_class,
                        shared_cov=args.shared_cov, gsr=gsr_config_from_args(args),
                        tail_replay_factor=args.anacp_tail_replay,
                        logit_power=args.anacp_logit_power)

    if name == "air":
        from .air import AIR
        return AIR(D=args.D, reg=args.reg, seed=args.seed, device=args.device)

    if name == "slda":
        from .slda import SLDA
        return SLDA(reg=args.reg, device=args.device)

    if name == "klda":
        from .klda import KLDA
        return KLDA(D=args.D, gamma=args.gamma, reg=args.reg, seed=args.seed, device=args.device)

    if name == "fecam":
        from .fecam import FeCAM
        return FeCAM(device=args.device)

    if name in ("ncm", "simplecil"):
        from .ncm import NCM
        return NCM(device=args.device)

    raise ValueError(f"Unknown model name: {args.model_name}")
