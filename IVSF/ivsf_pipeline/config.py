from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IVSFConfig:
    seed: int = 1337
    device: str = "cuda"

    # Dataset
    data_root: str = "./data"
    # 待生成指纹的原始样本数量
    num_base_samples: int = 2
    num_workers: int = 2
    # 每次是否随机抽取基础样本
    random_base_sampling: bool = True
    # 随机抽样种子。None 表示每次运行随机；设为整数可复现。
    base_sample_seed: Optional[int] = None
    # 固定基础样本索引。若为 None 则按随机/顺序采样策略选取。
    base_sample_indices: Optional[List[int]] = None

    # Fingerprint generation
    num_candidates_per_sample: int = 6
    top_k_per_sample: int = 3
    steps: int = 100
    lr: float = 0.02
    eps_linf: float = 8.0 / 255.0
    enforce_linf_constraint: bool = True
    candidate_sampling_std: float = 1.0 / 255.0
    tau_js: float = 0.08

    # Multi-objective weights
    lambda_boundary: float = 1.0
    lambda_detect: float = 0.6
    lambda_js: float = 0.8
    lambda_stealth: float = 0.5

    # Boundary-aware screening score
    bu_margin_lambda: float = 1.0
    score_alpha: float = 1.0
    score_beta: float = 1.0
    score_gamma: float = 0.2

    # Outputs
    output_dir: str = "./outputs/quick_run"
    visualization_count: int = 12

    # Offline verification artifacts (label + input-output MI)
    mi_pca_components_input: int = 16
    mi_pca_components_output: int = 8
    mi_ksg_k: int = 3
    mi_threshold: float = 0.05

    # Model
    checkpoint_path: str = "./checkpoints/resnet20_cifar10.pth"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def quick(output_dir: str = "./outputs/quick_run") -> "IVSFConfig":
        cfg = IVSFConfig()
        cfg.output_dir = output_dir
        return cfg

    @staticmethod
    def full(output_dir: str = "./outputs/full_run") -> "IVSFConfig":
        cfg = IVSFConfig()
        cfg.output_dir = output_dir

        # Keep base sample count and final fingerprint count unchanged.
        cfg.num_base_samples = 2
        cfg.top_k_per_sample = 3

        # Use a stronger but still practical generation setup.
        cfg.num_workers = 4
        cfg.random_base_sampling = True
        cfg.base_sample_seed = None
        cfg.base_sample_indices = None
        cfg.num_candidates_per_sample = 24
        cfg.steps = 300
        cfg.lr = 0.01
        cfg.enforce_linf_constraint = False
        cfg.candidate_sampling_std = 1.0 / 255.0
        cfg.visualization_count = 12
        return cfg

    @staticmethod
    def from_overrides(overrides: Dict) -> "IVSFConfig":
        cfg = IVSFConfig()
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
