from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IVSFConfig:
    seed: int = 1337
    device: str = "cuda"

    # Dataset
    data_root: str = "./data"
    # 筛选阶段的基础样本池大小 (n)
    num_screen_samples: int = 10
    # 筛选后保留的基础样本数 (topk)
    num_selected_bases: int = 2
    num_workers: int = 2
    # 每次是否随机抽取基础样本
    random_base_sampling: bool = True
    # 随机抽样种子。None 表示每次运行随机；设为整数可复现。
    base_sample_seed: Optional[int] = None
    # 固定基础样本索引。若为 None 则按随机/顺序采样策略选取。
    base_sample_indices: Optional[List[int]] = None

    # Fingerprint generation
    num_candidates_per_sample: int = 6
    steps: int = 100
    lr: float = 0.02
    eps_linf: float = 8.0 / 255.0
    enforce_linf_constraint: bool = True
    candidate_sampling_std: float = 1.0 / 255.0

    # Multi-objective weights
    lambda_boundary: float = 1.0
    lambda_detect: float = 0.6
    lambda_stealth: float = 0.5

    # Boundary-aware screening score
    bu_margin_lambda: float = 1.0
    score_alpha: float = 1.0
    score_beta: float = 1.0

    # Outputs
    output_dir: str = "./outputs/quick_run"
    visualization_count: int = 12

    # Offline verification threshold
    mi_threshold: float = 0.05

    # Model
    checkpoint_path: str = "./checkpoints/resnet20_cifar10.pth"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def quick(output_dir: str = "./outputs/quick_run") -> "IVSFConfig":
        cfg = IVSFConfig()
        cfg.output_dir = output_dir
        cfg.num_screen_samples = 5
        cfg.num_selected_bases = 2
        return cfg

    @staticmethod
    def full(output_dir: str = "./outputs/full_run") -> "IVSFConfig":
        cfg = IVSFConfig()
        cfg.output_dir = output_dir

        # Use a larger screening set while keeping selected base count unchanged.
        cfg.num_screen_samples = 300
        cfg.num_selected_bases = 10

        # Use a stronger but still practical generation setup.
        cfg.num_workers = 4
        cfg.random_base_sampling = True
        cfg.base_sample_seed = None
        cfg.base_sample_indices = None
        # 每个基础样本生成的候选指纹数量
        cfg.num_candidates_per_sample = 3

        # 实测更自然外观的一组默认参数（降低模糊感）
        cfg.steps = 220
        cfg.lr = 0.004
        cfg.eps_linf = 3.0 / 255.0
        cfg.enforce_linf_constraint = True
        cfg.candidate_sampling_std = 0.25 / 255.0

        cfg.lambda_boundary = 0.2
        cfg.lambda_detect = 1.5
        cfg.lambda_stealth = 3.0
        cfg.visualization_count = 12
        return cfg

    @staticmethod
    def time_evaluation(output_dir: str = "./outputs/cifar10_time_evaluation/ivsf") -> "IVSFConfig":
        cfg = IVSFConfig.full(output_dir=output_dir)
        cfg.num_screen_samples = 100
        cfg.num_selected_bases = 5
        cfg.num_candidates_per_sample = 3
        cfg.base_sample_seed = 1337
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
