from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass
class IVSFConfig:
    seed: int = 1337
    device: str = "cuda"

    # Dataset
    data_root: str = "~/.wrt/data"
    num_base_samples: int = 100
    num_workers: int = 2

    # Fingerprint generation
    num_candidates_per_sample: int = 6
    top_k_per_sample: int = 3
    steps: int = 50
    lr: float = 0.02
    eps_linf: float = 8.0 / 255.0
    tau_js: float = 0.08

    # Multi-objective weights
    lambda_boundary: float = 1.0
    lambda_detect: float = 0.6
    lambda_js: float = 0.8
    lambda_stealth: float = 0.15

    # Boundary-aware screening score
    bu_margin_lambda: float = 1.0
    score_alpha: float = 1.0
    score_beta: float = 1.0
    score_gamma: float = 0.2

    # Outputs
    output_dir: str = "IVSF/outputs/quick_run"
    visualization_count: int = 12

    # Model source
    model_source: str = "checkpoint"  # checkpoint|train
    checkpoint_path: str = "BIVBench-main/outputs/cifar10/random_start_models/resnet20/best_0.pth"
    train_config_path: str = "BIVBench-main/configs/cifar10/train_configs/resnet20.yaml"
    train_num_models: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def quick(output_dir: str = "IVSF/outputs/quick_run") -> "IVSFConfig":
        cfg = IVSFConfig()
        cfg.output_dir = output_dir
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
