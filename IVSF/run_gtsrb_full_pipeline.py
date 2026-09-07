"""GTSRB 完整实验入口脚本

一键式运行 GTSRB 数据集下完整的 IVSF 时间评估实验流程，并生成所有对比报告。

执行流程：
1. 生成 GTSRB VMIISF 指纹（基于 100 候选 / 5 基础样本 / 15 指纹）
2. 生成 GTSRB 基线指纹（SSF / PublicCheck / MiSentry，各 15 枚）
3. 生成表 6-4 时间评估报告 (CSV / Markdown)
4. 生成表 6-5 LPIPS / JS 隐蔽性对比报告 (CSV / Markdown)

运行环境：
- 默认在 CUDA 上运行（可通过 --device cpu 强制使用 CPU）
- 仓库根目录应为 `d:/桌面/tuili/shiyan-finger-model/`
- 已训练好的 GTSRB 模型：`checkpoints/gtsrb_six_layer_cnn.pth`
- GTSRB 数据：`data/gtsrb/`
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTSRB 一键式实验入口脚本。")
    parser.add_argument("--skip-fingerprints", action="store_true", help="跳过指纹生成阶段（直接基于已有产物生成报告）")
    parser.add_argument("--skip-baselines", action="store_true", help="只生成 IVSF 指纹，不跑基线方法")
    parser.add_argument("--device", type=str, default="cuda", help="强制指定运行设备，例如 cuda / cpu")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/gtsrb_six_layer_cnn.pth")
    parser.add_argument("--time-eval-root", type=str, default="./outputs/gtsrb_time_evaluation")
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--steps", type=int, default=220, help="基线方法的优化步数")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def run_step(args_list):
    """执行一个子命令并实时打印输出。"""
    print("\n>> " + " ".join(args_list))
    result = subprocess.run(args_list, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Step failed (exit={result.returncode}): {args_list}")


def main():
    args = parse_args()
    ivsf_output = Path(args.time_eval_root) / "ivsf"
    misentry_output = Path(args.time_eval_root) / "misentry"
    ssf_output = Path(args.time_eval_root) / "ssf"
    publiccheck_output = Path(args.time_eval_root) / "publiccheck"

    if not args.skip_fingerprints:
        # 1. 生成 VMIISF 指纹 (time_evaluation preset)
        run_step(
            [
                sys.executable,
                "run_ivsf_fingerprint.py",
                "--time-evaluation",
                "--dataset",
                "gtsrb",
                "--checkpoint-path",
                args.checkpoint_path,
                "--output-dir",
                str(ivsf_output),
                "--data-root",
                "./data",
                "--device",
                args.device,
            ]
        )

        if not args.skip_baselines:
            common_baseline_args = [
                "--dataset",
                "gtsrb",
                "--checkpoint-path",
                args.checkpoint_path,
                "--data-root",
                "./data",
                "--device",
                args.device,
                "--num-screen-samples",
                "100",
                "--num-selected-bases",
                "5",
                "--num-candidates-per-base",
                "3",
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
            ]

            for method_name, out_dir in [
                ("misentry", misentry_output),
                ("ssf", ssf_output),
                ("publiccheck", publiccheck_output),
            ]:
                run_step(
                    [
                        sys.executable,
                        "run_baseline_fingerprints.py",
                        "--methods",
                        method_name,
                        "--output-root",
                        str(args.time_eval_root),
                        *common_baseline_args,
                    ]
                )

    # 2. 生成表 6-4 时间评估报告
    run_step(
        [
            sys.executable,
            "create_gtsrb_timing_report.py",
            "--input-root",
            args.time_eval_root,
            "--output-dir",
            args.time_eval_root,
        ]
    )

    # 3. 生成表 6-5 LPIPS / JS 隐蔽性对比报告
    method_comparison_output = Path(args.output_root) / "gtsrb_method_comparison"
    run_step(
        [
            sys.executable,
            "run_gtsrb_method_comparison_report.py",
            "--checkpoint-path",
            args.checkpoint_path,
            "--ivsf-fingerprints",
            str(ivsf_output / "fingerprints.npz"),
            "--misentry-fingerprints",
            str(misentry_output / "fingerprints.npz"),
            "--ssf-fingerprints",
            str(ssf_output / "fingerprints.npz"),
            "--publiccheck-fingerprints",
            str(publiccheck_output / "fingerprints.npz"),
            "--output-dir",
            str(method_comparison_output),
            "--dataset",
            "GTSRB",
        ]
    )

    print("\nGTSRB 一键式实验流程执行完毕。")
    print(f"- 时间评估表: {Path(args.time_eval_root) / 'gtsrb_time_cost_results.md'}")
    print(f"- 方法对比表: {method_comparison_output / 'method_lpips_js_comparison.md'}")


if __name__ == "__main__":
    main()
