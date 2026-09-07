"""GTSRB 实验产物完整性检查脚本

检查 GTSRB 完整实验流程的所有预期产物是否齐全：

- 模型权重：`checkpoints/gtsrb_six_layer_cnn.pth`
- VMIISF 指纹：`outputs/gtsrb_time_evaluation/ivsf/fingerprints.npz`
- 基线指纹：`outputs/gtsrb_time_evaluation/{misentry,ssf,publiccheck}/fingerprints.npz`
- 表 6-4 时间报告：`outputs/gtsrb_time_evaluation/gtsrb_time_cost_results.{csv,md}`
- 表 6-5 隐蔽性对比：`outputs/gtsrb_method_comparison/method_lpips_js_comparison.{csv,md}`
- 模型损坏检测：`outputs/gtsrb_damage_detection/gtsrb_damage_detection_results.csv`

返回 exit code 0 表示全部齐全，否则 1。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


REQUIRED_PATHS = [
    # GTSRB 训练好的模型
    "checkpoints/gtsrb_six_layer_cnn.pth",
    # VMIISF 指纹（time evaluation preset）
    "outputs/gtsrb_time_evaluation/ivsf/fingerprints.npz",
    "outputs/gtsrb_time_evaluation/ivsf/timing_summary.json",
    "outputs/gtsrb_time_evaluation/ivsf/verification_info.csv",
    "outputs/gtsrb_time_evaluation/ivsf/stealth_metrics.csv",
    # 基线指纹
    "outputs/gtsrb_time_evaluation/misentry/fingerprints.npz",
    "outputs/gtsrb_time_evaluation/misentry/timing_summary.json",
    "outputs/gtsrb_time_evaluation/ssf/fingerprints.npz",
    "outputs/gtsrb_time_evaluation/ssf/timing_summary.json",
    "outputs/gtsrb_time_evaluation/publiccheck/fingerprints.npz",
    "outputs/gtsrb_time_evaluation/publiccheck/timing_summary.json",
    # 表 6-4 时间评估
    "outputs/gtsrb_time_evaluation/gtsrb_time_cost_results.csv",
    "outputs/gtsrb_time_evaluation/gtsrb_time_cost_results.md",
    # 表 6-5 LPIPS / JS 隐蔽性
    "outputs/gtsrb_method_comparison/method_lpips_js_comparison.csv",
    "outputs/gtsrb_method_comparison/method_lpips_js_comparison.md",
    # 表 6-2 模型损坏检测
    "outputs/gtsrb_damage_detection/gtsrb_damage_detection_results.csv",
    "outputs/gtsrb_damage_detection/table_6_2_gtsrb_damage_detection.md",
]


def main() -> int:
    print("=" * 78)
    print(" GTSRB 实验产物完整性检查")
    print("=" * 78)

    missing = []
    present = []

    for rel_path in REQUIRED_PATHS:
        abs_path = ROOT / rel_path
        if abs_path.is_file():
            size = abs_path.stat().st_size
            print("[OK]   %-70s (%7d B)" % (rel_path, size))
            present.append(rel_path)
        else:
            print("[缺失] %-70s" % rel_path)
            missing.append(rel_path)

    print("=" * 78)
    print(" 存在: %d / %d" % (len(present), len(REQUIRED_PATHS)))
    if missing:
        print(" 缺失: %d 个文件" % len(missing))
        print("")
        print("补齐方案：")
        # 根据缺失的文件给建议
        if any("fingerprints" in m for m in missing):
            print("  - 运行 `python run_gtsrb_full_pipeline.py` 自动生成所有指纹与报告")
        if any("time_cost_results" in m for m in missing):
            print("  - 运行 `python create_gtsrb_timing_report.py` 生成时间报告")
        if any("method_lpips_js_comparison" in m for m in missing):
            print("  - 运行 `python run_gtsrb_method_comparison_report.py` 生成隐蔽性报告")
        if any("damage_detection" in m for m in missing):
            print("  - 运行 `python run_gtsrb_damage_detection.py --methods ivsf` 生成损坏检测报告")
        return 1

    print("✓ 所有实验产物均已就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
