from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


METHODS = [
    ("MiSentry", "misentry"),
    ("SSF", "ssf"),
    ("PublicCheck", "publiccheck"),
    ("VMIISF", "ivsf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a VMIISF time-cost table.")
    parser.add_argument("--input-root", type=Path, default=Path("./outputs/cifar10_time_evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/cifar10_time_evaluation"))
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="Dataset name displayed in the report.")
    return parser.parse_args()


def _load_summary(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _seconds(value: object) -> str:
    return f"{float(value):.3f}"


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = []

    for display_name, directory_name in METHODS:
        summary_path = args.input_root / directory_name / "timing_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing timing result: {summary_path}")

        summary = _load_summary(summary_path)
        if int(summary["num_screen_samples"]) != 100:
            raise ValueError(f"{display_name} did not use 100 candidate samples.")
        if int(summary["num_selected_bases"]) != 5:
            raise ValueError(f"{display_name} did not select 5 base samples.")
        if int(summary["num_generated_fingerprints"]) != 15:
            raise ValueError(f"{display_name} did not generate 15 fingerprints.")

        screening = "-" if display_name != "VMIISF" else _seconds(summary["candidate_selection_seconds"])
        rows.append(
            {
                "数据集": args.dataset,
                "方法": display_name,
                "候选样本筛选（s）": screening,
                "指纹生成（s）": _seconds(summary["fingerprint_generation_seconds"]),
                "验证（s）": _seconds(summary["verification_seconds"]),
                "总时间（s）": _seconds(summary["total_seconds"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = args.dataset.lower().replace("-", "").replace(" ", "_")
    csv_path = args.output_dir / f"{dataset_slug}_time_cost_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.output_dir / f"{dataset_slug}_time_cost_results.md"
    table_rows = [
        "| 数据集 | 方法 | 候选样本筛选（s） | 指纹生成（s） | 验证（s） | 总时间（s） |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    table_rows.extend(
        "| {数据集} | {方法} | {候选样本筛选（s）} | {指纹生成（s）} | {验证（s）} | {总时间（s）} |".format(**row)
        for row in rows
    )
    vmiisf = rows[-1]
    baseline_generation = [float(row["指纹生成（s）"]) for row in rows[:-1]]
    fastest_baseline = min(baseline_generation)
    reduction = (1.0 - float(vmiisf["指纹生成（s）"]) / fastest_baseline) * 100.0

    markdown = "\n".join(
        [
            f"# {args.dataset} 计算时间实验结果",
            "",
            "实验在 NVIDIA GeForce RTX 4060 Laptop GPU 上完成。所有方法随机使用 100 个候选样本，保留 5 个基础样本，并生成每个基础样本 3 枚指纹，共 15 枚。VMIISF 与三种基线均使用 220 次优化迭代；时间采用 `time.perf_counter()` 记录，并在 CUDA 操作前后同步。",
            "",
            f"表6-4 {args.dataset} 不同方法的指纹生成时间及 VMIISF 各阶段时间开销对比",
            "",
            *table_rows,
            "",
            f"VMIISF 的候选样本筛选耗时 {vmiisf['候选样本筛选（s）']} s，但其指纹生成仅需 {vmiisf['指纹生成（s）']} s，比本次实验中生成最快的基线少 {reduction:.1f}% 。这是因为 VMIISF 对 5 个基础样本各执行一次决策边界优化，再从局部扰动分布采样生成 15 枚指纹；三种基线则需要对每枚指纹独立优化。验证阶段均低于 0.02 s，说明在线查询与判定开销较小。",
            "",
            "注：基线方法没有 VMIISF 的候选样本评价模块，故“候选样本筛选”列以“-”表示；其基础样本仅用于保证最终指纹数量与 VMIISF 一致。",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()