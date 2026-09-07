"""读取 GTSRB 表 6-5 LPIPS / JS 实验结果，并打印格式化报告。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_gtsrb_results() -> list:
    """读取已生成的 GTSRB 表 6-5 CSV 数据。"""
    csv_path = ROOT / "outputs" / "gtsrb_method_comparison" / "method_lpips_js_comparison.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing: {csv_path}")
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_cifar10_results() -> list:
    """读取 CIFAR-10 表 6-5 CSV 数据（如果存在）。"""
    csv_path = ROOT / "outputs" / "cifar10_method_comparison" / "method_lpips_js_comparison.csv"
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    cifar_rows = load_cifar10_results()
    gtsrb_rows = load_gtsrb_results()

    method_order = ["MiSentry", "SSF", "PublicCheck", "VMIISF"]

    print("=" * 90)
    print("表6-5 指纹输入和输出隐蔽性比较 (LPIPS / JS)")
    print("=" * 90)

    table_lines = [
        "| Dataset | Method | LPIPS | JS |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset_name, dataset_rows in [("CIFAR-10", cifar_rows), ("GTSRB", gtsrb_rows)]:
        if not dataset_rows:
            table_lines.append(f"| {dataset_name} | (无数据) | - | - |")
            continue
        for method in method_order:
            matched = [r for r in dataset_rows if r["Method"] == method]
            if matched:
                r = matched[0]
                lpips = float(r["LPIPS"])
                js = float(r["JS"])
                table_lines.append("| {} | {} | {:.6f} | {:.6f} |".format(dataset_name, method, lpips, js))
            else:
                table_lines.append(f"| {dataset_name} | {method} | (无数据) | (无数据) |")

    md_table = "\n".join(table_lines)
    print(md_table)
    print("=" * 90)

    out_path = ROOT / "outputs" / "table_6_5_lpips_js.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# 表6-5 指纹输入和输出隐蔽性比较\n\nLPIPS 越低表示指纹与原始样本在感知空间中距离越小（视觉一致性越高），JS 越低表示模型输出概率分布相对原始样本的变化越小（输出空间隐蔽性越高）。\n\n" + md_table + "\n",
        encoding="utf-8",
    )
    print(f"已写入: {out_path}")

    csv_path = ROOT / "outputs" / "table_6_5_lpips_js.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Method", "LPIPS", "JS"])
        for dataset_name, dataset_rows in [("CIFAR-10", cifar_rows), ("GTSRB", gtsrb_rows)]:
            for method in method_order:
                matched = [r for r in dataset_rows if r["Method"] == method]
                if matched:
                    r = matched[0]
                    writer.writerow([dataset_name, method, r["LPIPS"], r["JS"]])
                else:
                    writer.writerow([dataset_name, method, "-", "-"])
    print(f"已写入: {csv_path}")


if __name__ == "__main__":
    main()
