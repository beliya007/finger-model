# GTSRB 一键式实验流水线（Windows PowerShell 版本）
#
# 等价于 `python run_gtsrb_full_pipeline.py`，便于在 IDE / 命令行中直接执行。
# 默认所有路径相对于仓库根目录。

param(
    [switch]$SkipFingerprints,
    [switch]$SkipBaselines,
    [string]$Device = "cuda",
    [string]$CheckpointPath = "./checkpoints/gtsrb_six_layer_cnn.pth",
    [string]$TimeEvalRoot = "./outputs/gtsrb_time_evaluation",
    [string]$OutputRoot = "./outputs",
    [int]$Steps = 220,
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

if (-not $SkipFingerprints) {
    Write-Host "=== Step 1/3: Generate VMIISF fingerprints ===" -ForegroundColor Cyan
    python run_ivsf_fingerprint.py --time-evaluation --dataset gtsrb `
        --checkpoint-path $CheckpointPath `
        --output-dir "$TimeEvalRoot/ivsf" `
        --data-root ./data `
        --device $Device
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not $SkipBaselines) {
        Write-Host "=== Step 2/3: Generate baseline fingerprints ===" -ForegroundColor Cyan
        foreach ($method in @("misentry", "ssf", "publiccheck")) {
            python run_baseline_fingerprints.py `
                --methods $method `
                --output-root $TimeEvalRoot `
                --dataset gtsrb `
                --checkpoint-path $CheckpointPath `
                --data-root ./data `
                --device $Device `
                --num-screen-samples 100 `
                --num-selected-bases 5 `
                --num-candidates-per-base 3 `
                --steps $Steps `
                --seed $Seed
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}

Write-Host "=== Step 3/3: Generate reports ===" -ForegroundColor Cyan

# 表 6-4 时间评估报告
python create_gtsrb_timing_report.py `
    --input-root $TimeEvalRoot `
    --output-dir $TimeEvalRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 表 6-5 LPIPS / JS 隐蔽性报告
python run_gtsrb_method_comparison_report.py `
    --checkpoint-path $CheckpointPath `
    --ivsf-fingerprints "$TimeEvalRoot/ivsf/fingerprints.npz" `
    --misentry-fingerprints "$TimeEvalRoot/misentry/fingerprints.npz" `
    --ssf-fingerprints "$TimeEvalRoot/ssf/fingerprints.npz" `
    --publiccheck-fingerprints "$TimeEvalRoot/publiccheck/fingerprints.npz" `
    --output-dir "$OutputRoot/gtsrb_method_comparison" `
    --dataset GTSRB
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "GTSRB 一键式实验流程执行完毕。" -ForegroundColor Green
Write-Host "- 时间评估表: $TimeEvalRoot/gtsrb_time_cost_results.md"
Write-Host "- 隐蔽性对比: $OutputRoot/gtsrb_method_comparison/method_lpips_js_comparison.md"
