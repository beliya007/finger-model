# IVSF Fingerprint Pipeline (Initial Version)

This folder contains an initial implementation of IVSF fingerprint generation and screening for ResNet20 + CIFAR-10.

Implemented scope:
- Offline fingerprint generation
- Candidate screening and top-k selection
- Output artifacts: `fingerprints.npz`, `scores.csv`, `meta.json`, `preview.png`
- Placeholder for verification interface metadata (online verification not implemented)

## Quick Start

Run from repository root:

```bash
python IVSF/run_ivsf_fingerprint.py --quick --model-source checkpoint --checkpoint-path BIVBench-main/outputs/cifar10/random_start_models/resnet20/best_0.pth
```

Optional training-first mode:

```bash
python IVSF/run_ivsf_fingerprint.py --quick --model-source train
```

The default output directory is `IVSF/outputs/quick_run`.

## Notes

- This implementation keeps generation logic in `IVSF/ivsf_pipeline` and avoids modifying the BIVBench training stack.
- CIFAR-10 normalization is applied in pipeline preprocessing to match the ResNet20 training setup.
