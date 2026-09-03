from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from lpips import LPIPS as LPIPSModel

from .config import IVSFConfig
from .data import build_cifar10_subset_loader, cifar10_mean_std, normalize_batch
from .io_utils import (
    save_meta,
    save_npz,
    save_one_base_with_three_fingerprints,
    save_prediction_report_md,
    save_preview_figure,
    save_scores_csv,
    save_timing_summary,
    save_two_bases_with_three_fingerprints,
)
from .losses import (
    boundary_loss,
    boundary_uncertainty_score,
    detect_loss,
    js_divergence,
    softmax_probs,
    stealth_loss,
)
from .model import load_resnet20_cifar10


class IVSFFingerprintGenerator:
    def __init__(self, config: IVSFConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.model, self.checkpoint_path = load_resnet20_cifar10(
            checkpoint_path=Path(config.checkpoint_path),
            device=self.device,
        )

        self.mean, self.std = cifar10_mean_std(self.device)
        self.lpips_model = LPIPSModel(net="alex", version="0.1").to(self.device)
        self.lpips_model.eval()

    def _set_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _optimize_boundary_delta(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.empty_like(x).uniform_(-self.config.eps_linf, self.config.eps_linf)
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.config.lr)

        for _ in range(self.config.steps):
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
            logits_adv = self.model(normalize_batch(x_adv, self.mean, self.std))
            p_adv = softmax_probs(logits_adv)

            # 使用 Logits 计算边界损失以避免梯度饱和，符合公式 (5-2)
            l_boundary = boundary_loss(logits_adv, y).mean()
            l_detect = detect_loss(p_ref, p_adv).mean()
            l_stealth = stealth_loss(delta).mean()

            total = (
                self.config.lambda_boundary * l_boundary
                + self.config.lambda_detect * l_detect
                + self.config.lambda_stealth * l_stealth
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if self.config.enforce_linf_constraint:
                with torch.no_grad():
                    delta.clamp_(-self.config.eps_linf, self.config.eps_linf)

        return delta.detach()

    def _sample_candidate_delta(self, boundary_delta: torch.Tensor) -> torch.Tensor:
        sampled_delta = boundary_delta + torch.randn_like(boundary_delta) * self.config.candidate_sampling_std
        if self.config.enforce_linf_constraint:
            return sampled_delta.clamp(-self.config.eps_linf, self.config.eps_linf)
        return sampled_delta

    def _compute_lpips(self, base_image: torch.Tensor, fingerprint_image: torch.Tensor) -> float:
        with torch.no_grad():
            base_tensor = base_image.clone().to(self.device)
            fp_tensor = fingerprint_image.clone().to(self.device)
            if base_tensor.ndim == 3:
                base_tensor = base_tensor.unsqueeze(0)
                fp_tensor = fp_tensor.unsqueeze(0)
            base_tensor = (base_tensor * 2.0) - 1.0
            fp_tensor = (fp_tensor * 2.0) - 1.0
            return float(self.lpips_model(base_tensor, fp_tensor).item())

    def _evaluate_candidate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p_ref: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
            logits_adv = self.model(normalize_batch(x_adv, self.mean, self.std))
            p_adv = softmax_probs(logits_adv)

            l_boundary = boundary_loss(logits_adv, y).mean()
            l_detect = detect_loss(p_ref, p_adv).mean()
            l_stealth = stealth_loss(delta).mean()
            total = (
                self.config.lambda_boundary * l_boundary
                + self.config.lambda_detect * l_detect
                + self.config.lambda_stealth * l_stealth
            )

            s_bu = boundary_uncertainty_score(p_adv, self.config.bu_margin_lambda).mean()
            s_sens = js_divergence(p_ref, p_adv).mean()
            s_stealth = stealth_loss(delta).mean()
            s_total = (
                self.config.score_alpha * s_bu
                + self.config.score_beta * s_sens
            )

        lpips_score = self._compute_lpips(x, x_adv)

        scores = {
            "S_BU": float(s_bu.item()),
            "S_sens": float(s_sens.item()),
            "S_stealth": float(s_stealth.item()),
            "S_total": float(s_total.item()),
            "js_to_base": float(s_sens.item()),
            "lpips_to_base": float(lpips_score),
            "final_loss": float(total.item()),
            "final_boundary": float(l_boundary.item()),
            "final_detect": float(l_detect.item()),
            "final_stealth": float(l_stealth.item()),
        }
        return x_adv, p_adv, scores

    @staticmethod
    def _pointwise_mi_from_probs(selected_probs: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
        # Per-fingerprint MI surrogate: KL(p(y|x_i) || p(y)).
        probs = np.clip(selected_probs.astype(np.float64), eps, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        p_marginal = probs.mean(axis=0, keepdims=True)
        p_marginal = np.clip(p_marginal, eps, 1.0)

        local_mi = np.sum(probs * (np.log(probs) - np.log(p_marginal)), axis=1)
        return float(local_mi.mean()), local_mi.astype(np.float32)

    def _predict_probs_for_images(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        x = torch.from_numpy(images).to(device=self.device, dtype=torch.float32)
        probs_all: List[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                xb = x[i : i + batch_size]
                logits = self.model(normalize_batch(xb, self.mean, self.std))
                probs = softmax_probs(logits)
                probs_all.append(probs.detach().cpu().numpy())

        return np.concatenate(probs_all, axis=0).astype(np.float32)

    @staticmethod
    def _build_stealth_metrics(
        base_images: np.ndarray,
        selected_images: np.ndarray,
        selected_base_sample_indices: np.ndarray,
        selected_rank_in_sample: np.ndarray,
        base_dataset_indices: np.ndarray,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        rows: List[Dict] = []
        l_inf_vals: List[float] = []
        l2_vals: List[float] = []
        mae_vals: List[float] = []
        rmse_vals: List[float] = []
        psnr_vals: List[float] = []

        for i in range(int(selected_images.shape[0])):
            base_i = int(selected_base_sample_indices[i])
            rank_i = int(selected_rank_in_sample[i])

            base = base_images[base_i]
            fp = selected_images[i]
            diff = fp - base

            abs_diff = np.abs(diff)
            mse = float(np.mean(diff * diff))
            rmse = float(np.sqrt(mse))
            l_inf = float(np.max(abs_diff))
            l2 = float(np.linalg.norm(diff.reshape(-1), ord=2))
            mae = float(np.mean(abs_diff))
            psnr_db = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

            rows.append(
                {
                    "fingerprint_idx": i,
                    "base_sample_idx": base_i,
                    "dataset_idx": int(base_dataset_indices[base_i]),
                    "fingerprint_rank": rank_i,
                    "l_inf": l_inf,
                    "l2": l2,
                    "mae": mae,
                    "rmse": rmse,
                    "mse": mse,
                    "psnr_db": psnr_db,
                }
            )

            l_inf_vals.append(l_inf)
            l2_vals.append(l2)
            mae_vals.append(mae)
            rmse_vals.append(rmse)
            psnr_vals.append(psnr_db)

        def _stats(values: List[float]) -> Dict[str, float]:
            arr = np.array(values, dtype=np.float64)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        summary = {
            "l_inf": _stats(l_inf_vals),
            "l2": _stats(l2_vals),
            "mae": _stats(mae_vals),
            "rmse": _stats(rmse_vals),
            "psnr_db": _stats(psnr_vals),
        }
        return rows, summary

    def run(self) -> Dict[str, Path]:
        self._set_seed()
        output_dir = self.config.ensure_output_dir()
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        loader, sampled_indices = build_cifar10_subset_loader(
            data_root=self.config.data_root,
            num_samples=self.config.num_screen_samples,
            num_workers=self.config.num_workers,
            random_subset=self.config.random_base_sampling,
            subset_seed=self.config.base_sample_seed,
            fixed_indices=self.config.base_sample_indices,
        )

        scores_rows: List[Dict] = []
        base_images: List[np.ndarray] = []
        base_labels: List[int] = []
        base_dataset_indices: List[int] = []
        ref_probs: List[np.ndarray] = []
        boundary_deltas: List[np.ndarray] = []

        selected_images: List[np.ndarray] = []
        selected_deltas: List[np.ndarray] = []
        selected_probs: List[np.ndarray] = []
        selected_scores: List[float] = []
        selected_js_to_base: List[float] = []
        selected_lpips_to_base: List[float] = []
        selected_pred_labels: List[int] = []
        selected_base_sample_indices: List[int] = []
        selected_rank_in_sample: List[int] = []

        selected_images_grouped: List[np.ndarray] = []
        selected_deltas_grouped: List[np.ndarray] = []
        selected_probs_grouped: List[np.ndarray] = []
        selected_scores_grouped: List[np.ndarray] = []
        selected_pred_labels_grouped: List[np.ndarray] = []

        all_candidates_images: List[np.ndarray] = []
        all_candidates_deltas: List[np.ndarray] = []
        all_candidates_probs: List[np.ndarray] = []
        all_candidates_scores: List[np.ndarray] = []
        all_candidates_pred_labels: List[np.ndarray] = []

        self._synchronize_device()
        start_t = time.perf_counter()
        screening_elapsed = 0.0
        selection_elapsed = 0.0
        optimization_elapsed_by_base: List[float] = []
        generation_elapsed_by_base: List[float] = []
        generation_elapsed_by_fingerprint: List[float] = []

        screening_pool: List[Dict] = []
        for screen_idx, (x, y) in enumerate(loader):
            dataset_idx = int(sampled_indices[screen_idx])
            x = x.to(self.device)
            y = y.to(self.device)

            self._synchronize_device()
            screen_start = time.perf_counter()
            with torch.no_grad():
                p_ref = softmax_probs(self.model(normalize_batch(x, self.mean, self.std)))

            random_delta = torch.empty_like(x).uniform_(-self.config.eps_linf, self.config.eps_linf)
            if self.config.enforce_linf_constraint:
                random_delta = random_delta.clamp(-self.config.eps_linf, self.config.eps_linf)

            _, p_adv_screen, screen_score = self._evaluate_candidate(x, y, p_ref, random_delta)
            self._synchronize_device()
            screening_elapsed += time.perf_counter() - screen_start
            screen_pred_label = int(torch.argmax(p_adv_screen, dim=1).item())

            screening_pool.append(
                {
                    "dataset_idx": dataset_idx,
                    "x": x,
                    "y": y,
                    "p_ref": p_ref,
                    "base_label": int(y.item()),
                    "screen_score": float(screen_score["S_total"]),
                }
            )

            scores_rows.append(
                {
                    "phase": "screen",
                    "sample_idx": screen_idx,
                    "dataset_idx": dataset_idx,
                    "candidate_idx": -1,
                    "base_label": int(y.item()),
                    "candidate_pred_label": screen_pred_label,
                    "is_selected": 0,
                    **screen_score,
                }
            )

        if not screening_pool:
            raise RuntimeError("No samples are available for screening.")

        select_start = time.perf_counter()
        screening_scores = np.array([item["screen_score"] for item in screening_pool], dtype=np.float32)
        screening_order = np.argsort(screening_scores)[::-1]
        n_selected_bases = min(self.config.num_selected_bases, len(screening_pool))
        selected_screen_indices = screening_order[:n_selected_bases]
        selection_elapsed += time.perf_counter() - select_start

        for idx in selected_screen_indices:
            for row in scores_rows:
                if row["phase"] == "screen" and row["sample_idx"] == int(idx):
                    row["is_selected"] = 1
                    break

        for selected_sample_idx, source_idx in enumerate(selected_screen_indices):
            source_item = screening_pool[int(source_idx)]
            dataset_idx = int(source_item["dataset_idx"])
            x = source_item["x"]
            y = source_item["y"]
            p_ref = source_item["p_ref"]

            base_images.append(x[0].detach().cpu().numpy())
            base_labels.append(int(y.item()))
            base_dataset_indices.append(dataset_idx)
            ref_probs.append(p_ref[0].detach().cpu().numpy())

            self._synchronize_device()
            optimize_start = time.perf_counter()
            boundary_delta = self._optimize_boundary_delta(x, y, p_ref)
            self._synchronize_device()
            optimization_elapsed_by_base.append(time.perf_counter() - optimize_start)
            boundary_deltas.append(boundary_delta[0].detach().cpu().numpy())

            cand_imgs = []
            cand_dels = []
            cand_probs = []
            cand_scores = []
            cand_pred_labels = []

            base_generation_start = time.perf_counter()
            for cand_idx in range(self.config.num_candidates_per_sample):
                self._synchronize_device()
                candidate_start = time.perf_counter()
                delta = self._sample_candidate_delta(boundary_delta)
                x_adv, p_adv, score_dict = self._evaluate_candidate(x, y, p_ref, delta)
                self._synchronize_device()
                generation_elapsed_by_fingerprint.append(time.perf_counter() - candidate_start)

                img_np = x_adv[0].detach().cpu().numpy()
                del_np = delta[0].detach().cpu().numpy()
                prob_np = p_adv[0].detach().cpu().numpy()
                score_val = float(score_dict["S_total"])
                pred_label = int(np.argmax(prob_np))

                selected_images.append(img_np)
                selected_deltas.append(del_np)
                selected_probs.append(prob_np)
                selected_scores.append(score_val)
                selected_js_to_base.append(float(score_dict["js_to_base"]))
                selected_lpips_to_base.append(float(score_dict["lpips_to_base"]))
                selected_pred_labels.append(pred_label)
                selected_base_sample_indices.append(selected_sample_idx)
                selected_rank_in_sample.append(cand_idx)

                cand_imgs.append(img_np)
                cand_dels.append(del_np)
                cand_probs.append(prob_np)
                cand_scores.append(score_val)
                cand_pred_labels.append(pred_label)

                scores_rows.append(
                    {
                        "phase": "generation",
                        "sample_idx": selected_sample_idx,
                        "dataset_idx": dataset_idx,
                        "candidate_idx": cand_idx,
                        "base_label": int(y.item()),
                        "candidate_pred_label": pred_label,
                        "is_selected": 1,
                        **score_dict,
                    }
                )
            generation_elapsed_by_base.append(time.perf_counter() - base_generation_start)

            selected_images_grouped.append(np.stack(cand_imgs, axis=0))
            selected_deltas_grouped.append(np.stack(cand_dels, axis=0))
            selected_probs_grouped.append(np.stack(cand_probs, axis=0))
            selected_scores_grouped.append(np.array(cand_scores, dtype=np.float32))
            selected_pred_labels_grouped.append(np.array(cand_pred_labels, dtype=np.int64))

            all_candidates_images.append(np.stack(cand_imgs, axis=0))
            all_candidates_deltas.append(np.stack(cand_dels, axis=0))
            all_candidates_probs.append(np.stack(cand_probs, axis=0))
            all_candidates_scores.append(np.array(cand_scores, dtype=np.float32))
            all_candidates_pred_labels.append(np.array(cand_pred_labels, dtype=np.int64))

        self._synchronize_device()
        generation_flow_elapsed = time.perf_counter() - start_t

        payload = {
            "base_images": np.stack(base_images, axis=0).astype(np.float32),
            "base_labels": np.array(base_labels, dtype=np.int64),
            "base_dataset_indices": np.array(base_dataset_indices, dtype=np.int64),
            "ref_probs": np.stack(ref_probs, axis=0).astype(np.float32),
            "boundary_deltas": np.stack(boundary_deltas, axis=0).astype(np.float32),
            "selected_images": np.stack(selected_images, axis=0).astype(np.float32),
            "selected_deltas": np.stack(selected_deltas, axis=0).astype(np.float32),
            "selected_probs": np.stack(selected_probs, axis=0).astype(np.float32),
            "selected_scores": np.array(selected_scores, dtype=np.float32),
            "selected_js_to_base": np.array(selected_js_to_base, dtype=np.float32),
            "selected_lpips_to_base": np.array(selected_lpips_to_base, dtype=np.float32),
            "selected_pred_labels": np.array(selected_pred_labels, dtype=np.int64),
            "selected_base_sample_indices": np.array(selected_base_sample_indices, dtype=np.int64),
            "selected_rank_in_sample": np.array(selected_rank_in_sample, dtype=np.int64),
            "selected_images_grouped": np.stack(selected_images_grouped, axis=0).astype(np.float32),
            "selected_deltas_grouped": np.stack(selected_deltas_grouped, axis=0).astype(np.float32),
            "selected_probs_grouped": np.stack(selected_probs_grouped, axis=0).astype(np.float32),
            "selected_scores_grouped": np.stack(selected_scores_grouped, axis=0).astype(np.float32),
            "selected_pred_labels_grouped": np.stack(selected_pred_labels_grouped, axis=0).astype(np.int64),
            "candidate_images": np.stack(all_candidates_images, axis=0).astype(np.float32),
            "candidate_deltas": np.stack(all_candidates_deltas, axis=0).astype(np.float32),
            "candidate_probs": np.stack(all_candidates_probs, axis=0).astype(np.float32),
            "candidate_scores": np.stack(all_candidates_scores, axis=0).astype(np.float32),
            "candidate_pred_labels": np.stack(all_candidates_pred_labels, axis=0).astype(np.int64),
        }

        self._synchronize_device()
        verification_start = time.perf_counter()
        selected_probs_flat = payload["selected_probs"]
        n_selected = int(selected_probs_flat.shape[0])
        mi_ref, mi_per_fingerprint = self._pointwise_mi_from_probs(selected_probs_flat)
        rerun_probs = self._predict_probs_for_images(payload["selected_images"])
        rerun_mi_ref, rerun_mi_per_fingerprint = self._pointwise_mi_from_probs(rerun_probs)
        self._synchronize_device()
        verification_elapsed = time.perf_counter() - verification_start

        payload["verification_pred_labels"] = payload["selected_pred_labels"].astype(np.int64)
        payload["verification_mi_per_fingerprint"] = mi_per_fingerprint.astype(np.float32)
        payload["verification_probs_rerun"] = rerun_probs.astype(np.float32)
        payload["verification_mi_per_fingerprint_moin"] = rerun_mi_per_fingerprint.astype(np.float32)
        payload["verification_js_to_base"] = payload["selected_js_to_base"].astype(np.float32)
        payload["verification_lpips_to_base"] = payload["selected_lpips_to_base"].astype(np.float32)
        payload["verification_mi_ref"] = np.array([mi_ref], dtype=np.float32)
        payload["verification_mi_ref_moin"] = np.array([rerun_mi_ref], dtype=np.float32)
        payload["verification_threshold"] = np.array([self.config.mi_threshold], dtype=np.float32)

        npz_path = output_dir / "fingerprints.npz"
        save_npz(npz_path, payload)

        csv_path = output_dir / "scores.csv"
        save_scores_csv(csv_path, scores_rows)

        verification_rows: List[Dict] = []
        for idx in range(n_selected):
            base_i = int(payload["selected_base_sample_indices"][idx])
            rank_i = int(payload["selected_rank_in_sample"][idx])
            verification_mi_value = float(payload["verification_mi_per_fingerprint"][idx])
            verification_mi_moin_value = float(payload["verification_mi_per_fingerprint_moin"][idx])
            verification_rows.append(
                {
                    "fingerprint_idx": idx,
                    "base_sample_idx": base_i,
                    "dataset_idx": int(payload["base_dataset_indices"][base_i]),
                    "fingerprint_rank": rank_i,
                    "pred_label": int(payload["verification_pred_labels"][idx]),
                    "mi_input_output": verification_mi_value,
                    "mi_input_output_moin": verification_mi_moin_value,
                    "abs_mi_diff": abs(verification_mi_value - verification_mi_moin_value),
                    "js_divergence_to_base": float(payload["verification_js_to_base"][idx]),
                    "lpips_to_base": float(payload["verification_lpips_to_base"][idx]),
                }
            )

        verification_csv_path = output_dir / "verification_info.csv"
        save_scores_csv(verification_csv_path, verification_rows)

        stealth_rows, stealth_summary = self._build_stealth_metrics(
            base_images=payload["base_images"],
            selected_images=payload["selected_images"],
            selected_base_sample_indices=payload["selected_base_sample_indices"],
            selected_rank_in_sample=payload["selected_rank_in_sample"],
            base_dataset_indices=payload["base_dataset_indices"],
        )
        stealth_csv_path = output_dir / "stealth_metrics.csv"
        save_scores_csv(stealth_csv_path, stealth_rows)

        prediction_report_path = output_dir / "prediction_report.md"
        save_prediction_report_md(
            path=prediction_report_path,
            base_labels=payload["base_labels"],
            base_probs=payload["ref_probs"],
            base_dataset_indices=payload["base_dataset_indices"],
            selected_probs_grouped=payload["selected_probs_grouped"],
            selected_pred_labels_grouped=payload["selected_pred_labels_grouped"],
        )

        timing_summary = {
            "total_seconds": float(screening_elapsed + selection_elapsed + sum(optimization_elapsed_by_base) + sum(generation_elapsed_by_base) + verification_elapsed),
            "candidate_selection_seconds": float(screening_elapsed + selection_elapsed),
            "candidate_selection_from_dataset_seconds": float(screening_elapsed),
            "candidate_selection_topk_seconds": float(selection_elapsed),
            "fingerprint_generation_seconds": float(sum(optimization_elapsed_by_base) + sum(generation_elapsed_by_base)),
            "verification_seconds": float(verification_elapsed),
            "generation_flow_seconds": float(generation_flow_elapsed),
            "optimization_seconds_per_candidate_base": [float(v) for v in optimization_elapsed_by_base],
            "optimization_seconds_per_candidate_base_mean": float(np.mean(optimization_elapsed_by_base)) if optimization_elapsed_by_base else 0.0,
            "generation_seconds_per_candidate_base": [float(v) for v in generation_elapsed_by_base],
            "generation_seconds_per_candidate_base_mean": float(np.mean(generation_elapsed_by_base)) if generation_elapsed_by_base else 0.0,
            "generation_seconds_per_fingerprint": [float(v) for v in generation_elapsed_by_fingerprint],
            "avg_generation_seconds_per_fingerprint": float(np.mean(generation_elapsed_by_fingerprint)) if generation_elapsed_by_fingerprint else 0.0,
            "num_screen_samples": int(len(screening_pool)),
            "num_selected_bases": int(n_selected_bases),
            "num_candidates_per_sample": int(self.config.num_candidates_per_sample),
            "num_generated_fingerprints": int(n_selected),
        }

        timing_path = output_dir / "timing_summary.json"
        save_timing_summary(timing_path, timing_summary)

        meta = {
            "config": asdict(self.config),
            "device_used": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
            "num_processed_samples": len(base_images),
            "num_screen_samples": len(screening_pool),
            "num_selected_bases": n_selected_bases,
            "base_dataset_indices": base_dataset_indices,
            "random_base_sampling": self.config.random_base_sampling,
            "base_sample_seed": self.config.base_sample_seed,
            "num_candidates_per_sample": self.config.num_candidates_per_sample,
            "num_selected_fingerprints": n_selected,
            "elapsed_seconds": generation_flow_elapsed,
            "timing": timing_summary,
            "offline_verification": {
                "pred_labels": "selected labels are stored in fingerprints.npz::verification_pred_labels",
                "mi_per_fingerprint": "stored in fingerprints.npz::verification_mi_per_fingerprint",
                "js_divergence_to_base": "stored in fingerprints.npz::verification_js_to_base",
                "mi_ref": mi_ref,
                "mi_threshold": self.config.mi_threshold,
                "verification_csv": str(verification_csv_path),
            },
            "stealth_metrics": {
                "metric_note": "computed from effective perturbation (selected_image - base_image) in [0,1] pixel space",
                "csv": str(stealth_csv_path),
                "summary": stealth_summary,
            },
            "verification_interface": {
                "status": "placeholder",
                "note": "Online verification is intentionally not implemented in this stage.",
            },
        }
        meta_path = output_dir / "meta.json"
        save_meta(meta_path, meta)

        print("[IVSF] timing summary:")
        print(json.dumps(timing_summary, ensure_ascii=False, indent=2))

        fig_path = output_dir / "preview.png"
        save_preview_figure(
            path=fig_path,
            base_images=payload["base_images"],
            selected_images=payload["selected_images_grouped"][:, 0],
            deltas=payload["selected_deltas_grouped"][:, 0],
            scores=payload["selected_scores_grouped"][:, 0],
            max_items=self.config.visualization_count,
        )

        one_case_path = output_dir / "one_base_three_fingerprints.png"
        save_one_base_with_three_fingerprints(
            path=one_case_path,
            base_image=payload["base_images"][0],
            fingerprint_images=payload["selected_images_grouped"][0],
            fingerprint_scores=payload["selected_scores_grouped"][0],
            dataset_idx=int(payload["base_dataset_indices"][0]),
            label=int(payload["base_labels"][0]),
        )

        two_case_path = output_dir / "two_bases_three_fingerprints.png"
        save_two_bases_with_three_fingerprints(
            path=two_case_path,
            base_images=payload["base_images"],
            base_dataset_indices=payload["base_dataset_indices"],
            base_labels=payload["base_labels"],
            fingerprint_images_grouped=payload["selected_images_grouped"],
            fingerprint_scores_grouped=payload["selected_scores_grouped"],
            fingerprint_pred_labels_grouped=payload["selected_pred_labels_grouped"],
        )

        return {
            "fingerprints": npz_path,
            "scores": csv_path,
            "verification_info": verification_csv_path,
            "stealth_metrics": stealth_csv_path,
            "prediction_report": prediction_report_path,
            "meta": meta_path,
            "timing_summary": timing_path,
            "preview": fig_path,
            "one_base_three_fingerprints": one_case_path,
            "two_bases_three_fingerprints": two_case_path,
        }
