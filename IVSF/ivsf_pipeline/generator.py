from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import IVSFConfig
from .data import build_cifar10_subset_loader, cifar10_mean_std, normalize_batch
from .io_utils import (
    save_meta,
    save_npz,
    save_one_base_with_three_fingerprints,
    save_prediction_report_md,
    save_preview_figure,
    save_scores_csv,
    save_two_bases_with_three_fingerprints,
)
from .losses import (
    boundary_loss,
    boundary_uncertainty_score,
    detect_loss,
    js_divergence,
    js_range_loss,
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

    def _set_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

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

            l_boundary = boundary_loss(p_adv, y).mean()
            l_detect = detect_loss(p_ref, p_adv).mean()
            l_js = js_range_loss(p_ref, p_adv, self.config.tau_js).mean()
            l_stealth = stealth_loss(delta).mean()

            total = (
                self.config.lambda_boundary * l_boundary
                + self.config.lambda_detect * l_detect
                + self.config.lambda_js * l_js
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

    def _evaluate_candidate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p_ref: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
            p_adv = softmax_probs(self.model(normalize_batch(x_adv, self.mean, self.std)))

            l_boundary = boundary_loss(p_adv, y).mean()
            l_detect = detect_loss(p_ref, p_adv).mean()
            l_js = js_range_loss(p_ref, p_adv, self.config.tau_js).mean()
            l_stealth = stealth_loss(delta).mean()
            total = (
                self.config.lambda_boundary * l_boundary
                + self.config.lambda_detect * l_detect
                + self.config.lambda_js * l_js
                + self.config.lambda_stealth * l_stealth
            )

            s_bu = boundary_uncertainty_score(p_adv, self.config.bu_margin_lambda).mean()
            s_sens = js_divergence(p_ref, p_adv).mean()
            s_stealth = stealth_loss(delta).mean()
            s_total = (
                self.config.score_alpha * s_bu
                + self.config.score_beta * s_sens
                - self.config.score_gamma * s_stealth
            )

        scores = {
            "S_BU": float(s_bu.item()),
            "S_sens": float(s_sens.item()),
            "S_stealth": float(s_stealth.item()),
            "S_total": float(s_total.item()),
            "final_loss": float(total.item()),
            "final_boundary": float(l_boundary.item()),
            "final_detect": float(l_detect.item()),
            "final_js": float(l_js.item()),
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
            num_samples=self.config.num_base_samples,
            num_workers=self.config.num_workers,
            random_subset=self.config.random_base_sampling,
            subset_seed=self.config.base_sample_seed,
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

        start_t = time.time()

        for sample_idx, (x, y) in enumerate(loader):
            dataset_idx = int(sampled_indices[sample_idx])
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                p_ref = softmax_probs(self.model(normalize_batch(x, self.mean, self.std)))

            base_images.append(x[0].detach().cpu().numpy())
            base_labels.append(int(y.item()))
            base_dataset_indices.append(dataset_idx)
            ref_probs.append(p_ref[0].detach().cpu().numpy())

            boundary_delta = self._optimize_boundary_delta(x, y, p_ref)
            boundary_deltas.append(boundary_delta[0].detach().cpu().numpy())

            sample_candidates = []
            sample_candidate_scores = []

            for cand_idx in range(self.config.num_candidates_per_sample):
                delta = self._sample_candidate_delta(boundary_delta)
                x_adv, p_adv, score_dict = self._evaluate_candidate(x, y, p_ref, delta)

                pred_label = int(torch.argmax(p_adv, dim=1).item())

                sample_candidates.append((x_adv, delta, p_adv, score_dict))
                sample_candidate_scores.append(score_dict["S_total"])

                row = {
                    "sample_idx": sample_idx,
                    "dataset_idx": dataset_idx,
                    "candidate_idx": cand_idx,
                    "base_label": int(y.item()),
                    "candidate_pred_label": pred_label,
                    "is_selected": 0,
                    **score_dict,
                }
                scores_rows.append(row)

            order = np.argsort(np.array(sample_candidate_scores))[::-1]
            k = min(self.config.top_k_per_sample, len(order))
            top_idx = order[:k]

            cand_imgs = []
            cand_dels = []
            cand_probs = []
            cand_scores = []
            cand_pred_labels = []

            top_imgs = []
            top_dels = []
            top_probs = []
            top_scores = []
            top_pred_labels = []

            for rank_in_sample, idx in enumerate(top_idx):
                x_adv, delta, p_adv, score_dict = sample_candidates[int(idx)]
                img_np = x_adv[0].detach().cpu().numpy()
                del_np = delta[0].detach().cpu().numpy()
                prob_np = p_adv[0].detach().cpu().numpy()
                score_val = float(score_dict["S_total"])
                pred_label = int(np.argmax(prob_np))

                selected_images.append(img_np)
                selected_deltas.append(del_np)
                selected_probs.append(prob_np)
                selected_scores.append(score_val)
                selected_pred_labels.append(pred_label)
                selected_base_sample_indices.append(sample_idx)
                selected_rank_in_sample.append(rank_in_sample)

                top_imgs.append(img_np)
                top_dels.append(del_np)
                top_probs.append(prob_np)
                top_scores.append(score_val)
                top_pred_labels.append(pred_label)

                for row in scores_rows:
                    if row["sample_idx"] == sample_idx and row["candidate_idx"] == int(idx):
                        row["is_selected"] = 1
                        break

            if top_imgs:
                selected_images_grouped.append(np.stack(top_imgs, axis=0))
                selected_deltas_grouped.append(np.stack(top_dels, axis=0))
                selected_probs_grouped.append(np.stack(top_probs, axis=0))
                selected_scores_grouped.append(np.array(top_scores, dtype=np.float32))
                selected_pred_labels_grouped.append(np.array(top_pred_labels, dtype=np.int64))

            for x_adv, delta, p_adv, score_dict in sample_candidates:
                prob_np = p_adv[0].detach().cpu().numpy()
                cand_imgs.append(x_adv[0].detach().cpu().numpy())
                cand_dels.append(delta[0].detach().cpu().numpy())
                cand_probs.append(prob_np)
                cand_scores.append(float(score_dict["S_total"]))
                cand_pred_labels.append(int(np.argmax(prob_np)))

            all_candidates_images.append(np.stack(cand_imgs, axis=0))
            all_candidates_deltas.append(np.stack(cand_dels, axis=0))
            all_candidates_probs.append(np.stack(cand_probs, axis=0))
            all_candidates_scores.append(np.array(cand_scores, dtype=np.float32))
            all_candidates_pred_labels.append(np.array(cand_pred_labels, dtype=np.int64))

        elapsed = time.time() - start_t

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

        selected_probs_flat = payload["selected_probs"]
        n_selected = int(selected_probs_flat.shape[0])
        mi_ref, mi_per_fingerprint = self._pointwise_mi_from_probs(selected_probs_flat)

        payload["verification_pred_labels"] = payload["selected_pred_labels"].astype(np.int64)
        payload["verification_mi_per_fingerprint"] = mi_per_fingerprint.astype(np.float32)
        payload["verification_mi_ref"] = np.array([mi_ref], dtype=np.float32)
        payload["verification_threshold"] = np.array([self.config.mi_threshold], dtype=np.float32)

        npz_path = output_dir / "fingerprints.npz"
        save_npz(npz_path, payload)

        csv_path = output_dir / "scores.csv"
        save_scores_csv(csv_path, scores_rows)

        verification_rows: List[Dict] = []
        for idx in range(n_selected):
            base_i = int(payload["selected_base_sample_indices"][idx])
            rank_i = int(payload["selected_rank_in_sample"][idx])
            verification_rows.append(
                {
                    "fingerprint_idx": idx,
                    "base_sample_idx": base_i,
                    "dataset_idx": int(payload["base_dataset_indices"][base_i]),
                    "fingerprint_rank": rank_i,
                    "pred_label": int(payload["verification_pred_labels"][idx]),
                    "mi_input_output": float(payload["verification_mi_per_fingerprint"][idx]),
                }
            )

        verification_csv_path = output_dir / "verification_info.csv"
        save_scores_csv(verification_csv_path, verification_rows)

        prediction_report_path = output_dir / "prediction_report.md"
        save_prediction_report_md(
            path=prediction_report_path,
            base_labels=payload["base_labels"],
            base_probs=payload["ref_probs"],
            base_dataset_indices=payload["base_dataset_indices"],
            selected_probs_grouped=payload["selected_probs_grouped"],
            selected_pred_labels_grouped=payload["selected_pred_labels_grouped"],
        )

        meta = {
            "config": asdict(self.config),
            "device_used": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
            "num_processed_samples": len(base_images),
            "base_dataset_indices": base_dataset_indices,
            "random_base_sampling": self.config.random_base_sampling,
            "base_sample_seed": self.config.base_sample_seed,
            "num_candidates_per_sample": self.config.num_candidates_per_sample,
            "top_k_per_sample": self.config.top_k_per_sample,
            "num_selected_fingerprints": n_selected,
            "elapsed_seconds": elapsed,
            "offline_verification": {
                "pred_labels": "selected labels are stored in fingerprints.npz::verification_pred_labels",
                "mi_per_fingerprint": "stored in fingerprints.npz::verification_mi_per_fingerprint",
                "mi_ref": mi_ref,
                "mi_threshold": self.config.mi_threshold,
                "verification_csv": str(verification_csv_path),
            },
            "verification_interface": {
                "status": "placeholder",
                "note": "Online verification is intentionally not implemented in this stage.",
            },
        }
        meta_path = output_dir / "meta.json"
        save_meta(meta_path, meta)

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
            "prediction_report": prediction_report_path,
            "meta": meta_path,
            "preview": fig_path,
            "one_base_three_fingerprints": one_case_path,
            "two_bases_three_fingerprints": two_case_path,
        }
