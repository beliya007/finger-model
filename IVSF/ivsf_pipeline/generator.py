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

    def _single_candidate_optimize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        delta = torch.empty_like(x).uniform_(-self.config.eps_linf, self.config.eps_linf)
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.config.lr)

        history = {"loss": [], "l_boundary": [], "l_detect": [], "l_js": [], "l_stealth": []}

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

            with torch.no_grad():
                delta.clamp_(-self.config.eps_linf, self.config.eps_linf)

            history["loss"].append(float(total.detach().item()))
            history["l_boundary"].append(float(l_boundary.detach().item()))
            history["l_detect"].append(float(l_detect.detach().item()))
            history["l_js"].append(float(l_js.detach().item()))
            history["l_stealth"].append(float(l_stealth.detach().item()))

        with torch.no_grad():
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
            p_adv = softmax_probs(self.model(normalize_batch(x_adv, self.mean, self.std)))

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
            "final_loss": float(history["loss"][-1]),
            "final_boundary": float(history["l_boundary"][-1]),
            "final_detect": float(history["l_detect"][-1]),
            "final_js": float(history["l_js"][-1]),
            "final_stealth": float(history["l_stealth"][-1]),
        }
        return delta.detach(), scores

    def run(self) -> Dict[str, Path]:
        self._set_seed()
        output_dir = self.config.ensure_output_dir()
        # Always start from a clean output directory to avoid stale artifacts.
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

        selected_images: List[np.ndarray] = []
        selected_deltas: List[np.ndarray] = []
        selected_probs: List[np.ndarray] = []
        selected_scores: List[float] = []
        selected_pred_labels: List[int] = []

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

            sample_candidates = []
            sample_candidate_scores = []

            for cand_idx in range(self.config.num_candidates_per_sample):
                delta, score_dict = self._single_candidate_optimize(x, y, p_ref)
                with torch.no_grad():
                    x_adv = torch.clamp(x + delta, 0.0, 1.0)
                    p_adv = softmax_probs(self.model(normalize_batch(x_adv, self.mean, self.std)))

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

            for idx in top_idx:
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

                top_imgs.append(img_np)
                top_dels.append(del_np)
                top_probs.append(prob_np)
                top_scores.append(score_val)
                top_pred_labels.append(pred_label)

                # Mark selected candidates in the CSV rows.
                for row in scores_rows:
                    if (
                        row["sample_idx"] == sample_idx
                        and row["candidate_idx"] == int(idx)
                    ):
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
            "selected_images": np.stack(selected_images, axis=0).astype(np.float32),
            "selected_deltas": np.stack(selected_deltas, axis=0).astype(np.float32),
            "selected_probs": np.stack(selected_probs, axis=0).astype(np.float32),
            "selected_scores": np.array(selected_scores, dtype=np.float32),
            "selected_pred_labels": np.array(selected_pred_labels, dtype=np.int64),
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

        npz_path = output_dir / "fingerprints.npz"
        save_npz(npz_path, payload)

        csv_path = output_dir / "scores.csv"
        save_scores_csv(csv_path, scores_rows)

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
            "elapsed_seconds": elapsed,
            "verification_interface": {
                "status": "placeholder",
                "note": "Online verification is intentionally not implemented in this stage."
            },
        }
        meta_path = output_dir / "meta.json"
        save_meta(meta_path, meta)

        fig_path = output_dir / "preview.png"
        save_preview_figure(
            path=fig_path,
            base_images=payload["base_images"],
            # Use grouped tensors to keep base-fingerprint alignment by sample.
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
            "meta": meta_path,
            "preview": fig_path,
            "one_base_three_fingerprints": one_case_path,
            "two_bases_three_fingerprints": two_case_path,
        }
