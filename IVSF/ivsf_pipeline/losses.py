from __future__ import annotations

import torch


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def entropy_score(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    return -(p * p.log()).sum(dim=1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def boundary_loss(p_adv: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Use absolute top-1 margin to move points close to local boundary.
    p_true = p_adv.gather(1, y_true.view(-1, 1)).squeeze(1)
    top2 = torch.topk(p_adv, k=2, dim=1).values
    p_max = top2[:, 0]
    p_second = top2[:, 1]
    max_non_true = torch.where((p_true - p_max).abs() < 1e-8, p_second, p_max)
    return torch.abs(p_true - max_non_true)


def detect_loss(p_ref: torch.Tensor, p_adv: torch.Tensor) -> torch.Tensor:
    return -js_divergence(p_ref, p_adv)


def js_range_loss(p_ref: torch.Tensor, p_adv: torch.Tensor, tau_js: float) -> torch.Tensor:
    return (js_divergence(p_ref, p_adv) - tau_js).pow(2)


def stealth_loss(delta: torch.Tensor) -> torch.Tensor:
    flat = delta.view(delta.shape[0], -1)
    return torch.norm(flat, p=2, dim=1)


def boundary_uncertainty_score(p_adv: torch.Tensor, margin_lambda: float) -> torch.Tensor:
    ent = entropy_score(p_adv)
    top2 = torch.topk(p_adv, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return ent - margin_lambda * margin
