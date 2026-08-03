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


def boundary_loss(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    实现公式 (5-2): L_boundary = max(0, f_y - max_{i!=y} f_i)
    """
    # 获取真实类别的 logit: f_y
    f_y = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
    
    # 获取其余类别的最大 logit: max_{i!=y} f_i
    # 创建一个 mask 排除掉 y_true
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, y_true.view(-1, 1), False)
    f_max_other = logits[mask].view(logits.size(0), -1).max(dim=1).values
    
    margin = f_y - f_max_other
    return torch.clamp(margin, min=0.0)


def detect_loss(p_ref: torch.Tensor, p_adv: torch.Tensor) -> torch.Tensor:
    """实现公式 (5-3): L_detect = D_JS(f(x), f(x+delta))"""
    return js_divergence(p_ref, p_adv)


def stealth_loss(delta: torch.Tensor) -> torch.Tensor:
    """实现公式 (5-5): L_stealth = ||delta||_2^2"""
    flat = delta.view(delta.shape[0], -1)
    # 使用平方项，增加对大扰动的惩罚
    return torch.norm(flat, p=2, dim=1).pow(2)


def boundary_uncertainty_score(p_adv: torch.Tensor, margin_lambda: float) -> torch.Tensor:
    ent = entropy_score(p_adv)
    top2 = torch.topk(p_adv, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return ent - margin_lambda * margin
