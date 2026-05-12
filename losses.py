"""
Specialist loss: CE + regression with asymmetric penalties.
============================================================

LossConfig + SpecialistLoss (v7):
    CE (optional class weights, label smoothing, ignore_index=-1)
    + asymmetric sample weights on directional false positives (gamma default 1.5)
    + regression: SmoothL1 or Gaussian NLL with wrong-sign emphasis.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    cls_weight: float = 1.0
    reg_weight: float = 0.10
    use_class_weights: bool = False
    label_smoothing: float = 0.05
    reg_loss: str = "smooth_l1"  # or "nll"
    huber_beta: float = 0.5
    use_asymmetric_loss: bool = True
    asymmetric_gamma: float = 1.5


class SpecialistLoss(nn.Module):
    def __init__(self, cfg: LossConfig, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.class_weights = class_weights

    def forward(
        self,
        out: dict,
        y_cls: torch.Tensor,
        y_ret: torch.Tensor,
    ) -> dict:
        cfg = self.cfg
        logits = out["logits"]
        mu = out["mu"]

        ce_vec = F.cross_entropy(
            logits,
            y_cls,
            weight=self.class_weights if cfg.use_class_weights else None,
            label_smoothing=cfg.label_smoothing,
            reduction="none",
            ignore_index=-1,
        )
        valid_cls = y_cls >= 0
        if cfg.use_asymmetric_loss and cfg.asymmetric_gamma != 1.0:
            pred = logits.argmax(dim=1)
            fp_dir = ((pred == 2) & (y_cls == 0)) | ((pred == 0) & (y_cls == 2))
            ce_w = torch.where(fp_dir, ce_vec.new_tensor(cfg.asymmetric_gamma), ce_vec.new_tensor(1.0))
            ce_vec = ce_vec * ce_w
        ce_denom = valid_cls.sum().clamp_min(1).to(ce_vec.dtype)
        ce = (ce_vec * valid_cls.float()).sum() / ce_denom

        eps = 1e-6
        reg_mask = valid_cls.float()
        if cfg.reg_loss == "nll":
            ls = out["log_sigma"]
            sig = torch.exp(ls)
            nll_elem = 0.5 * ((y_ret - mu) ** 2 / (sig ** 2 + 1e-8)) + ls
            if cfg.use_asymmetric_loss and cfg.asymmetric_gamma != 1.0:
                wrong_way = ((mu > eps) & (y_ret < -eps)) | ((mu < -eps) & (y_ret > eps))
                rw = torch.where(wrong_way, nll_elem.new_tensor(cfg.asymmetric_gamma),
                                 nll_elem.new_tensor(1.0))
                nll_elem = nll_elem * rw
            reg = (nll_elem * reg_mask).sum() / reg_mask.sum().clamp_min(1.0)
        else:
            sl1 = F.smooth_l1_loss(mu, y_ret, beta=cfg.huber_beta, reduction="none")
            if cfg.use_asymmetric_loss and cfg.asymmetric_gamma != 1.0:
                wrong_way = ((mu > eps) & (y_ret < -eps)) | ((mu < -eps) & (y_ret > eps))
                rw = torch.where(wrong_way, sl1.new_tensor(cfg.asymmetric_gamma),
                                 sl1.new_tensor(1.0))
                sl1 = sl1 * rw
            reg = (sl1 * reg_mask).sum() / reg_mask.sum().clamp_min(1.0)

        total = cfg.cls_weight * ce + cfg.reg_weight * reg
        return {"loss": total, "ce": ce.detach(), "nll": reg.detach()}
