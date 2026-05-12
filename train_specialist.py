"""
Train the Physics Specialist (Stage 4D)
========================================

Supervised training:
    - Loss = λ_cls·CE + λ_reg·reg(μ)  (defaults 1.0 / 0.1; CE weights [1.7,1,1] short-heavy + asymmetric + short/long-lean)
    - Optimiser: AdamW + cosine LR
    - Time-based train / val split (already in dataset)

Validation metrics:
    - Accuracy (overall + per class)
    - Confusion matrix
    - **Calibration check** — when model says "70% up", how often does
      it actually go up?  Reports calibration MAE per probability bucket.
    - **Tradable subset** — accuracy when model is highly confident
      (filters: max_prob ≥ 0.55 / 0.60 / 0.70).
      This is what matters for trading.

Output:
    runs/<timestamp>/best.pt        weights at lowest val loss
    runs/<timestamp>/last.pt        last-epoch weights
    runs/<timestamp>/config.json
    runs/<timestamp>/metrics.json
    runs/<timestamp>/calibration.txt
    runs/<timestamp>/training.log
"""

from __future__ import annotations
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from losses import CE_CLASS_WEIGHTS
from physics_specialist import PhysicsSpecialist, SpecialistConfig, SpecialistLoss, LossConfig


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class NPZSpecialistDataset(Dataset):
    def __init__(self, npz_path: Path, idx: np.ndarray):
        self.npz_path = npz_path
        self.idx = idx
        # mmap-mode keeps memory low and is fast on local disk
        z = np.load(npz_path, mmap_mode="r", allow_pickle=False)
        self.X = z["X"]
        self.y_ret = z["y_ret"]
        self.y_cls = z["y_cls"]
        self.weights = z["weights"]
        self.times = z["times"]
        self._z = z

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ii = int(self.idx[i])
        x   = self.X[ii].astype(np.float32, copy=False)
        cls = np.int64(self.y_cls[ii])
        # Clip extreme regression targets — beyond ±6 ATR is mostly news shocks
        # (the classification head still distinguishes them as 'up'/'down').
        ret = np.float32(np.clip(self.y_ret[ii], -6.0, 6.0))
        return x, cls, ret


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 800) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, p)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, K: int = 3) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def class_accuracies(cm: np.ndarray) -> list[float]:
    out = []
    for k in range(cm.shape[0]):
        denom = cm[k].sum()
        out.append(float(cm[k, k] / denom) if denom > 0 else 0.0)
    return out


def calibration_report(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> str:
    """probs: confidence in predicted class.  correct: 0/1 hit."""
    bins = np.linspace(0.33, 1.0, n_bins + 1)
    lines = ["bin            n     acc     conf    diff"]
    total_diff = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n < 30:
            lines.append(f"{lo:0.2f}-{hi:0.2f} {n:>8d}    --      --      --")
            continue
        acc  = float(correct[mask].mean())
        conf = float(probs[mask].mean())
        d    = abs(acc - conf)
        total_diff += d * n
        lines.append(f"{lo:0.2f}-{hi:0.2f} {n:>8d}  {acc:0.3f}  {conf:0.3f}  {d:+0.3f}")
    return "\n".join(lines)


def tradable_subset_metrics(
    probs_max: np.ndarray, preds: np.ndarray, y: np.ndarray,
    thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.70),
) -> str:
    lines = ["confidence   coverage   acc_overall   acc_directional"]
    for thr in thresholds:
        mask = probs_max >= thr
        cov = float(mask.mean())
        if mask.sum() < 30:
            lines.append(f"  >={thr:0.2f}    {cov:0.3f}       --            --")
            continue
        acc_overall = float((preds[mask] == y[mask]).mean())
        # directional accuracy = excludes 'noisy' label, asks: when model
        # says up/down, is it right (vs y_cls in {0, 2})?
        dir_mask = mask & (preds != 1)
        if dir_mask.sum() < 10:
            acc_dir = float("nan")
        else:
            acc_dir = float((preds[dir_mask] == y[dir_mask]).mean())
        lines.append(
            f"  >={thr:0.2f}    {cov:0.3f}       {acc_overall:0.3f}         {acc_dir:0.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args):
    device = best_device(args.device)
    print(f"[train] device = {device}")

    z = np.load(args.dataset, mmap_mode="r", allow_pickle=False)
    train_idx = z["train_idx"]
    val_idx   = z["val_idx"]
    n_features = int(z["X"].shape[2])
    window     = int(z["X"].shape[1])
    print(f"[train] dataset: train={len(train_idx):,}  val={len(val_idx):,}  "
          f"window={window}  features={n_features}")

    # Class counts (informational; CE uses losses.CE_CLASS_WEIGHTS when use_class_weights=True)
    cls_counts = np.bincount(z["y_cls"][train_idx].astype(np.int64), minlength=3).astype(np.float64)
    cls_frac = cls_counts / max(1.0, cls_counts.sum())
    print(f"[train] class counts: {cls_counts.astype(int).tolist()}  frac={cls_frac.round(3).tolist()}")

    # Datasets / loaders
    train_ds = NPZSpecialistDataset(Path(args.dataset), train_idx)
    val_ds   = NPZSpecialistDataset(Path(args.dataset), val_idx)

    if args.balanced_sampler:
        sample_w = z["weights"][train_idx]
        sampler = WeightedRandomSampler(
            weights=sample_w, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=False, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    # Model
    cfg = SpecialistConfig(
        n_features=n_features,
        window=window,
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    )
    model = PhysicsSpecialist(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    loss_fn = SpecialistLoss(
        LossConfig(
            cls_weight=1.0,
            reg_weight=args.reg_weight,
            use_class_weights=True,
            reg_loss=args.reg_loss,
            huber_beta=args.huber_beta,
            use_asymmetric_loss=not args.no_asymmetric_loss,
            asymmetric_gamma=args.asymmetric_gamma,
        ),
        class_weights=None,
    )
    print(f"[train] CE class weights (short,noisy,long): {list(CE_CLASS_WEIGHTS)}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Output dir
    out = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] outputs → {out}")
    (out / "config.json").write_text(json.dumps({
        "model": cfg.to_dict(),
        "args":  vars(args),
        "n_params": n_params,
    }, indent=2))

    log_path = out / "training.log"
    log_f = log_path.open("w")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"[train] device={device}  hidden={args.hidden}  layers={args.layers}  "
        f"dropout={args.dropout}  lr={args.lr}  bs={args.batch_size}  epochs={args.epochs}")

    best_val_loss = float("inf")
    best_metrics  = None
    history = []
    epochs_no_improve = 0
    stopped_early = False

    total_steps = len(train_loader) * args.epochs
    step = 0

    for epoch in range(1, args.epochs + 1):
        # ---- TRAIN ----
        model.train()
        t0 = time.time()
        train_loss_sum, train_n = 0.0, 0
        ce_sum, nll_sum = 0.0, 0.0
        for xb, cb, rb in train_loader:
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = cosine_lr(step, total_steps, args.lr)
            optim.zero_grad()
            out_dict = model(xb)
            losses = loss_fn(out_dict, cb, rb)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss_sum += float(losses["loss"].detach()) * xb.size(0)
            ce_sum         += float(losses["ce"].detach())   * xb.size(0)
            nll_sum        += float(losses["nll"].detach())  * xb.size(0)
            train_n += xb.size(0)
            step += 1

        train_loss = train_loss_sum / max(1, train_n)
        train_ce   = ce_sum / max(1, train_n)
        train_nll  = nll_sum / max(1, train_n)

        # ---- VAL ----
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_logits, all_y_cls, all_y_ret = [], [], []
        with torch.no_grad():
            for xb, cb, rb in val_loader:
                xb = xb.to(device); cb = cb.to(device); rb = rb.to(device)
                out_dict = model(xb)
                losses = loss_fn(out_dict, cb, rb)
                val_loss_sum += float(losses["loss"]) * xb.size(0)
                val_n += xb.size(0)
                all_logits.append(out_dict["logits"].cpu().numpy())
                all_y_cls.append(cb.cpu().numpy())
                all_y_ret.append(rb.cpu().numpy())

        val_loss = val_loss_sum / max(1, val_n)
        logits = np.concatenate(all_logits, axis=0)
        probs  = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs  = probs / probs.sum(axis=1, keepdims=True)
        preds  = probs.argmax(axis=1)
        y_cls  = np.concatenate(all_y_cls,  axis=0)

        cm = confusion_matrix(y_cls, preds, K=3)
        accs = class_accuracies(cm)
        overall_acc = float((preds == y_cls).mean())
        prob_max = probs.max(axis=1)
        correct  = (preds == y_cls).astype(np.float32)

        elapsed = time.time() - t0
        log(f"[ep {epoch:>2d}/{args.epochs}] train_loss={train_loss:.4f}  "
            f"(ce={train_ce:.4f}  nll={train_nll:.4f})  "
            f"val_loss={val_loss:.4f}  acc={overall_acc:.3f}  "
            f"down/noisy/up acc = {accs[0]:.2f}/{accs[1]:.2f}/{accs[2]:.2f}  "
            f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_acc":    overall_acc,
            "acc_down":   accs[0],
            "acc_noisy":  accs[1],
            "acc_up":     accs[2],
        })

        # save checkpoints
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "config":      cfg.to_dict()}, out / "best.pt")
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": overall_acc,
                "confusion_matrix": cm.tolist(),
                "per_class_acc": accs,
            }
            log(f"  ↳ new best (val_loss={val_loss:.4f}) → best.pt")
        else:
            epochs_no_improve += 1
            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                log(f"  ↳ early stopping (no val improvement for {epochs_no_improve} epochs, "
                    f"patience={args.early_stopping_patience})")
                stopped_early = True
                ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(ck["model_state"])
                break

        torch.save({"model_state": model.state_dict(),
                    "config":      cfg.to_dict()}, out / "last.pt")

    if stopped_early:
        log(f"[train] resumed weights from best.pt for final eval (stopped at epoch {epoch})")

    # ---- FINAL: full validation pass (best weights if early-stopped) ----
    model.eval()
    val_loss_sum, val_n = 0.0, 0
    all_logits, all_y_cls, all_y_ret = [], [], []
    with torch.no_grad():
        for xb, cb, rb in val_loader:
            xb = xb.to(device)
            cb = cb.to(device)
            rb = rb.to(device)
            out_dict = model(xb)
            losses = loss_fn(out_dict, cb, rb)
            val_loss_sum += float(losses["loss"]) * xb.size(0)
            val_n += xb.size(0)
            all_logits.append(out_dict["logits"].cpu().numpy())
            all_y_cls.append(cb.cpu().numpy())
            all_y_ret.append(rb.cpu().numpy())

    val_loss = val_loss_sum / max(1, val_n)
    logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    y_cls = np.concatenate(all_y_cls, axis=0)
    cm = confusion_matrix(y_cls, preds, K=3)
    accs = class_accuracies(cm)
    overall_acc = float((preds == y_cls).mean())
    prob_max = probs.max(axis=1)
    correct = (preds == y_cls).astype(np.float32)

    # ---- FINAL: detailed validation report ----
    log("\n========== FINAL EVALUATION ==========")
    log(f"(val_loss={val_loss:.4f}  acc={overall_acc:.3f})")
    log("Confusion matrix (rows=true, cols=pred):")
    log("           pred_down  pred_noisy  pred_up")
    log(f"true_down  {cm[0,0]:>9d}  {cm[0,1]:>10d}  {cm[0,2]:>7d}")
    log(f"true_noisy {cm[1,0]:>9d}  {cm[1,1]:>10d}  {cm[1,2]:>7d}")
    log(f"true_up    {cm[2,0]:>9d}  {cm[2,1]:>10d}  {cm[2,2]:>7d}")

    cal = calibration_report(prob_max, correct)
    log("\nCalibration report (max-prob bucket):")
    log(cal)
    (out / "calibration.txt").write_text(cal)

    trad = tradable_subset_metrics(prob_max, preds, y_cls)
    log("\nTradable subset (accuracy when confident):")
    log(trad)
    (out / "tradable_subset.txt").write_text(trad)

    (out / "metrics.json").write_text(json.dumps({
        "history":     history,
        "best":        best_metrics,
        "final_val_acc": overall_acc,
        "stopped_early": stopped_early,
        "epochs_ran":    len(history),
    }, indent=2))

    log_f.close()
    print(f"\n[train] done → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",   required=True)
    ap.add_argument("--epochs",    type=int, default=15)
    ap.add_argument("--batch-size",type=int, default=256)
    ap.add_argument("--lr",        type=float, default=5e-4)
    ap.add_argument("--hidden",    type=int, default=96)
    ap.add_argument("--n-heads",   type=int, default=8, help="Transformer heads (hidden must divide this)")
    ap.add_argument("--ff-mult",   type=int, default=4, help="FF dim = hidden * ff_mult")
    ap.add_argument("--layers",    type=int, default=2)
    ap.add_argument("--dropout",   type=float, default=0.3)
    ap.add_argument("--reg-weight",type=float, default=0.10)
    ap.add_argument("--reg-loss", choices=("smooth_l1", "nll"), default="smooth_l1",
                    help="regression objective on μ (smooth_l1 recommended for trade labels)")
    ap.add_argument("--huber-beta", type=float, default=0.5,
                    help="SmoothL1 beta when reg-loss=smooth_l1")
    ap.add_argument("--workers",   type=int, default=0)
    ap.add_argument("--device",    default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--asymmetric-gamma", type=float, default=1.5)
    ap.add_argument("--no-asymmetric-loss", action="store_true")
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="stop if val_loss does not improve for N epochs (0=disabled). "
        "Restores best.pt weights for final evaluation.",
    )
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
