"""
Walk-Forward Validation (Stage 4G / v7)
======================================

v7 stack (v7.1 defaults): causal Transformer specialist, asymmetric loss,
validation checkpoint scored as 0.7*PF + 0.3*avg_R with preference for validation
max drawdown strictly under 8 percent when viable; **`--min-bt-trades` default 30** (v7.2),
optional stressed spread during checkpoint BT (`--no-stress-spread-bt` to disable).

Trains independent models on K folds and back-tests each on its own OOS slice.
Reports per-fold metrics + aggregate so we can verify the result is robust,
not the artefact of one lucky period.

Layout:
    fold k (k = 1..K):
        train: bars  [start  .. cut_k_train]
        val  : bars  [cut_k_train + gap .. cut_k_val]   (gap = 5 bars buffer)

Default K=3, expanding window:
    fold 1   train ends   ~ 33% of timeline
    fold 2   train ends   ~ 60%
    fold 3   train ends   ~ 80%
    val takes the next ~ (next-30%) slice

Each fold:
    1. Build dataset NPZ slice
    2. Train (10 epochs, smaller for speed)
    3. Backtest on val slice with realistic spread
    4. Save metrics

Usage:
    PYTHONUNBUFFERED=1 python3 walkforward.py --m15 ../real_data/xauusd_m15.csv

Each run writes:
    wf_runs/<run-tag>/walkforward_report.json   (default unless --out is set)
    wf_runs/<run-tag>/run_config.json           (full args + seed for audit)
Use --seed 42 (default) for repeatable runs on CPU; --seed -1 to disable seeding.

Legacy one-line report in cwd:

    python3 walkforward.py --out walkforward_summary.json --run-tag mytag
"""

from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physics_features import build_feature_frame, FEATURE_COLUMNS
from build_dataset import make_labels, make_labels_trade
from physics_specialist import PhysicsSpecialist, SpecialistConfig, SpecialistLoss, LossConfig
from backtest_specialist import (
    TradeRules, run_backtest, compute_features_and_atr, load_model
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_rng_seed(seed: int) -> None:
    """Torch / NumPy / Python random — same `--seed` → same WF run on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_device(prefer: str = "auto") -> torch.device:
    pref = str(prefer).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[wf] warning: requested cuda but CUDA is unavailable, falling back.")
    if pref == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[wf] warning: requested mps but MPS is unavailable, falling back.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_fold_dataset(
    df: pd.DataFrame, train_start_idx: int, train_end_idx: int,
    val_start_idx: int, val_end_idx: int,
    window: int, horizon: int, threshold_pctl: float,
    label_mode: str = "trade",
    sl_atr_mult: float = 1.5,
    slippage_atr_frac: float = 0.025,
    spread_mult: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns: X_all, y_cls, y_ret, train_mask, val_mask, threshold
    Indexed over the global `df` rows (after feature warm-up trimming done
    by the caller).
    """
    feats_df = build_feature_frame(df)
    df_full = pd.concat([df.reset_index(drop=True),
                         feats_df[FEATURE_COLUMNS]], axis=1)
    df_full = df_full.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    high  = df_full["high"].values.astype(float)
    low   = df_full["low"].values.astype(float)
    close = df_full["close"].values.astype(float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low, np.abs(high - prev_close), np.abs(low - prev_close)
    ])
    atr = pd.Series(tr).rolling(48, min_periods=12).mean().bfill().values
    df_full["atr"] = atr

    spread_col = df_full["spread_mean"].astype(float).values \
        if "spread_mean" in df_full.columns \
        else np.full(len(close), 0.40, dtype=np.float64)

    if label_mode == "mid":
        y_ret, y_cls, thr = make_labels(close, atr, horizon, threshold_pctl)
    else:
        y_ret, y_cls, thr = make_labels_trade(
            close, atr, spread_col,
            high=high, low=low, horizon=horizon,
            threshold_pctl=threshold_pctl,
            sl_atr_mult=sl_atr_mult,
            slippage_atr_frac=slippage_atr_frac,
            spread_mult=spread_mult,
        )

    fmat = np.clip(df_full[FEATURE_COLUMNS].astype(np.float32).values, -8, 8)
    return df_full, fmat, y_cls, y_ret, atr, thr


def make_indices(n: int, window: int, horizon: int,
                 train_start: int, train_end: int,
                 val_start: int, val_end: int) -> tuple[np.ndarray, np.ndarray]:
    """All sample indices t such that the WHOLE window [t-W+1, t] and the
    label horizon t+H both lie inside the requested slice."""
    first_t = window - 1
    last_t  = n - horizon - 1
    all_t = np.arange(first_t, last_t + 1)

    tr = all_t[(all_t >= max(first_t, train_start + window - 1)) &
               (all_t <= min(last_t, train_end))]
    va = all_t[(all_t >= max(first_t, val_start + window - 1)) &
               (all_t <= min(last_t, val_end))]
    return tr, va


def train_model_quick(
    fmat: np.ndarray, y_cls: np.ndarray, y_ret: np.ndarray,
    train_t: np.ndarray, val_t: np.ndarray,
    window: int, n_features: int,
    epochs: int = 10, batch_size: int = 256, lr: float = 8e-4,
    device: torch.device = torch.device("cpu"),
    log=lambda s: print(s),
    val_bt_df: pd.DataFrame | None = None,
    bt_rules: TradeRules | None = None,
    label_horizon: int = 16,
    bt_every: int = 4,
    min_bt_trades: int = 30,
    reg_loss: str = "smooth_l1",
    huber_beta: float = 0.5,
    stress_spread_bt: bool = True,
    wf_seed: int = 42,
    fold_idx: int = 1,
    asymmetric_gamma: float = 1.25,
    use_asymmetric_loss: bool = False,
) -> tuple[PhysicsSpecialist, dict]:
    cfg = SpecialistConfig(n_features=n_features, window=window)
    model = PhysicsSpecialist(cfg).to(device)

    loss_fn = SpecialistLoss(
        LossConfig(
            cls_weight=1.0,
            reg_weight=0.10,
            use_class_weights=True,
            reg_loss=reg_loss,
            huber_beta=huber_beta,
            use_asymmetric_loss=use_asymmetric_loss,
            asymmetric_gamma=asymmetric_gamma,
        ),
        class_weights=None,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # build sliding windows once (memory-light: lazy slicing)
    from numpy.lib.stride_tricks import sliding_window_view
    swv = sliding_window_view(fmat, window_shape=window, axis=0).transpose(0, 2, 1)
    # swv[i] corresponds to bar t = i + W - 1

    def to_swv_idx(t_arr): return t_arr - (window - 1)

    train_swv_idx = to_swv_idx(train_t)
    val_swv_idx   = to_swv_idx(val_t)

    train_clip_ret = np.clip(y_ret[train_t], -6.0, 6.0).astype(np.float32)
    train_y_cls    = y_cls[train_t].astype(np.int64)
    val_clip_ret   = np.clip(y_ret[val_t],   -6.0, 6.0).astype(np.float32)
    val_y_cls      = y_cls[val_t].astype(np.int64)

    best_val_loss = float("inf")
    best_state = None
    checkpoint_dd_cap = 0.08  # prefer validation BT max DD strictly below this
    best_ckpt_strict_score = float("-inf")
    best_ckpt_strict_state = None
    best_ckpt_strict_metrics: dict | None = None
    best_ckpt_any_score = float("-inf")
    best_ckpt_any_state = None
    best_ckpt_any_metrics: dict | None = None

    n_train = len(train_t)
    n_steps = (n_train // batch_size) * epochs
    step = 0
    for ep in range(1, epochs + 1):
        # shuffle
        perm = np.random.permutation(n_train)
        train_loss_sum, n_seen = 0.0, 0
        model.train()
        t0 = time.time()
        for i in range(0, n_train - batch_size, batch_size):
            sel = perm[i:i + batch_size]
            xb = torch.from_numpy(swv[train_swv_idx[sel]].copy().astype(np.float32)).to(device)
            cb = torch.from_numpy(train_y_cls[sel]).to(device)
            rb = torch.from_numpy(train_clip_ret[sel]).to(device)
            # cosine LR with warmup
            if step < 400:
                cur_lr = lr * (step + 1) / 400
            else:
                p = (step - 400) / max(1, n_steps - 400)
                cur_lr = 0.5 * lr * (1.0 + np.cos(np.pi * min(1.0, p)))
            for g in optim.param_groups: g["lr"] = float(cur_lr)
            optim.zero_grad()
            out = model(xb)
            losses = loss_fn(out, cb, rb)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss_sum += float(losses["loss"].detach()) * xb.size(0)
            n_seen += xb.size(0)
            step += 1
        train_loss = train_loss_sum / max(1, n_seen)

        # eval on val
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(val_t), batch_size):
                xb = torch.from_numpy(swv[val_swv_idx[i:i+batch_size]].copy()
                                       .astype(np.float32)).to(device)
                cb = torch.from_numpy(val_y_cls[i:i+batch_size]).to(device)
                rb = torch.from_numpy(val_clip_ret[i:i+batch_size]).to(device)
                out = model(xb)
                losses = loss_fn(out, cb, rb)
                val_loss_sum += float(losses["loss"]) * xb.size(0)
                val_n += xb.size(0)
                all_logits.append(out["logits"].cpu().numpy())
        val_loss = val_loss_sum / max(1, val_n)
        logits = np.concatenate(all_logits, axis=0)
        preds = logits.argmax(axis=1)
        acc = float((preds == val_y_cls).mean())
        line = (
            f"   ep {ep:>2d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"acc={acc:.3f}  ({time.time()-t0:.1f}s)"
        )

        bt_part = ""
        if (
            val_bt_df is not None
            and bt_rules is not None
            and len(val_bt_df) > window + label_horizon + 2
            and (ep % max(1, bt_every) == 0 or ep == epochs)
        ):
            model.eval()
            rng_ep = int(ep) * 1_000_003 + int(fold_idx) * 97_631
            base = (wf_seed % 2_147_483_647) if wf_seed >= 0 else 0
            stress_rng = np.random.default_rng((base + rng_ep) % (2**32))
            bt_result = run_backtest(
                val_bt_df, model, cfg, bt_rules, device, verbose=False,
                stress_spread_jitter=stress_spread_bt,
                stress_rng=stress_rng,
            )
            s = bt_result["summary"]
            ntr = int(s["n_trades"])
            if ntr >= min_bt_trades:
                pf_bt = float(s["profit_factor"])
                if pf_bt != pf_bt or np.isnan(pf_bt):
                    pf_bt = 1.0
                if np.isinf(pf_bt):
                    pf_bt = 9999.0
                avg_r_bt = float(s["avg_R"])
                max_dd_bt = float(s["max_dd_pct"])
                score_bt = 0.7 * pf_bt + 0.3 * avg_r_bt

                ck_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                sanitized = {}
                for k, v in s.items():
                    if isinstance(v, (bool, np.bool_)):
                        sanitized[k] = bool(v)
                    elif isinstance(v, (float, np.floating)):
                        sanitized[k] = float(v)
                    elif isinstance(v, (int, np.integer)):
                        sanitized[k] = int(v)
                    else:
                        sanitized[k] = v
                sanitized["pick_epoch"] = ep
                sanitized["checkpoint_score"] = float(score_bt)
                sanitized["checkpoint_prefers_dd_lt_8pct"] = bool(max_dd_bt < checkpoint_dd_cap)

                if score_bt > best_ckpt_any_score:
                    best_ckpt_any_score = score_bt
                    best_ckpt_any_state = ck_state
                    best_ckpt_any_metrics = dict(sanitized)

                if max_dd_bt < checkpoint_dd_cap and score_bt > best_ckpt_strict_score:
                    best_ckpt_strict_score = score_bt
                    best_ckpt_strict_state = ck_state
                    best_ckpt_strict_metrics = dict(sanitized)

                pref = "*" if max_dd_bt < checkpoint_dd_cap else ""
                bt_part = (
                    f"  | BT n={ntr} PF={float(s['profit_factor']):.3f} "
                    f"avgR={float(s['avg_R']):+.4f} DD={max_dd_bt*100:.2f}% sc={score_bt:.4f}{pref}"
                )
            else:
                bt_part = f"  | BT n={ntr} (need>={min_bt_trades}; skip ckpt)"
        log(line + bt_part)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Prefer validation checkpoint that passes DD<8%; else best composite score any DD.
    if best_ckpt_strict_state is not None:
        model.load_state_dict(best_ckpt_strict_state)
        chosen = "val_backtest(0.7*PF+0.3*avg_R,prefer_maxDD<8%)"
        train_info_bt = dict(best_ckpt_strict_metrics or {})
    elif best_ckpt_any_state is not None:
        model.load_state_dict(best_ckpt_any_state)
        chosen = "val_backtest(0.7*PF+0.3*avg_R,fallback_DD>=8%)"
        train_info_bt = dict(best_ckpt_any_metrics or {})
    else:
        if best_state is not None:
            model.load_state_dict(best_state)
        chosen = "val_loss"
        train_info_bt = {}

    return model, {
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "n_train": n_train,
        "n_val": len(val_t),
        "checkpoint_criterion": chosen,
        **({"val_bt_pick": train_info_bt} if train_info_bt else {}),
    }


# ---------------------------------------------------------------------------
# Main walkforward
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m15", default="../real_data/xauusd_m15.csv")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--window",  type=int, default=64)
    ap.add_argument("--horizon", type=int, default=16,
                    help="label horizon (bars); should align loosely with swing length")
    ap.add_argument("--threshold-pctl", type=float, default=0.35,
                    help="v7.2: slightly looser label edge quantile (was 0.40)")
    ap.add_argument("--epochs", type=int, default=25,
                    help="v7.2+: more epochs for Transformer (F=len(FEATURE_COLUMNS))")
    ap.add_argument("--label-mode", choices=("trade", "mid"), default="trade")
    ap.add_argument("--bt-every", type=int, default=4,
                    help="run validation-slice backtest every N epochs for checkpoint pick")
    ap.add_argument("--min-bt-trades", type=int, default=30,
                    help="v7.2: min val-BT trades for checkpoint score (was 50 in v7.1)")
    ap.add_argument("--reg-loss", choices=("smooth_l1", "nll"), default="smooth_l1")
    ap.add_argument("--huber-beta", type=float, default=0.5)
    ap.add_argument("--slippage-atr", type=float, default=0.025,
                    help="label path slippage (match TradeRules)")
    ap.add_argument("--spread-mult", type=float, default=1.0, help="label spread multiplier")
    ap.add_argument("--spread-scale", type=float, default=1.0,
                    help="multiply spread_mean/spread_max by this factor before labels+BT (e.g. 0.01 for points→quote)")
    # Backtest filters (tighter defaults than v1 — reduce low-edge churn)
    ap.add_argument("--dir-prob", type=float, default=0.35,
                    help="v7.2: P(winning direction) ≥ … (relaxed vs 0.38)")
    ap.add_argument("--dir-ratio", type=float, default=1.18,
                    help="P(win)/P(lose) ≥ …")
    ap.add_argument("--conf", type=float, default=0.38,
                    help="max-class prob ≥ … (0 = off)")
    ap.add_argument("--rr-min", type=float, default=0.0,
                    help="min predicted |μ| in ATR vs direction (0=off; enable after model calibrates)")
    ap.add_argument("--max-noisy", type=float, default=0.45,
                    help="v7.2: skip entry if P(noisy) ≥ … (relaxed vs 0.42)")
    ap.add_argument("--sl-atr-mult", type=float, default=1.5,
                    help="initial SL = this × ATR (1R = full risk budget)")
    ap.add_argument(
        "--profit-mode",
        choices=("fixed", "hybrid"),
        default="hybrid",
        help="val backtest exit: hybrid = EA RL_Agent_V9 InpProfitMode=Hybrid (atr_z + probs); default hybrid",
    )
    # Execution-time filters used by backtest_specialist.TradeRules
    ap.add_argument("--session-start-hour", type=int, default=7,
                    help="UTC hour to start allowing entries")
    ap.add_argument("--session-end-hour", type=int, default=21,
                    help="UTC hour to stop allowing entries (exclusive)")
    ap.add_argument("--disable-monday-early-skip", action="store_true",
                    help="do not block Monday early entries")
    ap.add_argument("--disable-friday-late-skip", action="store_true",
                    help="do not block Friday late entries")
    ap.add_argument("--spread-max-quote", type=float, default=0.80,
                    help="maximum spread quote allowed for entry")
    ap.add_argument("--spread-atr-max-ratio", type=float, default=0.40,
                    help="maximum spread/ATR ratio allowed for entry")
    ap.add_argument("--atr-z-min", type=float, default=0.40,
                    help="minimum atr_z allowed for entry")
    ap.add_argument("--atr-z-max", type=float, default=3.00,
                    help="maximum atr_z allowed for entry")
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility; use -1 to leave Python/NumPy/PyTorch unseeded.",
    )
    ap.add_argument(
        "--run-tag", default=None,
        help="Name under wf_runs/<run-tag>/ ; default: UTC timestamp YYMMDD_HHMMSS.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Where to save JSON summary. Default: wf_runs/<run-tag>/walkforward_report.json",
    )
    ap.add_argument(
        "--no-stress-spread-bt",
        action="store_true",
        help="disable stochastic 1.5–2× spread jitter during val checkpoint backtests",
    )
    ap.add_argument("--asymmetric-gamma", type=float, default=1.25,
                    help="v7.2: penalty on directional FP (CE + mis-signed μ)")
    ap.add_argument("--asymmetric-loss", action="store_true",
                    help="Enable asymmetric loss in WF inner training (default: off)")
    ap.add_argument("--no-asymmetric-loss", action="store_true",
                    help="Force asymmetric loss off")
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto",
                    help="training device selection (auto prefers CUDA, then MPS, then CPU)")
    args = ap.parse_args()

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("wf_runs") / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.out) if args.out is not None else (out_dir / "walkforward_report.json")

    if args.seed >= 0:
        set_rng_seed(args.seed)

    device = best_device(args.device)

    manifest = {"run_tag": run_tag, "seed": args.seed, "argv": vars(args).copy()}
    (out_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"[wf] run_tag={run_tag}  report → {report_path.resolve()}")
    if args.seed >= 0:
        print(f"[wf] seed={args.seed} (deterministic-ish on CPU)")
    print(f"[wf] device={device}")
    print(f"[wf] reading {args.m15}")
    df = pd.read_csv(args.m15)
    if abs(args.spread_scale - 1.0) > 1e-12:
        for c in ("spread_mean", "spread_max"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * args.spread_scale
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc").reset_index(drop=True)
    print(f"[wf] M15 bars total: {len(df):,}  range: {df.time_utc.iloc[0]} → {df.time_utc.iloc[-1]}")

    # Build features+labels once on full data
    print("[wf] computing features + labels (once)...")
    df_full, fmat, y_cls, y_ret, atr, thr = build_fold_dataset(
        df, 0, 0, 0, 0, args.window, args.horizon, args.threshold_pctl,
        label_mode=args.label_mode,
        sl_atr_mult=args.sl_atr_mult,
        slippage_atr_frac=args.slippage_atr,
        spread_mult=args.spread_mult,
    )
    n_full = len(df_full)
    print(f"[wf] usable bars after warmup: {n_full:,}  ({args.label_mode} thr={thr:.4f})")

    # Define folds — expanding window
    K = args.folds
    fold_specs = []
    for k in range(K):
        # train end fraction grows: 0.40, 0.55, 0.70, ... + val window
        train_end_frac = 0.40 + 0.20 * k     # 0.40, 0.60, 0.80
        val_end_frac   = train_end_frac + 0.18
        if val_end_frac > 0.98:
            val_end_frac = 0.98
        fold_specs.append((0,
                           int(n_full * train_end_frac),
                           int(n_full * (train_end_frac + 0.005)),  # 0.5% gap buffer
                           int(n_full * val_end_frac)))

    rules = TradeRules(
        dir_prob_min=args.dir_prob,
        dir_ratio_min=args.dir_ratio,
        conf_threshold=args.conf,
        rr_min=args.rr_min,
        max_noisy_prob=args.max_noisy,
        sl_atr_mult=args.sl_atr_mult,
        profit_mode=args.profit_mode,
        session_start_hour=args.session_start_hour,
        session_end_hour=args.session_end_hour,
        skip_monday_early=not args.disable_monday_early_skip,
        skip_friday_late=not args.disable_friday_late_skip,
        spread_max_quote=args.spread_max_quote,
        spread_atr_max_ratio=args.spread_atr_max_ratio,
        atr_z_min=args.atr_z_min,
        atr_z_max=args.atr_z_max,
    )
    fold_results = []
    log_f = (out_dir / "walkforward.log").open("w")
    def log(s):
        print(s); log_f.write(s + "\n"); log_f.flush()

    log(f"[wf] run_tag={run_tag}  seed={args.seed}")
    log(f"[wf] label={args.label_mode} H={args.horizon}  reg_loss={args.reg_loss}  "
        f"val_BT every {args.bt_every} ep (min trades {args.min_bt_trades})")
    log(f"[wf] bt_rules: p_noisy<{args.max_noisy:.2f}  dir_prob>={args.dir_prob}  "
        f"dir_ratio>={args.dir_ratio}  conf>={args.conf}  rr_min>={args.rr_min}  "
        f"SL={args.sl_atr_mult}×ATR  profit_mode={args.profit_mode}  epochs={args.epochs}")
    log(f"[wf] data: spread_scale={args.spread_scale}")
    log(f"[wf] bt_filters: session={args.session_start_hour:02d}-{args.session_end_hour:02d} UTC  "
        f"skip_mon_early={not args.disable_monday_early_skip}  "
        f"skip_fri_late={not args.disable_friday_late_skip}  "
        f"spread_max={args.spread_max_quote:.4f}  spread/atr_max={args.spread_atr_max_ratio:.3f}  "
        f"atr_z=[{args.atr_z_min:.2f},{args.atr_z_max:.2f}]")
    stress_on = not args.no_stress_spread_bt
    asym_on = args.asymmetric_loss and not args.no_asymmetric_loss
    log(f"[wf] v7: stress_spread_checkpoint_BT={stress_on}  asymmetric_loss="
        f"{asym_on}  gamma={args.asymmetric_gamma}")

    for k, (ts, te, vs, ve) in enumerate(fold_specs, start=1):
        log(f"\n──────── FOLD {k} ────────")
        train_dt = df_full.time_utc.iloc[ts:te]
        val_dt   = df_full.time_utc.iloc[vs:ve]
        log(f"   train: {train_dt.iloc[0]}  →  {train_dt.iloc[-1]}    bars={te-ts:,}")
        log(f"   val  : {val_dt.iloc[0]}  →  {val_dt.iloc[-1]}    bars={ve-vs:,}")

        train_t, val_t = make_indices(n_full, args.window, args.horizon,
                                      ts, te, vs, ve)
        log(f"   samples train={len(train_t):,}  val={len(val_t):,}")

        val_bt_prep = df_full.iloc[vs:ve].reset_index(drop=True)

        model, train_info = train_model_quick(
            fmat, y_cls, y_ret, train_t, val_t,
            window=args.window, n_features=fmat.shape[1],
            epochs=args.epochs, device=device, log=log,
            val_bt_df=val_bt_prep,
            bt_rules=rules,
            label_horizon=args.horizon,
            bt_every=args.bt_every,
            min_bt_trades=args.min_bt_trades,
            reg_loss=args.reg_loss,
            huber_beta=args.huber_beta,
            stress_spread_bt=stress_on,
            wf_seed=args.seed,
            fold_idx=k,
            asymmetric_gamma=args.asymmetric_gamma,
            use_asymmetric_loss=args.asymmetric_loss and not args.no_asymmetric_loss,
        )
        log(f"   train pick: {train_info.get('checkpoint_criterion', '?')}  "
            f"{train_info.get('val_bt_pick', {})}")

        # save weights (full Transformer config for reproducible reload)
        ck = out_dir / f"fold{k}.pt"
        torch.save({"model_state": model.state_dict(),
                    "config":      model.cfg.to_dict()}, ck)

        # Backtest on val slice (no spread jitter — forensic PF on nominal spread path)
        log("   running backtest on val slice (with realistic spread)...")
        val_df = val_bt_prep

        cfg_model = model.cfg
        bt_result = run_backtest(val_df, model, cfg_model, rules, device,
                                  initial_equity=10_000.0, verbose=False)
        s = bt_result["summary"]
        log(f"   trades={s['n_trades']}  wr={s['win_rate']*100:.1f}%  "
            f"PF={s['profit_factor']:.3f}  PnL={s['pnl_total_pct']*100:+.2f}%  "
            f"DD={s['max_dd_pct']*100:.2f}%")

        fold_results.append({
            "fold": k,
            "train_range": [str(train_dt.iloc[0]), str(train_dt.iloc[-1])],
            "val_range":   [str(val_dt.iloc[0]),   str(val_dt.iloc[-1])],
            "n_train": int(len(train_t)),
            "n_val":   int(len(val_t)),
            "train_meta": train_info,
            "summary": s,
            "checkpoint": str(ck),
        })

        # Free memory
        del model
        import gc; gc.collect()

    # ---- Aggregate ----
    summary_keys = ["n_trades", "win_rate", "avg_R", "profit_factor",
                    "pnl_total_pct", "max_dd_pct"]
    log("\n══════ WALK-FORWARD AGGREGATE ══════")
    log(f"{'metric':<18} " + "  ".join([f"fold{k+1:>3d}" for k in range(K)]) + "    mean      median")
    for key in summary_keys:
        vals = [fold_results[k]["summary"][key] for k in range(K)]
        log(f"{key:<18} " + "  ".join([f"{v:>8.3f}" for v in vals])
            + f"    {np.mean(vals):>8.3f}    {np.median(vals):>8.3f}")

    # Aggregate verdict
    pfs = [fold_results[k]["summary"]["profit_factor"] for k in range(K)]
    pnl = [fold_results[k]["summary"]["pnl_total_pct"] for k in range(K)]
    dds = [fold_results[k]["summary"]["max_dd_pct"] for k in range(K)]
    profitable_folds = sum(p > 1.0 for p in pfs)
    log(f"\nProfitable folds: {profitable_folds}/{K}  (PF > 1.0)")
    log(f"Mean PF       : {np.mean(pfs):.3f}")
    log(f"Worst-case DD : {max(dds)*100:.2f}%")
    log(f"Worst-case PF : {min(pfs):.3f}")

    final = {
        "run": {
            "tag": run_tag,
            "seed": int(args.seed),
            "out_dir": str(out_dir.resolve()),
            "report": str(report_path.resolve()),
        },
        "fold_count": K,
        "folds": fold_results,
        "agg": {
            "mean_pf":  float(np.mean(pfs)),
            "median_pf":float(np.median(pfs)),
            "min_pf":   float(min(pfs)),
            "max_dd":   float(max(dds)),
            "profitable_folds": int(profitable_folds),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(final, indent=2, default=str))
    log(f"\n[wf] saved → {report_path}")
    log(f"[wf] checkpoints → {out_dir}")
    log_f.close()


if __name__ == "__main__":
    main()
