"""
Build supervised dataset for the Physics Specialist (Stage 4B)
==============================================================

Reads M15 OHLC, computes physics features, and produces (X, y_class, y_ret)
arrays suitable for sequence-modelling.

Sample at index t:
    X[t]      = features[t-W+1 : t+1, :]              shape (W, F)
    y_cls[t]  = 0 (short edge) | 1 (noisy) | 2 (long edge)   (trade-aligned mode)
               or 0 (down) | 1 (noisy) | 2 (up)               (legacy mid-return mode)
    y_ret[t]  = R_long − R_short  in **R multiples** (1 R = sl_atr_mult × ATR at entry),
                or legacy: log-return / ATR  (mid-only)

Train/val split is **time-based** (last `val_frac` is OOS).

Output NPZ:
    X         float32  (N, W, F)
    y_ret     float32  (N,)            scaled by ATR (return_units = y_ret/atr_at_t)
    y_cls     int64    (N,)
    weights   float32  (N,)            inverse-frequency class weights for sampler use
    times     int64    (N,)            unix sec — for time-aware splits
    feature_names   ndarray[str]
    train_idx       int64   (Ntrain,)  indices into N belonging to train
    val_idx         int64   (Nval,)
    horizon         int                H
    window          int                W
    threshold_pctl  float              percentile used to decide up/down

Usage:
    python3 build_dataset.py \
        --input ../real_data/xauusd_m15.csv \
        --out   dataset_real_5y.npz \
        --window 64 --horizon 3 --val-frac 0.20 --threshold-pctl 0.52
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from physics_features import build_feature_frame, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Label generator
# ---------------------------------------------------------------------------

def make_labels(
    close: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    threshold_pctl: float = 0.52,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build (y_ret, y_cls, threshold_used).

    y_ret   = log return over horizon, scaled by atr at entry  (return-in-ATR units)
    y_cls   = 0 (down) / 1 (noisy) / 2 (up)
    threshold = the |y_ret| boundary used (in ATR units), chosen so the
                bottom and top `threshold_pctl` slices map to down/up.
    """
    n = len(close)
    log_close = np.log(close)
    fwd = np.empty(n, dtype=np.float64)
    fwd[:n - horizon] = log_close[horizon:] - log_close[:n - horizon]
    fwd[n - horizon:] = np.nan

    # Scale by ATR at entry (atr aligned to entry time); atr in price terms,
    # so use atr / close_t to convert to log-return units.
    safe_atr_log = atr / np.maximum(close, 1e-9)
    safe_atr_log = np.where(safe_atr_log > 0, safe_atr_log, np.nanmedian(safe_atr_log))
    y_ret = fwd / np.maximum(safe_atr_log, 1e-9)

    # Mask invalid tail
    valid = ~np.isnan(y_ret)
    y_ret[~valid] = 0.0

    # Threshold from quantiles of |y_ret| over the valid set
    abs_v = np.abs(y_ret[valid])
    threshold = float(np.quantile(abs_v, 1.0 - threshold_pctl))

    y_cls = np.full(n, 1, dtype=np.int64)  # default = noisy
    y_cls[(y_ret >  threshold)] = 2        # up
    y_cls[(y_ret < -threshold)] = 0        # down
    y_cls[~valid] = -1                     # invalid

    return y_ret.astype(np.float32), y_cls, threshold


def _fill_entry(direction: float, mid: float, spread: float, atr: float,
               slip_frac: float, spread_mult: float) -> float:
    half = 0.5 * spread * spread_mult
    slip = slip_frac * atr
    return mid + direction * (half + slip)


def _fill_entry_vec(direction: float, mid: np.ndarray, spread: np.ndarray, atr: np.ndarray,
                    slip_frac: float, spread_mult: float) -> np.ndarray:
    half = 0.5 * spread * spread_mult
    slip = slip_frac * atr
    return mid + direction * (half + slip)


def _fill_exit(direction: float, mid: float, spread: float, atr: float,
              slip_frac: float, spread_mult: float) -> float:
    half = 0.5 * spread * spread_mult
    slip = slip_frac * atr
    return mid - direction * (half + slip)


def _fill_exit_vec(direction: float, mid: np.ndarray, spread: np.ndarray, atr: np.ndarray,
                   slip_frac: float, spread_mult: float) -> np.ndarray:
    half = 0.5 * spread * spread_mult
    slip = slip_frac * atr
    return mid - direction * (half + slip)


def _r_multiple(direction: float, entry: float, exit_px: float, r_unit: float) -> float:
    if r_unit <= 1e-12:
        return 0.0
    return float((exit_px - entry) * direction / r_unit)


def _r_multiple_vec(direction: float, entry: np.ndarray, exit_px: np.ndarray,
                    r_unit: np.ndarray) -> np.ndarray:
    ru = np.maximum(r_unit, 1e-12)
    return (exit_px - entry) * direction / ru


def _spread_price_per_bar(spread: np.ndarray, n: int) -> np.ndarray:
    """
    Per-bar spread in price units (same as CSV spread_mean / run_backtest spread_arr).
    Invalid or non-positive values are filled with the median of positive spreads,
    then 0.40 only if no positive samples exist (matches backtest fallback scale).
    """
    sp = np.asarray(spread, dtype=np.float64).reshape(-1)
    if len(sp) != n:
        raise ValueError(f"spread length {len(sp)} != n={n}")
    pos = sp[np.isfinite(sp) & (sp > 0)]
    med = float(np.median(pos)) if pos.size > 0 else 0.40
    out = np.where(np.isfinite(sp) & (sp > 0), sp, med)
    if not np.any(np.isfinite(out) & (out > 0)):
        out = np.full(n, 0.40, dtype=np.float64)
    return out


def simulate_long_short_R_path(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    spread: np.ndarray,
    horizon: int,
    sl_atr_mult: float = 1.5,
    slippage_atr_frac: float = 0.025,
    spread_mult: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized path simulation for t in [0, n-H-1]. Tail indices n-H..n-1 are nan.

    Friction matches backtest_specialist.run_backtest + TradeRules:
      - R_unit_price at entry t: ATR[t] * max(sl_atr_mult, 0.1)
      - Entry at t: half-spread + slippage_atr_frac * ATR[t] (adverse side via _fill_entry_vec).
      - Horizon exit at t+H: half-spread + slippage_atr_frac * ATR[t+H] (_fill_exit_vec).
      - SL exit: stop price ± slippage_atr_frac * ATR[stop_bar] (same slip model as backtest).

    spread must be length n (typically df['spread_mean']); see _spread_price_per_bar.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(close)
    H = int(horizon)
    r_long = np.full(n, np.nan, dtype=np.float64)
    r_short = np.full(n, np.nan, dtype=np.float64)
    if n <= H:
        return r_long.astype(np.float32), r_short.astype(np.float32)

    sp = _spread_price_per_bar(spread, n)
    atr64 = atr.astype(np.float64)
    close64 = close.astype(np.float64)
    high64 = high.astype(np.float64)
    low64 = low.astype(np.float64)
    # Identical to run_backtest: R_unit_price = atr_t * max(rules.sl_atr_mult, 0.1)
    r_unit = atr64 * max(sl_atr_mult, 0.1)

    m = n - H  # starts t = 0 .. m-1
    lows_f = sliding_window_view(low64, H)[1 : n - H + 1]
    highs_f = sliding_window_view(high64, H)[1 : n - H + 1]

    entry_L = _fill_entry_vec(1.0, close64[:m], sp[:m], atr64[:m], slippage_atr_frac, spread_mult)
    sl_L = entry_L - r_unit[:m]
    hit_L = lows_f <= sl_L[:, np.newaxis]
    has_L = hit_L.any(axis=1)
    first_j_L = np.argmax(hit_L, axis=1)
    idx_sl_L = np.arange(m, dtype=np.int64) + 1 + first_j_L
    exit_stop_L = sl_L - slippage_atr_frac * atr64[idx_sl_L]

    horizon_row_idx = np.arange(m, dtype=np.int64) + H
    exit_hor_L = _fill_exit_vec(
        1.0, close64[horizon_row_idx], sp[horizon_row_idx],
        atr64[horizon_row_idx], slippage_atr_frac, spread_mult,
    )
    exit_L = np.where(has_L, exit_stop_L, exit_hor_L)
    r_long[:m] = _r_multiple_vec(1.0, entry_L, exit_L, r_unit[:m])

    entry_S = _fill_entry_vec(-1.0, close64[:m], sp[:m], atr64[:m], slippage_atr_frac, spread_mult)
    sl_S = entry_S + r_unit[:m]
    hit_S = highs_f >= sl_S[:, np.newaxis]
    has_S = hit_S.any(axis=1)
    first_j_S = np.argmax(hit_S, axis=1)
    idx_sl_S = np.arange(m, dtype=np.int64) + 1 + first_j_S
    exit_stop_S = sl_S + slippage_atr_frac * atr64[idx_sl_S]
    exit_hor_S = _fill_exit_vec(
        -1.0, close64[horizon_row_idx], sp[horizon_row_idx],
        atr64[horizon_row_idx], slippage_atr_frac, spread_mult,
    )
    exit_S = np.where(has_S, exit_stop_S, exit_hor_S)
    r_short[:m] = _r_multiple_vec(-1.0, entry_S, exit_S, r_unit[:m])

    return r_long.astype(np.float32), r_short.astype(np.float32)


def make_labels_trade(
    close: np.ndarray,
    atr: np.ndarray,
    spread: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    horizon: int,
    threshold_pctl: float = 0.52,
    sl_atr_mult: float = 1.5,
    slippage_atr_frac: float = 0.025,
    spread_mult: float = 1.0,
    ret_clip_abs: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    y_ret = R_long - R_short  (positive → long dominates at H / SL horizon)
    y_cls = 0 short edge | 1 noisy | 2 long edge
    Threshold on |edge| mirrors legacy quantile convention on |y|.
    """
    r_long, r_short = simulate_long_short_R_path(
        high, low, close, atr, spread, horizon,
        sl_atr_mult=sl_atr_mult,
        slippage_atr_frac=slippage_atr_frac,
        spread_mult=spread_mult,
    )
    edge = r_long - r_short
    n = len(close)
    valid = np.isfinite(edge) & (np.arange(n) < n - horizon)
    y_ret = np.nan_to_num(edge, nan=0.0).astype(np.float64)
    y_ret = np.clip(y_ret, -ret_clip_abs, ret_clip_abs).astype(np.float32)

    abs_e = np.abs(edge[valid])
    if len(abs_e) == 0:
        thr = 0.1
    else:
        thr = float(np.quantile(abs_e, 1.0 - threshold_pctl))

    y_cls = np.full(n, 1, dtype=np.int64)
    y_cls[(edge > thr) & valid] = 2
    y_cls[(edge < -thr) & valid] = 0
    y_cls[~valid | ~np.isfinite(edge)] = -1
    return y_ret, y_cls, thr


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    input_csv: Path,
    out_npz: Path,
    window: int,
    horizon: int,
    val_frac: float,
    threshold_pctl: float,
    feature_clip: float = 8.0,
    label_mode: str = "trade",
    sl_atr_mult: float = 1.5,
    slippage_atr_frac: float = 0.025,
    spread_mult: float = 1.0,
):
    print(f"[ds] reading {input_csv}")
    df = pd.read_csv(input_csv)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc").reset_index(drop=True)
    print(f"[ds] M15 bars in: {len(df):,}")

    print("[ds] computing physics features...")
    feats = build_feature_frame(df)

    # Align purely by index since both frames are already sorted on the same time grid.
    assert len(feats) == len(df)
    df = pd.concat([df.drop(columns=[c for c in FEATURE_COLUMNS if c in df.columns]),
                    feats[FEATURE_COLUMNS]], axis=1)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    print(f"[ds] usable bars after warmup: {len(df):,}")

    # ---- ATR (price units) for label scaling ----
    high  = df["high"].astype(float).values
    low   = df["low"].astype(float).values
    close = df["close"].astype(float).values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    atr = pd.Series(tr).rolling(48, min_periods=12).mean().bfill().values

    spread_col = df["spread_mean"].astype(float).values if "spread_mean" in df.columns \
        else np.full(len(close), 0.40, dtype=np.float64)

    # ---- Labels ----
    if label_mode == "mid":
        print(f"[ds] making MID labels (horizon={horizon}, threshold_pctl={threshold_pctl})")
        y_ret, y_cls, thr = make_labels(close, atr, horizon, threshold_pctl)
        print(f"[ds] threshold (ATR-normalized mid-return) = {thr:.3f}")
    else:
        print(f"[ds] making TRADE labels (horizon={horizon}, SL={sl_atr_mult}×ATR, pctl={threshold_pctl})")
        y_ret, y_cls, thr = make_labels_trade(
            close, atr, spread_col,
            high=high, low=low, horizon=horizon,
            threshold_pctl=threshold_pctl,
            sl_atr_mult=sl_atr_mult,
            slippage_atr_frac=slippage_atr_frac,
            spread_mult=spread_mult,
        )
        print(f"[ds] threshold (|R_long−R_short|) = {thr:.4f}")

    # ---- Feature matrix ----
    F = len(FEATURE_COLUMNS)
    fmat = df[FEATURE_COLUMNS].astype(np.float32).values
    fmat = np.nan_to_num(fmat, nan=0.0, posinf=0.0, neginf=0.0)
    fmat = np.clip(fmat, -feature_clip, feature_clip)

    # ---- Build sliding windows ----
    n = len(df)
    last_valid_t = n - horizon - 1  # need t+horizon to exist for label
    first_valid_t = window - 1
    if last_valid_t <= first_valid_t:
        raise RuntimeError("Not enough rows for window+horizon.")

    valid_t = np.arange(first_valid_t, last_valid_t + 1)
    valid_t = valid_t[y_cls[valid_t] >= 0]

    print(f"[ds] valid samples: {len(valid_t):,}  (window={window}, horizon={horizon})")

    # Allocate
    N = len(valid_t)
    X = np.empty((N, window, F), dtype=np.float32)
    print(f"[ds] dataset memory: {(N*window*F*4)/1e6:.1f} MB")

    t_start = time.time()
    # Build via vectorised stride trick (fast)
    from numpy.lib.stride_tricks import sliding_window_view
    swv = sliding_window_view(fmat, window_shape=window, axis=0)  # (n - W + 1, F, W)
    swv = np.ascontiguousarray(swv.transpose(0, 2, 1))            # (n - W + 1, W, F)
    # Index by (t - (window-1)) since swv[i] represents bars [i:i+W]
    rel = valid_t - (window - 1)
    X = swv[rel].astype(np.float32, copy=False)
    print(f"[ds] sliding view took {time.time() - t_start:.2f}s")

    y_r  = y_ret[valid_t].astype(np.float32)
    y_c  = y_cls[valid_t].astype(np.int64)
    times = (
        df["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
                       .astype("datetime64[s]").astype("int64").values
    )[valid_t]

    # ---- Class weights (inverse frequency) ----
    counts = np.bincount(y_c, minlength=3).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / counts, 0.0)
    inv = inv / inv.sum() * 3.0  # normalize so mean weight ≈ 1 across 3 classes
    weights_per_class = inv.astype(np.float32)
    sample_weights = weights_per_class[y_c]

    print(f"[ds] class counts:  down={counts[0]:.0f}  noisy={counts[1]:.0f}  up={counts[2]:.0f}")
    print(f"[ds] class weights: {weights_per_class.round(3).tolist()}")

    # ---- Time-based split ----
    split_idx = int(N * (1.0 - val_frac))
    train_idx = np.arange(split_idx, dtype=np.int64)
    val_idx   = np.arange(split_idx, N, dtype=np.int64)
    print(f"[ds] split: train={len(train_idx):,}   val={len(val_idx):,}")

    train_dt = pd.to_datetime(times[train_idx[0]],   unit="s", utc=True)
    train_end = pd.to_datetime(times[train_idx[-1]], unit="s", utc=True)
    val_dt   = pd.to_datetime(times[val_idx[0]],     unit="s", utc=True)
    val_end  = pd.to_datetime(times[val_idx[-1]],    unit="s", utc=True)
    print(f"[ds] train range: {train_dt}  →  {train_end}")
    print(f"[ds] val   range: {val_dt}  →  {val_end}")

    # ---- Save ----
    print(f"[ds] saving {out_npz}")
    np.savez_compressed(
        out_npz,
        X=X,
        y_ret=y_r,
        y_cls=y_c,
        weights=sample_weights,
        times=times,
        feature_names=np.array(FEATURE_COLUMNS),
        train_idx=train_idx,
        val_idx=val_idx,
        horizon=np.int32(horizon),
        window=np.int32(window),
        threshold_pctl=np.float32(threshold_pctl),
        threshold_atr=np.float32(thr),
        label_mode=np.array(label_mode),
        sl_atr_mult=np.float32(sl_atr_mult),
    )
    sz_mb = out_npz.stat().st_size / 1e6
    print(f"[ds] saved → {out_npz.name}   ({sz_mb:.1f} MB)")

    print("\n[ds] feature names:")
    for i, name in enumerate(FEATURE_COLUMNS):
        print(f"     {i:>2d} {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--out",    default="dataset.npz")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--horizon",type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--threshold-pctl", type=float, default=0.52,
                    help="top|bottom fraction of |edge| considered long/short (trade) or up/down (mid)")
    ap.add_argument("--label-mode", choices=("trade", "mid"), default="trade",
                    help="trade: SL-path R-multiples + spread; mid: legacy ATR-normalized log return")
    ap.add_argument("--sl-atr-mult", type=float, default=1.5,
                    help="label SL distance / 1 R (match backtest)")
    ap.add_argument("--slippage-atr", type=float, default=0.025, help="slippage as fraction of ATR (matches TradeRules default)")
    ap.add_argument("--spread-mult", type=float, default=1.0)
    args = ap.parse_args()

    build(
        Path(args.input),
        Path(args.out),
        window=args.window,
        horizon=args.horizon,
        val_frac=args.val_frac,
        threshold_pctl=args.threshold_pctl,
        label_mode=args.label_mode,
        sl_atr_mult=args.sl_atr_mult,
        slippage_atr_frac=args.slippage_atr,
        spread_mult=args.spread_mult,
    )


if __name__ == "__main__":
    main()
