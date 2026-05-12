"""
Physics Feature Pipeline (Stage 4A)
====================================

Compute multi-timeframe physics features from M15 OHLC for the
Specialist model.  All features are causal (no future leakage) and
robust-scaled (inputs to the network end up roughly in [-3, 3]).

Feature groups (per bar):

    Kinematic (M15)
        v1      log return                         r_t = ln(C_t/C_{t-1})
        v_z     v_t standardised by rolling σ_v
        a       acceleration  v_t − v_{t-1}
        a_z     a_t standardised by rolling σ_a
        j       jerk           a_t − a_{t-1}
        E       energy         v_t²
        E_pct   energy percentile in rolling window

    Statistical (M15)
        zclose  (Close − rolling_mean) / rolling_std
        sharpe  rolling_mean(v) / rolling_std(v)
        atr_z   ATR / median(ATR)            (regime-aware vol level)
        bb_pos  Bollinger position           (Close − BB_mid) / (BB_upper − BB_mid)

    MTF context (resample on the fly)
        v1_h1   1-hour velocity   ln(C_t / C_{t-4})
        v1_h4   4-hour velocity   ln(C_t / C_{t-16})
        trend_h1   sign(ema_h1_slope)
        trend_h4   sign(ema_h4_slope)

    Calendar  (cyclical encodings — never leaky)
        hour_sin, hour_cos   from UTC hour
        dow_sin , dow_cos    from UTC day-of-week
        day_of_week          linear weekday in [-1, 1] (Mon −1 … Sun +1); complements sin/cos

    Microstructure
        sprd_z   spread / median(spread)
        sprd_max_z spread_max / median(spread_max)
        spr_atr_ratio spread_mean / ATR(price) — friction vs typical bar range

    Structure (causal range / Donchian-style, same window as sigma ~1d M15)
        struct_rng_pos   2*(Close−LL)/(HH−LL)−1  in [−1,1]  (position inside recent range)
        struct_edge_atr  min((HH−Close)/ATR, (Close−LL)/ATR)  (distance to nearest range edge in ATR)

Total = configurable list in FEATURE_COLUMNS.

Usage:
    from physics_features import build_feature_frame, FEATURE_COLUMNS
    feats = build_feature_frame(pd.read_csv("xauusd_m15.csv"))
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    sigma_window:   int = 96    # 1 day of M15
    energy_window:  int = 96
    z_window:       int = 96
    bb_window:      int = 48
    atr_window:     int = 48
    median_window:  int = 480   # 5 days
    ema_h1_span:    int = 10
    ema_h4_span:    int = 12
    struct_window: int = 96   # ~1 trading day of M15; rolling HH/LL for structure


FEATURE_COLUMNS: list[str] = [
    "v1", "v_z", "a", "a_z", "j", "E", "E_pct",
    "zclose", "sharpe", "atr_z", "bb_pos",
    "v1_h1", "v1_h4", "trend_h1", "trend_h4",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "day_of_week",
    "sprd_z", "sprd_max_z", "spr_atr_ratio",
    "struct_rng_pos", "struct_edge_atr",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0.0, np.nan)).fillna(fill)


def _rolling_z(x: pd.Series, w: int) -> pd.Series:
    mu = x.rolling(w, min_periods=max(8, w // 4)).mean()
    sd = x.rolling(w, min_periods=max(8, w // 4)).std()
    return _safe_div(x - mu, sd)


def _rolling_pctl(x: pd.Series, w: int) -> pd.Series:
    """Percentile rank of x_t within the rolling window (0..1)."""
    return x.rolling(w, min_periods=max(8, w // 4)).apply(
        lambda v: (v[-1] >= v).mean(), raw=True
    )


def _atr(df: pd.DataFrame, w: int) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=max(4, w // 4)).mean()


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_feature_frame(
    df: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    df must contain: time_utc, open, high, low, close,
                     spread_mean (optional), spread_max (optional).
    Returns: DataFrame with ['time_utc'] + FEATURE_COLUMNS, no NaNs at the head
             trimmed but kept (caller handles dropna).
    """
    cfg = cfg or FeatureConfig()
    out = pd.DataFrame()

    # Time
    out["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    out = out.sort_values("time_utc").reset_index(drop=True)

    close = df["close"].astype(float).reset_index(drop=True)
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)

    # ---- Kinematics ----
    v1 = np.log(close).diff()
    a  = v1.diff()
    j  = a.diff()
    E  = v1 * v1

    out["v1"]    = v1
    out["v_z"]   = _rolling_z(v1, cfg.sigma_window)
    out["a"]     = a
    out["a_z"]   = _rolling_z(a, cfg.sigma_window)
    out["j"]     = j
    out["E"]     = E
    out["E_pct"] = _rolling_pctl(E, cfg.energy_window)

    # ---- Statistical ----
    mu_close = close.rolling(cfg.z_window, min_periods=cfg.z_window // 4).mean()
    sd_close = close.rolling(cfg.z_window, min_periods=cfg.z_window // 4).std()
    out["zclose"] = _safe_div(close - mu_close, sd_close)

    mu_v = v1.rolling(cfg.sigma_window, min_periods=cfg.sigma_window // 4).mean()
    sd_v = v1.rolling(cfg.sigma_window, min_periods=cfg.sigma_window // 4).std()
    out["sharpe"] = _safe_div(mu_v, sd_v)

    atr = _atr(df.reset_index(drop=True), cfg.atr_window)
    atr_med = atr.rolling(cfg.median_window, min_periods=cfg.median_window // 4).median()
    out["atr_z"] = _safe_div(atr, atr_med, fill=1.0)

    bb_mid = close.rolling(cfg.bb_window, min_periods=cfg.bb_window // 4).mean()
    bb_std = close.rolling(cfg.bb_window, min_periods=cfg.bb_window // 4).std()
    bb_upper = bb_mid + 2.0 * bb_std
    out["bb_pos"] = _safe_div(close - bb_mid, bb_upper - bb_mid)

    # ---- MTF (M15 resampled to H1, H4 — using rolling because timestamps are aligned) ----
    # M15 bars per H1 = 4, per H4 = 16
    out["v1_h1"] = np.log(close).diff(4)
    out["v1_h4"] = np.log(close).diff(16)

    ema_h1 = close.ewm(span=cfg.ema_h1_span, adjust=False).mean()
    ema_h4 = close.ewm(span=cfg.ema_h4_span * 4, adjust=False).mean()
    out["trend_h1"] = np.sign(ema_h1.diff(4)).fillna(0.0).astype(float)
    out["trend_h4"] = np.sign(ema_h4.diff(16)).fillna(0.0).astype(float)

    # ---- Calendar (cyclical) ----
    t = out["time_utc"].dt
    out["hour_sin"] = np.sin(2.0 * np.pi * t.hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * t.hour / 24.0)
    out["dow_sin"]  = np.sin(2.0 * np.pi * t.dayofweek / 7.0)
    out["dow_cos"]  = np.cos(2.0 * np.pi * t.dayofweek / 7.0)
    # Linear complement to sin/cos (Mon −1 … Sun +1)
    dow = t.dayofweek.astype(np.float64)
    out["day_of_week"] = (2.0 * dow / 6.0) - 1.0

    # ---- Microstructure ----
    if "spread_mean" in df.columns:
        sp = df["spread_mean"].astype(float).reset_index(drop=True)
        sp_med = sp.rolling(cfg.median_window, min_periods=20).median()
        out["sprd_z"] = _safe_div(sp, sp_med, fill=1.0)
    else:
        out["sprd_z"] = 0.0

    if "spread_max" in df.columns:
        spm = df["spread_max"].astype(float).reset_index(drop=True)
        spm_med = spm.rolling(cfg.median_window, min_periods=20).median()
        out["sprd_max_z"] = _safe_div(spm, spm_med, fill=1.0)
    else:
        out["sprd_max_z"] = 0.0

    # Spread / ATR(price): high → costly relative to volatility (helps filter friction-heavy bars)
    if "spread_mean" in df.columns:
        sp_here = df["spread_mean"].astype(float).reset_index(drop=True)
        out["spr_atr_ratio"] = _safe_div(sp_here, atr, fill=0.0)
        out["spr_atr_ratio"] = out["spr_atr_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["spr_atr_ratio"] = 0.0

    # ---- Structure (causal Donchian / range position) ----
    Ws = cfg.struct_window
    min_s = max(8, Ws // 4)
    hh = high.rolling(Ws, min_periods=min_s).max()
    ll = low.rolling(Ws, min_periods=min_s).min()
    rng_br = (hh - ll).replace(0.0, np.nan)
    out["struct_rng_pos"] = (2.0 * _safe_div(close - ll, rng_br) - 1.0).fillna(0.0)
    d_top = _safe_div(hh - close, atr.replace(0.0, np.nan))
    d_bot = _safe_div(close - ll, atr.replace(0.0, np.nan))
    out["struct_edge_atr"] = np.minimum(d_top, d_bot).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    # Final hygiene: clip extremes to keep network inputs bounded
    for col in FEATURE_COLUMNS:
        if col in (
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "day_of_week",
            "trend_h1", "trend_h4",
        ):
            continue
        out[col] = out[col].clip(-10.0, 10.0)

    return out[["time_utc"] + FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="../real_data/xauusd_m15.csv")
    ap.add_argument("--output", default=None)
    ap.add_argument("--head",   type=int, default=5)
    args = ap.parse_args()

    src = Path(args.input).resolve()
    print(f"[features] reading {src}")
    df = pd.read_csv(src)
    print(f"[features] M15 bars in: {len(df):,}")

    feats = build_feature_frame(df)
    pre = len(feats)
    feats = feats.dropna().reset_index(drop=True)
    print(f"[features] dropped {pre - len(feats):,} bars at warm-up "
          f"→ usable {len(feats):,}")

    print("\n[features] preview:")
    cols_show = ["time_utc", "v1", "v_z", "a_z", "E_pct",
                 "zclose", "sharpe", "atr_z", "bb_pos",
                 "v1_h1", "trend_h1", "hour_sin", "day_of_week", "sprd_z", "spr_atr_ratio"]
    print(feats[cols_show].head(args.head).to_string(index=False))

    print("\n[features] summary stats:")
    print(feats[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(4))

    if args.output:
        out = Path(args.output).resolve()
        feats.to_csv(out, index=False)
        print(f"\n[features] saved → {out}")


if __name__ == "__main__":
    _main()
