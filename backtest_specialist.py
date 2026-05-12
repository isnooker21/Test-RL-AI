"""
Backtest the Physics Specialist (Stage 4E)
===========================================

Runs the trained Specialist over M15 OHLC and simulates **single-shot**
trading with the agreed philosophy:

    * Single position at a time (no overlap)
    * Risk per trade  = 0.30% of equity            (deterministic)
    * Daily loss cap  = 2.0% (rolling daily reset, weekday UTC)
    * Weekly loss cap = 5.0% (Mon-Fri rolling)
    * Max DD ceiling  = 10.0%  → AI disabled until user reset
    * Entry filter:
          - max-class probability >= conf_threshold (default 0.55)
          - direction = arg-max class (skip if 'noisy')
          - predicted R:R from regression head >= rr_min (default 1.5)
    * Exit (asymmetric) — defaults aligned with RL_Agent_V9 InpBt* single profile:
          - Hard SL: −1 R (initial distance = sl_atr_mult × ATR)
          - Breakeven / partial / trail thresholds from TradeRules (defaults ~0.45 / 0.70 / 1.10×ATR)
          - **profit_mode** (default **hybrid**): at entry, pick conservative / standard(InpBt*)
            / aggressive from atr_z and probs (same rules as SpecialistV72_BtLive.mqh); use 'fixed' for one profile only
          - confidence-drop close: every M bars re-evaluate; if model now
            predicts opposite class with prob >= conf_drop_threshold OR
            current direction prob falls < conf_floor → close all

Key metrics:
    n_trades, win_rate, avg_R, profit_factor,
    pnl_total_pct, max_dd_pct, sharpe_per_trade,
    daily_circuit_breaker_hits, weekly_circuit_breaker_hits,
    longest_losing_streak,
    R-distribution histogram

Usage:
    python3 backtest_specialist.py \
        --model      runs/<TS>/best.pt \
        --m15        ../real_data/xauusd_m15.csv \
        --start      2025-04-08              # OOS start
        --conf       0.55
        --rr-min     1.5
"""

from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physics_features import build_feature_frame, FEATURE_COLUMNS
from physics_specialist import PhysicsSpecialist, SpecialistConfig


# ---------------------------------------------------------------------------
# Risk + Trade engine
# ---------------------------------------------------------------------------

@dataclass
class TradeRules:
    risk_per_trade: float = 0.003
    daily_loss_cap: float = 0.020
    weekly_loss_cap: float = 0.050
    max_drawdown:   float = 0.100
    dir_prob_min:   float = 0.46
    dir_ratio_min:  float = 1.15
    rr_min:         float = 0.0
    conf_threshold: float = 0.0
    # Skip bars where the “noisy / uncertain” class dominates.
    max_noisy_prob: float = 0.42
    conf_floor:     float = 0.20
    conf_drop_dir:  float = 0.40
    reeval_every_bars: int = 1
    # Initial stop distance (in multiples of entry ATR).
    # 1×ATR = tight (more stop-outs); 1.25–2.0 trades whipsaws for tighter costs.
    sl_atr_mult: float = 1.5

    # Standard / fixed exit leg (RL_Agent_V9 InpBt* defaults)
    breakeven_at_R: float = 0.45
    partial_at_R:   float = 0.70
    partial_share:  float = 0.60
    trail_atr_mult: float = 1.10
    trail_atr_tight_R: float = 1.30
    trail_atr_tight_mult: float = 0.85

    # ---- Hybrid profit (EA InpProfitMode=Hybrid) ----
    profit_mode: str = "hybrid"  # "fixed" | "hybrid" (default hybrid = EA + Python aligned)

    hyb_atr_z_conservative: float = 1.25
    hyb_p_win_aggressive: float = 0.38
    hyb_noise_max_aggressive: float = 0.52  # use >= 0.999 to disable noise gate

    hyb0_breakeven_at_R: float = 0.55
    hyb0_partial_at_R: float = 0.88
    hyb0_partial_share: float = 0.45
    hyb0_trail_atr_mult: float = 1.30
    hyb0_trail_atr_tight_R: float = 1.55
    hyb0_trail_atr_tight_mult: float = 0.90

    hyb2_breakeven_at_R: float = 0.35
    hyb2_partial_at_R: float = 0.58
    hyb2_partial_share: float = 0.65
    hyb2_trail_atr_mult: float = 0.95
    hyb2_trail_atr_tight_R: float = 1.18
    hyb2_trail_atr_tight_mult: float = 0.78

    # ---- Realistic broker friction ----
    spread_mult:        float = 1.00
    spread_widen_jumps: float = 1.50
    slippage_atr_frac:  float = 0.025
    commission_bps:     float = 0.0
    spread_buffer_atr_frac: float = 0.0

    # ---- ENTRY FILTERS (Step 1) ----
    # Session filter — only trade during these UTC hours (London + NY overlap)
    session_start_hour: int = 7            # 07:00 UTC = London open
    session_end_hour:   int = 21           # 21:00 UTC = NY close
    skip_friday_late:   bool = True        # avoid weekend gap
    friday_close_hour:  int = 19           # close any open trade Fri 19:00 UTC
    skip_monday_early:  bool = True
    monday_open_hour:   int = 8            # don't enter Mon before 08:00 UTC

    # Spread filter — skip entries when spread is exceptionally wide
    spread_max_quote:   float = 0.80       # 80¢ — corresponds to ~p95 of XAUUSD
    spread_atr_max_ratio: float = 0.40     # also skip if spread > 40% of ATR

    # Volatility filter — skip extreme regimes (both too dead and too wild)
    atr_z_min:          float = 0.40       # avoid dead market (vol << median)
    atr_z_max:          float = 3.00       # avoid news/crisis bars (vol >> 3x)


@dataclass
class TradeRecord:
    entry_time:  str
    exit_time:   str
    direction:   int            # +1 long / -1 short
    entry_price: float
    exit_price:  float
    sl_price:    float
    risk_R:      float          # how many R captured (after partials, fees)
    pnl_pct:     float          # realized PnL on equity at entry
    bars_held:   int
    confidence:  float          # entry confidence
    exit_reason: str            # "sl", "trail", "reverse", "confidence", "tp_partial+trail"


def clamp_exit_params(
    be: float,
    partial_r: float,
    pshare: float,
    trail_mult: float,
    tight_r: float,
    tight_mult: float,
) -> tuple[float, float, float, float, float, float]:
    """Match SpecialistV72_BtLive.mqh Sv72BtClampProfitParams."""
    be = float(max(0.05, min(be, 5.0)))
    partial_r = float(max(0.05, min(partial_r, 5.0)))
    pshare = float(max(0.05, min(pshare, 0.95)))
    trail_mult = float(max(0.2, min(trail_mult, 5.0)))
    tight_r = float(max(0.05, min(tight_r, 5.0)))
    tight_mult = float(max(0.2, min(tight_mult, 5.0)))
    return be, partial_r, pshare, trail_mult, tight_r, tight_mult


def resolve_exit_params(
    rules: TradeRules,
    atr_z: float,
    p_down: float,
    p_noisy: float,
    p_up: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Per-trade exit thresholds. Fixed mode uses rules.breakeven_at_R … only.
    Hybrid mirrors RL_Agent_V9 InpProfitMode=Hybrid + Sv72BtPickHybridProfile.
    """
    if (rules.profit_mode or "hybrid").lower() != "hybrid":
        return clamp_exit_params(
            rules.breakeven_at_R,
            rules.partial_at_R,
            rules.partial_share,
            rules.trail_atr_mult,
            rules.trail_atr_tight_R,
            rules.trail_atr_tight_mult,
        )

    p_win = max(float(p_up), float(p_down))
    az = float(atr_z)
    if az >= rules.hyb_atr_z_conservative - 1e-12:
        return clamp_exit_params(
            rules.hyb0_breakeven_at_R,
            rules.hyb0_partial_at_R,
            rules.hyb0_partial_share,
            rules.hyb0_trail_atr_mult,
            rules.hyb0_trail_atr_tight_R,
            rules.hyb0_trail_atr_tight_mult,
        )
    noise_off = rules.hyb_noise_max_aggressive >= 0.999
    noise_ok = noise_off or (float(p_noisy) <= rules.hyb_noise_max_aggressive + 1e-12)
    if noise_ok and p_win >= rules.hyb_p_win_aggressive - 1e-12:
        return clamp_exit_params(
            rules.hyb2_breakeven_at_R,
            rules.hyb2_partial_at_R,
            rules.hyb2_partial_share,
            rules.hyb2_trail_atr_mult,
            rules.hyb2_trail_atr_tight_R,
            rules.hyb2_trail_atr_tight_mult,
        )
    return clamp_exit_params(
        rules.breakeven_at_R,
        rules.partial_at_R,
        rules.partial_share,
        rules.trail_atr_mult,
        rules.trail_atr_tight_R,
        rules.trail_atr_tight_mult,
    )


# ---------------------------------------------------------------------------
# Per-bar pipeline
# ---------------------------------------------------------------------------

def load_model(path: Path, device: torch.device) -> tuple[PhysicsSpecialist, SpecialistConfig]:
    chk = torch.load(path, map_location=device, weights_only=False)
    raw = chk.get("config", {})
    valid = {f.name for f in fields(SpecialistConfig)}
    merged = {f.name: getattr(SpecialistConfig(), f.name) for f in fields(SpecialistConfig)}
    for k, v in raw.items():
        if k in valid:
            merged[k] = v
    cfg = SpecialistConfig(**merged)
    model = PhysicsSpecialist(cfg).to(device)
    model.load_state_dict(chk["model_state"])
    model.eval()
    return model, cfg


def compute_features_and_atr(df_m15: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feats = build_feature_frame(df_m15)
    out = pd.concat([df_m15.reset_index(drop=True),
                     feats[FEATURE_COLUMNS]], axis=1)
    out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    high  = out["high"].values.astype(float)
    low   = out["low"].values.astype(float)
    close = out["close"].values.astype(float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    atr = pd.Series(tr).rolling(48, min_periods=12).mean().bfill().values

    out["atr"] = atr
    return out, atr


def predict_window(
    model: PhysicsSpecialist,
    feature_window: np.ndarray,        # (W, F) float32
    device: torch.device,
) -> dict:
    with torch.no_grad():
        x = torch.from_numpy(feature_window).unsqueeze(0).to(device).float()
        out = model(x)
        logits = out["logits"].cpu().numpy()[0]
        mu     = float(out["mu"].cpu().numpy()[0])
        ls     = float(out["log_sigma"].cpu().numpy()[0])
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return {
        "p_down": float(probs[0]),
        "p_noisy": float(probs[1]),
        "p_up":   float(probs[2]),
        "mu":     mu,
        "log_sigma": ls,
    }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    model: PhysicsSpecialist,
    cfg_model: SpecialistConfig,
    rules: TradeRules,
    device: torch.device,
    initial_equity: float = 10_000.0,
    verbose: bool = False,
    stress_spread_jitter: bool = False,
    stress_spread_prob: float = 0.30,
    stress_spread_lo: float = 1.5,
    stress_spread_hi: float = 2.0,
    stress_rng: np.random.Generator | None = None,
) -> dict:
    W = cfg_model.window
    n = len(df)
    if n <= W + 5:
        raise RuntimeError("Not enough bars after feature warm-up")

    fmat = df[FEATURE_COLUMNS].astype(np.float32).values
    fmat = np.nan_to_num(fmat, nan=0.0, posinf=0.0, neginf=0.0)
    fmat = np.clip(fmat, -8.0, 8.0)

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    atr   = df["atr"].values.astype(float)
    times = pd.to_datetime(df["time_utc"], utc=True).values

    # Spread (from bar) and realistic friction
    if "spread_mean" in df.columns:
        spread_arr = df["spread_mean"].values.astype(float)
    else:
        spread_arr = np.full(n, 0.40)   # fallback
    if "spread_max" in df.columns:
        spread_max_arr = df["spread_max"].values.astype(float)
    else:
        spread_max_arr = spread_arr * 1.2

    equity = initial_equity
    peak_equity = initial_equity
    daily_pnl = 0.0
    weekly_pnl = 0.0
    last_day = None
    last_week = None

    in_trade = False
    direction = 0
    entry_idx = -1
    entry_price = 0.0
    sl = 0.0
    R_unit_price = 0.0
    risk_dollar = 0.0
    contracts = 0.0          # in equity-fraction terms (we just track $-PnL)
    partial_taken = False
    breakeven_set = False
    trail_active = False
    trail_atr_mult = rules.trail_atr_mult
    entry_conf = 0.0
    entry_dir_prob = 0.0
    captured_R_partial = 0.0   # R locked from partial close
    # Per-trade exit params (hybrid resolves at entry; used for whole ticket)
    exit_breakeven_r = rules.breakeven_at_R
    exit_partial_r = rules.partial_at_R
    exit_partial_share = rules.partial_share
    exit_trail_mult_base = rules.trail_atr_mult
    exit_trail_tight_r = rules.trail_atr_tight_R
    exit_trail_tight_mult = rules.trail_atr_tight_mult

    trades: list[TradeRecord] = []
    daily_breaker_hits = 0
    weekly_breaker_hits = 0
    halted = False
    halt_reason = ""

    if stress_spread_jitter and stress_rng is None:
        stress_rng = np.random.default_rng(0)

    # Iterate bar by bar starting from W (need full window)
    for t in range(W - 1, n - 1):
        # Daily/weekly rollover
        ts_naive = pd.Timestamp(times[t])
        ts = ts_naive.tz_localize("UTC") if ts_naive.tz is None else ts_naive.tz_convert("UTC")
        day = ts.date()
        wk = (ts.isocalendar().year, ts.isocalendar().week)
        if last_day != day:
            last_day = day
            daily_pnl = 0.0
        if last_week != wk:
            last_week = wk
            weekly_pnl = 0.0

        # Check global halt by max drawdown
        dd = 1.0 - equity / peak_equity
        if dd >= rules.max_drawdown:
            halted = True
            halt_reason = f"max_drawdown {dd*100:.2f}%"
            break

        sp_bar = float(spread_arr[t])
        if stress_spread_jitter and stress_rng is not None \
                and stress_rng.random() < stress_spread_prob:
            sp_bar *= float(stress_rng.uniform(stress_spread_lo, stress_spread_hi))

        # ---- Manage open trade ----
        if in_trade:
            bar_high = high[t]
            bar_low  = low[t]

            # 0) Friday-close — close any open trade before weekend
            if rules.skip_friday_late and ts.weekday() == 4 and ts.hour >= rules.friday_close_hour:
                exit_price = adverse_fill_exit(direction, close[t], sp_bar,
                                                atr[t], rules)
                R_realized = R_after_exit(direction, entry_price, exit_price, R_unit_price)
                pnl_dollars = (R_realized * (1.0 - exit_partial_share if partial_taken else 1.0)
                               + captured_R_partial) * risk_dollar
                pnl_pct = pnl_dollars / equity
                equity += pnl_dollars
                trades.append(TradeRecord(
                    entry_time=str(pd.Timestamp(times[entry_idx])),
                    exit_time=str(pd.Timestamp(times[t])),
                    direction=direction, entry_price=entry_price,
                    exit_price=exit_price, sl_price=sl,
                    risk_R=(R_realized * (1.0 - exit_partial_share if partial_taken else 1.0)
                            + captured_R_partial),
                    pnl_pct=pnl_pct, bars_held=t - entry_idx,
                    confidence=entry_conf, exit_reason="friday_close",
                ))
                daily_pnl += pnl_pct
                weekly_pnl += pnl_pct
                in_trade = False
                if equity > peak_equity: peak_equity = equity
                continue

            # 1) check SL hit
            if (direction == 1 and bar_low <= sl) or (direction == -1 and bar_high >= sl):
                # Stops fill at SL with negative slippage (already-adverse-side fill)
                slip_at_sl = rules.slippage_atr_frac * atr[t]
                exit_price = sl - direction * slip_at_sl    # extra slip beyond SL
                R_realized = R_after_exit(direction, entry_price, exit_price, R_unit_price)
                pnl_dollars = R_realized * risk_dollar  # already includes partial multiplier below
                # Add captured partial if not yet recorded
                pnl_dollars += captured_R_partial * risk_dollar
                pnl_pct = pnl_dollars / equity
                equity += pnl_dollars
                trade = TradeRecord(
                    entry_time=str(pd.Timestamp(times[entry_idx])),
                    exit_time=str(pd.Timestamp(times[t])),
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    sl_price=sl,
                    risk_R=R_realized + captured_R_partial,
                    pnl_pct=pnl_pct,
                    bars_held=t - entry_idx,
                    confidence=entry_conf,
                    exit_reason="sl_or_breakeven",
                )
                trades.append(trade)
                daily_pnl += pnl_pct
                weekly_pnl += pnl_pct
                in_trade = False
                if equity > peak_equity:
                    peak_equity = equity
                # Daily/weekly cap check
                if -daily_pnl >= rules.daily_loss_cap:
                    daily_breaker_hits += 1
                if -weekly_pnl >= rules.weekly_loss_cap:
                    weekly_breaker_hits += 1
                continue

            # 2) compute current R-multiple based on close
            close_t = close[t]
            R_now = R_after_exit(direction, entry_price, close_t, R_unit_price)

            # 3) breakeven shift at +1R
            if not breakeven_set and R_now >= exit_breakeven_r:
                sl = entry_price
                breakeven_set = True

            # 4) partial take — captured at actual exit price (with friction)
            if not partial_taken and R_now >= exit_partial_r:
                partial_exit_price = adverse_fill_exit(direction, close_t, sp_bar,
                                                       atr[t], rules)
                R_at_partial = R_after_exit(direction, entry_price,
                                             partial_exit_price, R_unit_price)
                captured_R_partial = exit_partial_share * R_at_partial
                partial_taken = True
                trail_active  = True

            # 5) trail tightening
            if R_now >= exit_trail_tight_r:
                trail_atr_mult = exit_trail_tight_mult

            # 6) trailing stop (only on remaining 50% / non-partial side)
            if trail_active or partial_taken:
                trail_dist = trail_atr_mult * atr[t]
                if direction == 1:
                    new_sl = max(sl, close_t - trail_dist)
                else:
                    new_sl = min(sl, close_t + trail_dist)
                sl = new_sl

            # 7) re-evaluate confidence every bar
            if (t - entry_idx) % rules.reeval_every_bars == 0:
                pred = predict_window(model, fmat[t - W + 1: t + 1], device)
                cur_dir_prob = pred["p_up"] if direction == 1 else pred["p_down"]
                opp_dir_prob = pred["p_down"] if direction == 1 else pred["p_up"]
                # Confidence-drop exit: opposing side now stronger
                if (cur_dir_prob < rules.conf_floor and opp_dir_prob > cur_dir_prob) or \
                   opp_dir_prob >= rules.conf_drop_dir:
                    # Market-out with realistic exit cost
                    exit_price = adverse_fill_exit(direction, close_t, sp_bar,
                                                    atr[t], rules)
                    R_realized = R_after_exit(direction, entry_price, exit_price, R_unit_price)
                    pnl_dollars = (R_realized * (1.0 - exit_partial_share if partial_taken else 1.0)
                                   + captured_R_partial) * risk_dollar
                    pnl_pct = pnl_dollars / equity
                    equity += pnl_dollars
                    trades.append(TradeRecord(
                        entry_time=str(pd.Timestamp(times[entry_idx])),
                        exit_time=str(pd.Timestamp(times[t])),
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        sl_price=sl,
                        risk_R=(R_realized * (1.0 - exit_partial_share if partial_taken else 1.0)
                                + captured_R_partial),
                        pnl_pct=pnl_pct,
                        bars_held=t - entry_idx,
                        confidence=entry_conf,
                        exit_reason="confidence_drop",
                    ))
                    daily_pnl += pnl_pct
                    weekly_pnl += pnl_pct
                    in_trade = False
                    if equity > peak_equity:
                        peak_equity = equity
                    continue
            continue  # don't open new trade in same bar as managing

        # ---- Open new trade?  STEP 1 FILTERS ----
        # Daily / weekly cap
        if -daily_pnl >= rules.daily_loss_cap:
            continue
        if -weekly_pnl >= rules.weekly_loss_cap:
            continue

        # Session filter (London + NY)
        hour = ts.hour
        weekday = ts.weekday()  # 0=Mon, 4=Fri
        if hour < rules.session_start_hour or hour >= rules.session_end_hour:
            continue
        if rules.skip_monday_early and weekday == 0 and hour < rules.monday_open_hour:
            continue
        if rules.skip_friday_late and weekday == 4 and hour >= rules.friday_close_hour:
            continue

        # Spread filter
        sp_t = sp_bar
        if sp_t > rules.spread_max_quote:
            continue
        if atr[t] > 0 and sp_t / atr[t] > rules.spread_atr_max_ratio:
            continue

        # Volatility regime filter (use atr_z column from features if available)
        if "atr_z" in df.columns:
            atrz = float(df["atr_z"].iloc[t])
            if atrz < rules.atr_z_min or atrz > rules.atr_z_max:
                continue

        pred = predict_window(model, fmat[t - W + 1: t + 1], device)

        # ---- UNCERTAINTY Gate ----
        if pred["p_noisy"] >= rules.max_noisy_prob:
            continue

        # ---- DIRECTIONAL filter (primary) ----
        p_up   = pred["p_up"]
        p_down = pred["p_down"]
        p_lose = min(p_up, p_down)
        p_win  = max(p_up, p_down)
        if p_win < rules.dir_prob_min:
            continue
        if p_win / max(p_lose, 1e-9) < rules.dir_ratio_min:
            continue
        direction = +1 if p_up > p_down else -1

        # ---- Optional secondary filters ----
        if rules.conf_threshold > 0:
            max_p = max(p_down, pred["p_noisy"], p_up)
            if max_p < rules.conf_threshold:
                continue
        if rules.rr_min > 0:
            mu = pred["mu"]
            predicted_R = abs(mu) if ((direction == +1 and mu > 0) or
                                       (direction == -1 and mu < 0)) else 0.5 * abs(mu)
            if predicted_R < rules.rr_min:
                continue

        # Open with realistic adverse fill
        atr_t = atr[t]
        spread_t = sp_bar * (rules.spread_widen_jumps if abs(close[t] - close[t-1]) > 3*atr_t else 1.0)
        entry_price = adverse_fill_entry(direction, close[t], spread_t, atr_t, rules)
        R_unit_price = atr_t * max(rules.sl_atr_mult, 0.1)
        sl = entry_price - direction * R_unit_price
        risk_dollar = equity * rules.risk_per_trade
        in_trade = True
        entry_idx = t
        partial_taken = False
        breakeven_set = False
        trail_active = False
        captured_R_partial = 0.0
        entry_conf = p_win
        entry_dir_prob = pred["p_up"] if direction == 1 else pred["p_down"]

        atr_z_bar = float(df["atr_z"].iloc[t]) if "atr_z" in df.columns else 0.0
        (
            exit_breakeven_r,
            exit_partial_r,
            exit_partial_share,
            exit_trail_mult_base,
            exit_trail_tight_r,
            exit_trail_tight_mult,
        ) = resolve_exit_params(rules, atr_z_bar, p_down, pred["p_noisy"], p_up)
        trail_atr_mult = exit_trail_mult_base

    # Close any open trade at last bar
    if in_trade:
        sp_last = float(spread_arr[n - 1])
        if stress_spread_jitter and stress_rng is not None \
                and stress_rng.random() < stress_spread_prob:
            sp_last *= float(stress_rng.uniform(stress_spread_lo, stress_spread_hi))
        exit_price = adverse_fill_exit(direction, close[n - 1], sp_last,
                                        atr[n - 1], rules)
        R_realized = R_after_exit(direction, entry_price, exit_price, R_unit_price)
        pnl_dollars = (R_realized * (1.0 - exit_partial_share if partial_taken else 1.0)
                       + captured_R_partial) * risk_dollar
        pnl_pct = pnl_dollars / equity
        equity += pnl_dollars
        trades.append(TradeRecord(
            entry_time=str(pd.Timestamp(times[entry_idx])),
            exit_time=str(pd.Timestamp(times[n - 1])),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl_price=sl,
            risk_R=(R_realized * (1.0 - exit_partial_share if partial_taken else 1.0)
                    + captured_R_partial),
            pnl_pct=pnl_pct,
            bars_held=(n - 1) - entry_idx,
            confidence=entry_conf,
            exit_reason="end_of_data",
        ))

    # ---- Stats ----
    trade_dicts = [asdict(t) for t in trades]
    if trades:
        Rs = np.array([t.risk_R for t in trades])
        wins = Rs > 0
        n_trades = len(trades)
        win_rate = float(wins.mean()) if n_trades > 0 else 0.0
        avg_R = float(Rs.mean())
        avg_R_win = float(Rs[wins].mean()) if wins.any() else 0.0
        avg_R_loss= float(Rs[~wins].mean()) if (~wins).any() else 0.0
        gross_win = float(Rs[wins].sum())
        gross_loss= float(-Rs[~wins].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        # equity curve
        ec = [initial_equity]
        for t in trades:
            ec.append(ec[-1] + t.pnl_pct * ec[-1])
        ec = np.array(ec)
        peak = np.maximum.accumulate(ec)
        dd_curve = 1.0 - ec / peak
        max_dd = float(dd_curve.max())
        per_R_pct = (ec[-1] / initial_equity) - 1.0
    else:
        n_trades = 0
        win_rate = avg_R = avg_R_win = avg_R_loss = pf = max_dd = per_R_pct = 0.0
        gross_win = gross_loss = 0.0

    summary = {
        "n_trades":      n_trades,
        "win_rate":      win_rate,
        "avg_R":         avg_R,
        "avg_R_win":     avg_R_win,
        "avg_R_loss":    avg_R_loss,
        "profit_factor": pf,
        "pnl_total_pct": per_R_pct,
        "max_dd_pct":    max_dd,
        "final_equity":  equity,
        "halted":        halted,
        "halt_reason":   halt_reason,
        "daily_breaker_hits":  daily_breaker_hits,
        "weekly_breaker_hits": weekly_breaker_hits,
    }
    if verbose:
        for line in [
            f"trades       = {summary['n_trades']}",
            f"win rate     = {summary['win_rate']*100:.1f}%",
            f"avg R        = {summary['avg_R']:+.3f}",
            f"avg R (win)  = {summary['avg_R_win']:+.3f}",
            f"avg R (loss) = {summary['avg_R_loss']:+.3f}",
            f"profit factor= {summary['profit_factor']:.3f}",
            f"PnL total    = {summary['pnl_total_pct']*100:+.2f}%",
            f"max DD       = {summary['max_dd_pct']*100:.2f}%",
            f"final equity = {summary['final_equity']:.2f}",
            f"halted       = {summary['halted']} ({summary['halt_reason']})",
        ]:
            print(line)

    return {"summary": summary, "trades": trade_dicts}


def R_after_exit(direction: int, entry: float, exit: float, R_unit_price: float) -> float:
    if R_unit_price <= 0:
        return 0.0
    delta = (exit - entry) * direction
    return float(delta / R_unit_price)


def adverse_fill_entry(direction: int, mid_price: float, spread: float,
                       atr: float, rules: TradeRules) -> float:
    """
    Entry price = MID + half-spread on the WORSE side + slippage.
    Long  buys at ASK = mid + 0.5*spread + slippage
    Short sells at BID = mid - 0.5*spread - slippage
    """
    half_spread = 0.5 * spread * rules.spread_mult
    slip = rules.slippage_atr_frac * atr
    total_adverse = half_spread + slip
    return mid_price + direction * total_adverse


def adverse_fill_exit(direction: int, mid_price: float, spread: float,
                      atr: float, rules: TradeRules) -> float:
    """
    Exit price (unwinding) = MID − half-spread on the WORSE side − slippage.
    Long sells at BID = mid - 0.5*spread - slippage
    Short buys at ASK = mid + 0.5*spread + slippage
    """
    half_spread = 0.5 * spread * rules.spread_mult
    slip = rules.slippage_atr_frac * atr
    total_adverse = half_spread + slip
    return mid_price - direction * total_adverse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--m15",   default="../real_data/xauusd_m15.csv")
    ap.add_argument("--start", default=None, help="OOS start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end",   default=None)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--conf",      type=float, default=0.0,
                    help="(secondary) max-class probability threshold; 0 disables")
    ap.add_argument("--rr-min",    type=float, default=0.0,
                    help="(secondary) min predicted R:R from regression head; 0 disables")
    ap.add_argument("--dir-prob",  type=float, default=0.46,
                    help="primary directional filter: P(winning_dir) ≥ this")
    ap.add_argument("--dir-ratio", type=float, default=1.15,
                    help="primary: P(win)/P(lose) ≥ this")
    ap.add_argument("--sl-atr-mult", type=float, default=1.5,
                    help="initial SL distance = this × ATR (1 R = full risk)")
    ap.add_argument("--max-noisy", type=float, default=0.42,
                    help="skip entry if P(noisy) >= this")
    ap.add_argument(
        "--profit-mode",
        choices=("fixed", "hybrid"),
        default="hybrid",
        help="exit param set: hybrid=EA-aligned profile from atr_z + probs at entry (default); "
             "fixed=single profile from breakeven/partial/trail fields only",
    )
    ap.add_argument("--out",       default="backtest_report.json")
    ap.add_argument("--device",    default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[bt] device = {device}")

    model, cfg_model = load_model(Path(args.model), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bt] model loaded ({n_params:,} params)  window={cfg_model.window}")

    df = pd.read_csv(args.m15)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc").reset_index(drop=True)
    if args.start:
        df = df[df["time_utc"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df[df["time_utc"] <= pd.Timestamp(args.end, tz="UTC")]
    df = df.reset_index(drop=True)
    print(f"[bt] M15 bars: {len(df):,}  range: {df.time_utc.iloc[0]} → {df.time_utc.iloc[-1]}")

    # We need feature warm-up: include lookback bars before start.
    # Easiest: just compute features on the slice; the function handles dropna.
    df, _ = compute_features_and_atr(df)
    print(f"[bt] usable bars after feature warmup: {len(df):,}")

    rules = TradeRules(
        conf_threshold=args.conf, rr_min=args.rr_min,
        dir_prob_min=args.dir_prob, dir_ratio_min=args.dir_ratio,
        sl_atr_mult=args.sl_atr_mult,
        max_noisy_prob=args.max_noisy,
        profit_mode=args.profit_mode,
    )
    print(f"[bt] rules: dir_prob>={rules.dir_prob_min}  dir_ratio>={rules.dir_ratio_min}  "
          f"conf>={rules.conf_threshold}  rr_min={rules.rr_min}  sl_ATR×{rules.sl_atr_mult}  "
          f"profit_mode={rules.profit_mode}")
    print(f"[bt]        risk/trade={rules.risk_per_trade*100:.2f}%  "
          f"daily/weekly cap={rules.daily_loss_cap*100:.1f}/{rules.weekly_loss_cap*100:.1f}%  "
          f"maxDD={rules.max_drawdown*100:.0f}%")

    print()
    print("[bt] running simulation...")
    result = run_backtest(df, model, cfg_model, rules, device,
                          initial_equity=args.initial_equity, verbose=True)

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[bt] saved → {out}")


if __name__ == "__main__":
    main()
