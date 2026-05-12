#!/usr/bin/env python3
"""
After training, if backtest shows 0 trades on the val window, summarize how tight
entry gates are vs the model's softmax distribution.

Usage:
  python3 val_prob_gate_report.py --model runs/<tag>/best.pt \\
      --m15 ../real_data/xauusd_m15.csv --start 2025-04-07 --end 2026-04-30 --device mps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest_specialist import compute_features_and_atr, load_model, predict_window
from physics_features import FEATURE_COLUMNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--m15", type=Path, default=Path("../real_data/xauusd_m15.csv"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.model, device)

    df = pd.read_csv(args.m15)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df[(df["time_utc"] >= pd.Timestamp(args.start, tz="UTC"))
            & (df["time_utc"] <= pd.Timestamp(args.end, tz="UTC"))].reset_index(drop=True)
    df, _ = compute_features_and_atr(df)
    W = cfg.window
    n = len(df)
    fmat = np.clip(
        np.nan_to_num(df[list(FEATURE_COLUMNS)].astype(np.float32).values,
                      nan=0.0, posinf=0.0, neginf=0.0),
        -8.0,
        8.0,
    )

    ps, pn, pl = [], [], []
    p_win, ratio = [], []
    for t in range(W, n - 1):
        pred = predict_window(model, fmat[t - W + 1 : t + 1], device)
        pu, pd_ = pred["p_up"], pred["p_down"]
        ps.append(pd_)
        pn.append(pred["p_noisy"])
        pl.append(pu)
        pw = max(pu, pd_)
        pls = min(pu, pd_)
        p_win.append(pw)
        ratio.append(pw / max(pls, 1e-9))

    ps, pn, pl = map(np.asarray, (ps, pn, pl))
    p_win = np.asarray(p_win)
    ratio = np.asarray(ratio)

    def pct(x: np.ndarray, q: float) -> float:
        return float(np.quantile(x, q))

    print("=== Softmax marginals (all val bars with features) ===")
    print(f"n_bars={len(pn)}")
    for name, arr in [("p_down", ps), ("p_noisy", pn), ("p_up", pl), ("p_win", p_win), ("p_win/p_lose", ratio)]:
        print(f"\n{name}: min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")
        print(f"  p10/p25/p50/p75/p90: {pct(arr,0.1):.4f} {pct(arr,0.25):.4f} {pct(arr,0.5):.4f} "
              f"{pct(arr,0.75):.4f} {pct(arr,0.9):.4f}")

    # Gate pass rates (session ignored here — same as unrestricted clock)
    max_noisy = 0.42
    dir_prob = 0.30
    dir_ratio = 1.10
    conf = 0.0
    ok = (pn < max_noisy) & (p_win >= dir_prob) & (ratio >= dir_ratio)
    if conf > 0:
        mx = np.maximum.reduce([ps, pn, pl])
        ok = ok & (mx >= conf)
    print("\n=== Default gate pass rate (no session / spread / atr_z) ===")
    print(f"max_noisy<{max_noisy}, p_win>={dir_prob}, ratio>={dir_ratio}: "
          f"{100.0 * ok.mean():.2f}% of bars")

    for max_noisy in (0.45, 0.48, 0.50):
        ok2 = (pn < max_noisy) & (p_win >= dir_prob) & (ratio >= dir_ratio)
        print(f"  max_noisy<{max_noisy}: {100.0 * ok2.mean():.2f}%")

    print("\nSuggest: lower max_noisy toward p75–p90 of p_noisy if you need more trades; "
          "lower dir_prob / dir_ratio toward p50 of p_win / ratio.")


if __name__ == "__main__":
    main()
