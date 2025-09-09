#!/usr/bin/env python3
"""
EMG Preprocessing — Single, Standalone (Activation-Timing Focus)

Used argument
    --in /.csv 위치
    --fs 250
    --band 20 100
    --env-cutoff 10
    --detrend median
    --trim-sec 0.5
    --plot-mode both 
    --plot-pct 98 
    --plot-norm zscore
    --plot


Outputs (next to the input CSV by default):
  - *_act.csv           : CH*_bp (bandpassed), CH*_env (envelope)
  - *_act_preview.png   : full preview
  - *_act_zoom.png      : zoom preview

Requires: pip install numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Headless-safe matplotlib
import matplotlib
try:
    if "matplotlib.backends" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt


# ------------------------
# Filtering helpers
# ------------------------
def butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(f"Band must satisfy 0<low<high<{nyq:.3f} for fs={fs}. Got {low_hz}–{high_hz}.")
    b, a = butter(order, [low_hz/nyq, high_hz/nyq], btype='band')
    return b, a

def butter_lowpass(cut_hz: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    if not (0 < cut_hz < nyq):
        raise ValueError(f"Low-pass cutoff must be in (0,{nyq:.3f}) for fs={fs}. Got {cut_hz}.")
    b, a = butter(order, cut_hz/nyq, btype='low')
    return b, a

def _safe_padlen(b: np.ndarray, a: np.ndarray, n: int) -> int:
    base = 3 * (max(len(a), len(b)) - 1)
    if n <= 1:
        return 0
    return max(0, min(base, n - 1))

def bandpass_filter(X: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order=order)
    padlen = _safe_padlen(b, a, X.shape[0])
    return filtfilt(b, a, X, axis=0, padlen=padlen)

def lowpass_filter(X: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter_lowpass(cutoff, fs, order=order)
    padlen = _safe_padlen(b, a, X.shape[0])
    return filtfilt(b, a, X, axis=0, padlen=padlen)


# ------------------------
# Data helpers
# ------------------------
def guess_emg_columns(df: pd.DataFrame, required: int = 8) -> List[str]:
    """Return 8 EMG columns. Prefer CH1..CH8 / Channel1.. / EMG1.. . Fallback: first 8 numeric (exclude time)."""
    lower = {c.lower().strip(): c for c in df.columns}
    for pref in ("ch", "channel", "emg"):
        cols = [lower.get(f"{pref}{i}") for i in range(1, required+1)]
        cols = [c for c in cols if c is not None]
        if len(cols) == required:
            return cols
    for pref in ("ch", "channel", "emg"):
        cols = [lower.get(f"{pref}{i:02d}") for i in range(1, required+1)]
        cols = [c for c in cols if c is not None]
        if len(cols) == required:
            return cols
    num = df.select_dtypes(include='number')
    num = num.drop(columns=[c for c in num.columns if c.lower() in ("time", "timestamp", "t", "ms", "s")], errors='ignore')
    if num.shape[1] < required:
        raise ValueError(f"Need {required} numeric EMG columns; found {num.shape[1]}. Rename to CH1..CH8 or use --names.")
    return list(num.columns[:required])


# ------------------------
# Plotting
# ------------------------
def quick_plot(bp: np.ndarray, env: np.ndarray, fs: float, ch_names: List[str], out_png: Path,
               show: bool = False, skip_sec: float = 0.5, pct: float = 99.0,
               mode: str = 'env', norm: bool = True) -> None:
    t = np.arange(bp.shape[0]) / fs
    n = bp.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    skip = int(round(skip_sec * fs))

    # Normalize (plot only) so small channels remain visible
    B = bp.copy(); E = env.copy()
    if norm and B.shape[0] > skip:
        seg = slice(skip, None)
        def _z(x: np.ndarray) -> np.ndarray:
            mu = np.nanmean(x[seg, :], axis=0, keepdims=True)
            sd = np.nanstd(x[seg, :], axis=0, ddof=0, keepdims=True)
            sd[sd == 0] = 1.0
            return (x - mu) / sd
        B = _z(B); E = _z(E)

    for i, ax in enumerate(axes):
        if mode in ("both", "bp"):
            ax.plot(t, B[:, i], lw=0.7, label='bandpassed')
        if mode in ("both", "env"):
            ax.plot(t, E[:, i], lw=1.0, label='envelope')
        ax.set_ylabel(ch_names[i])
        if mode == 'env':
            seg = E[skip:, i]
            if seg.size > 0:
                a = np.percentile(seg, pct)
                if a > 0:
                    ax.set_ylim(0, a * 1.05)
        else:
            seg = B[skip:, i]
            if seg.size > 0:
                a = np.percentile(np.abs(seg), pct)
                if a > 0:
                    ax.set_ylim(-a * 1.05, a * 1.05)
        if i == 0:
            ax.legend(loc='upper right', frameon=False)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------
# Main
# ------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="EMG preprocessing (single-file, activation-timing focus)")
    ap.add_argument('--in', dest='in_path', required=True, help='Input CSV path')
    ap.add_argument('--out', dest='out_path', default=None, help='Output CSV path (default: *_act.csv next to input)')
    ap.add_argument('--fs', type=float, default=250.0, help='Sampling rate in Hz (default: 250)')
    ap.add_argument('--band', nargs=2, type=float, metavar=('LOW', 'HIGH'), default=[20.0, 100.0], help='Band-pass in Hz')
    ap.add_argument('--env-cutoff', type=float, default=10.0, help='Envelope LPF cutoff (Hz), default 10')
    ap.add_argument('--detrend', choices=['none', 'mean', 'median'], default='median', help='Baseline removal (default: median)')
    ap.add_argument('--names', nargs=8, default=None, help='Explicit names of 8 EMG columns (override auto-detect)')
    ap.add_argument('--timecol', default=None, help='Optional time column to keep')
    ap.add_argument('--trim-sec', type=float, default=0.5, help='Trim first N seconds in outputs & plots')
    ap.add_argument('--view-start', type=float, default=10.0, help='Zoom preview start time (s)')
    ap.add_argument('--view-dur', type=float, default=2.0, help='Zoom preview duration (s)')
    ap.add_argument('--plot', action='store_true', help='Also show interactive plot (PNGs saved regardless)')
    ap.add_argument('--plot-pct', type=float, default=98.0, help='Percentile for autoscaling y-limits per channel (e.g., 95–99.5).')
    ap.add_argument('--plot-skip-sec', type=float, default=0.5, help='Ignore first N seconds when autoscaling (to avoid transients).')
    ap.add_argument('--plot-mode', choices=['env', 'bp', 'both'], default='env', help='What to draw in previews: envelope, bandpassed, or both.')
    ap.add_argument('--plot-norm', choices=['zscore', 'none'], default='zscore',help='Per-channel normalization for plotting only (does not affect CSV).')
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Decide output paths
    out_path = Path(args.out_path) if args.out_path else in_path.with_name(in_path.stem + "_act.csv")
    png_full = in_path.with_name(in_path.stem + "_act_preview.png")
    png_zoom = in_path.with_name(in_path.stem + "_act_zoom.png")

    print(f"[INFO] Loading CSV: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)

    # Time column
    time_col = args.timecol if args.timecol in df.columns else None
    if time_col is None:
        for c in df.columns:
            if c.lower() in ("time", "timestamp", "t"):
                time_col = c
                break

    # Pick 8 EMG columns
    if args.names:
        emg_cols = list(args.names)
        missing = [c for c in emg_cols if c not in df.columns]
        if missing:
            print(f"[ERROR] EMG column(s) not found in CSV: {missing}", file=sys.stderr); sys.exit(1)
    else:
        emg_cols = guess_emg_columns(df, 8)
    print(f"[INFO] Using EMG columns: {emg_cols}")
    if time_col:
        print(f"[INFO] Preserving time column: {time_col}")

    X = df[emg_cols].to_numpy(dtype=float, copy=True)
    fs = float(args.fs)
    low, high = float(args.band[0]), float(args.band[1])

    # 1) Detrend
    if args.detrend == 'mean':
        baseline = np.nanmean(X, axis=0, keepdims=True)
    elif args.detrend == 'median':
        baseline = np.nanmedian(X, axis=0, keepdims=True)
    else:
        baseline = 0.0
    X = X - baseline

    # 2) Band-pass
    print(f"[INFO] Band-pass: {low}-{high} Hz @ fs={fs}")
    X_bp = bandpass_filter(X, low, high, fs, order=4)

    # 3) Rectify + LPF envelope
    print(f"[INFO] Envelope: rectified + low-pass {args.env_cutoff} Hz")
    rect = np.abs(X_bp)
    X_env = lowpass_filter(rect, args.env_cutoff, fs, order=4)

    # 4) Optional trim
    trim_n = int(round(max(0.0, float(args.trim_sec)) * fs))
    if trim_n > 0 and trim_n < X_bp.shape[0]:
        print(f"[INFO] Trimming first {args.trim_sec} s ({trim_n} samples)")
        X_bp = X_bp[trim_n:, :]; X_env = X_env[trim_n:, :]
        if time_col and time_col in df.columns and len(df[time_col]) >= trim_n:
            df = df.iloc[trim_n:, :].reset_index(drop=True)

    # 5) Save CSV
    out_bp = pd.DataFrame(X_bp, columns=[f"{c}_bp" for c in emg_cols])
    out_env = pd.DataFrame(X_env, columns=[f"{c}_env" for c in emg_cols])
    if time_col and time_col in df.columns and len(df[time_col]) == len(out_bp):
        out_df = pd.concat([df[[time_col]].reset_index(drop=True), out_bp, out_env], axis=1)
    else:
        out_df = pd.concat([out_bp, out_env], axis=1)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved CSV  -> {out_path}")

    # 6) Save previews (full + zoom)
    try:
        quick_plot(
            X_bp, X_env, fs, emg_cols, png_full,
            show=args.plot,
            skip_sec=float(args.plot_skip_sec),
            pct=float(args.plot_pct),
            mode=args.plot_mode,
            norm=(args.plot_norm == 'zscore')
        )
        print(f"[INFO] Saved FIG  -> {png_full}")
    except Exception as e:
        print(f"[WARN] preview(full) failed: {e}")

    try:
        s = int(round(float(args.view_start) * fs))
        e = s + int(round(float(args.view_dur) * fs))
        s = max(0, min(s, X_bp.shape[0]-1)); e = max(s+1, min(e, X_bp.shape[0]))
        quick_plot(
            X_bp[s:e, :], X_env[s:e, :], fs, emg_cols, png_zoom,
            show=False,
            skip_sec=0.0,
            pct=float(args.plot_pct),
            mode=args.plot_mode,
            norm=(args.plot_norm == 'zscore')
        )
        print(f"[INFO] Saved FIG  -> {png_zoom}")
    except Exception as e:
        print(f"[WARN] preview(zoom) failed: {e}")

    print("[DONE]")

if __name__ == "__main__":
    main()