#!/usr/bin/env python3
"""Parse training logs and plot validation metrics progression.

Works with log lines like:
  [seed=42 fold=0] epoch=31 val_logloss=0.605384

Usage:
  python plot_metrics_from_log.py --log logs/.../01_jax_teacher.log

Outputs PNGs into <log_dir>/plots/ by default.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


LINE_RE = re.compile(
    r"^\[seed=(?P<seed>\d+)\s+fold=(?P<fold>\d+)\]\s+epoch=(?P<epoch>\d+)\s+(?P<metric>val_[A-Za-z0-9_]+)=(?P<value>[-+0-9.]+(?:[eE][-+]?\d+)?)\s*$"
)


def parse_lines(lines: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for line in lines:
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "seed": int(d["seed"]),
                "fold": int(d["fold"]),
                "epoch": int(d["epoch"]),
                "metric": d["metric"],
                "value": float(d["value"]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["seed", "fold", "epoch", "metric", "value"])

    df = (
        pd.DataFrame(rows)
        .sort_values(["metric", "seed", "fold", "epoch"])
        .reset_index(drop=True)
    )
    return df


def metric_direction(metric: str) -> str:
    """Return 'min' or 'max' depending on what we want to optimize."""
    m = metric.lower()
    if "auc" in m:
        return "max"
    if "loss" in m:
        return "min"
    # conservative default: assume it's a loss-like metric
    return "min"


def plot_metric(
    df_metric: pd.DataFrame, *, metric: str, outpath: Path, title_prefix: str = ""
) -> None:
    if df_metric.empty:
        return

    plt.figure(figsize=(10, 5))

    # one line per (seed, fold)
    for (seed, fold), g in df_metric.groupby(["seed", "fold"], sort=True):
        plt.plot(
            g["epoch"],
            g["value"],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"seed={seed} fold={fold}",
        )

    direction = metric_direction(metric)
    if direction == "min":
        best_idx = df_metric["value"].idxmin()
    else:
        best_idx = df_metric["value"].idxmax()

    best_epoch = int(df_metric.loc[best_idx, "epoch"]) if best_idx is not None else None
    best_value = (
        float(df_metric.loc[best_idx, "value"]) if best_idx is not None else None
    )

    ttl = f"{title_prefix}{metric}"
    if best_epoch is not None and best_value is not None:
        ttl += f" | best={best_value:.6f} @ epoch={best_epoch}"

    plt.title(ttl)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)

    # If too many lines, legend becomes unreadable; keep it but make it compact.
    if df_metric[["seed", "fold"]].drop_duplicates().shape[0] <= 12:
        plt.legend(loc="best", fontsize=8)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True, help="Path to the .log file")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <log_dir>/plots)",
    )
    args = ap.parse_args()

    log_path: Path = args.log
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    outdir = args.outdir if args.outdir is not None else (log_path.parent / "plots")

    df = parse_lines(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    if df.empty:
        raise SystemExit(
            "No matching metric lines found. Expected lines like: '[seed=42 fold=0] epoch=1 val_logloss=0.603455'"
        )

    stem = log_path.stem

    # Save CSV for convenience
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{stem}_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Plot each metric
    for metric in sorted(df["metric"].unique()):
        df_m = df[df["metric"] == metric]
        outpath = outdir / f"{stem}_{metric}.png"
        plot_metric(df_m, metric=metric, outpath=outpath, title_prefix=f"{stem} | ")

    # Print quick summary
    print(f"Parsed {len(df)} points from {log_path}")
    print(f"Metrics: {sorted(df['metric'].unique())}")
    print(f"Wrote: {csv_path}")
    for metric in sorted(df["metric"].unique()):
        df_m = df[df["metric"] == metric]
        direction = metric_direction(metric)
        best = df_m["value"].min() if direction == "min" else df_m["value"].max()
        print(f"  {metric}: best={best:.6f} ({direction})")
    print(f"Plots in: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
