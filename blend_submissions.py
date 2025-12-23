#!/usr/bin/env python3
"""Blend multiple Kaggle submissions by averaging probabilities (or logits).

Usage:
  python blend_submissions.py --inputs sub/a.csv sub/b.csv --out sub/blend.csv
  python blend_submissions.py --inputs sub/a.csv sub/b.csv --weights 0.6 0.4 --zip-output

Assumes all inputs have columns: id, diagnosed_diabetes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    z = np.asarray(z)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def blend(
    inputs: list[Path],
    out: Path,
    weights: list[float] | None,
    mode: str,
    zip_output: bool,
    verbose: int,
) -> None:
    if len(inputs) < 2:
        raise ValueError("Provide at least 2 --inputs")

    frames: list[pd.DataFrame] = []
    for p in inputs:
        df = pd.read_csv(p)
        if set(df.columns) != {"id", "diagnosed_diabetes"}:
            raise ValueError(f"Bad columns in {p}: {list(df.columns)}")
        frames.append(df)

    ids = frames[0]["id"].to_numpy()
    for i, df in enumerate(frames[1:], start=2):
        if not np.array_equal(ids, df["id"].to_numpy()):
            raise ValueError(f"IDs do not match between inputs[0] and inputs[{i-1}]")

    preds = np.vstack(
        [df["diagnosed_diabetes"].to_numpy(dtype=np.float64) for df in frames]
    )

    if weights is None:
        w = np.ones(preds.shape[0], dtype=np.float64) / preds.shape[0]
    else:
        if len(weights) != preds.shape[0]:
            raise ValueError("--weights must have same length as --inputs")
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any():
            raise ValueError("--weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("--weights sum must be > 0")
        w = w / w.sum()

    mode = str(mode).strip().lower()
    if mode not in {"prob", "logit"}:
        raise ValueError("--mode must be one of: prob, logit")

    if mode == "prob":
        blended = np.clip((preds.T @ w), 0.0, 1.0)
    else:
        # Average in logit space, then map back to probabilities.
        z = _logit(preds)
        blended = np.clip(_sigmoid(z.T @ w), 0.0, 1.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": ids, "diagnosed_diabetes": blended})
    sub.to_csv(out, index=False)

    if verbose:
        print(f"[blend] wrote {out} from {len(inputs)} files", flush=True)

    if zip_output:
        zip_path = out.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(out, arcname=out.name)
        if verbose:
            print(f"[blend] wrote {zip_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blend multiple submission CSVs")
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("submission_blend.csv"))
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument(
        "--mode",
        type=str,
        default="prob",
        help="Blend mode: prob (average probabilities) or logit (average logits)",
    )
    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args(argv)

    blend(
        inputs=args.inputs,
        out=args.out,
        weights=args.weights,
        mode=args.mode,
        zip_output=bool(args.zip_output),
        verbose=int(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
