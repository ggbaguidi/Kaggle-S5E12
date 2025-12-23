#!/usr/bin/env python3
"""Optimize ensemble weights using OOF logloss and apply to test submissions.

Given:
- OOF files: each `id,y,oof_pred`
- Test submissions: each `id,diagnosed_diabetes`

This script:
1) Checks strict ID alignment across OOFs and subs.
2) Finds weights that minimize OOF logloss for either:
   - prob space: p = sum_i w_i p_i
   - logit space: p = sigmoid(sum_i w_i logit(p_i))
3) Writes blended test submission.

Optimization:
- Random simplex search (Dirichlet) + optional local refinement.
- CPU-only, no heavy deps beyond numpy/pandas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float((-(y * np.log(p) + (1 - y) * np.log1p(-p))).mean())


def _read_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "y", "oof_pred"}
    if set(df.columns) != need:
        raise ValueError(
            f"Bad columns in {path}: {list(df.columns)} (need {sorted(need)})"
        )
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(np.int64)
    df["y"] = np.where(df["y"].to_numpy() > 0, 1, 0).astype(np.int64)
    df["oof_pred"] = (
        pd.to_numeric(df["oof_pred"], errors="coerce")
        .fillna(0.5)
        .clip(0, 1)
        .astype(np.float64)
    )
    return df


def _read_sub(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "diagnosed_diabetes"}
    if set(df.columns) != need:
        raise ValueError(
            f"Bad columns in {path}: {list(df.columns)} (need {sorted(need)})"
        )
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    df["diagnosed_diabetes"] = (
        pd.to_numeric(df["diagnosed_diabetes"], errors="coerce")
        .fillna(0.5)
        .clip(0, 1)
        .astype(np.float64)
    )
    return df


@dataclass
class Best:
    loss: float
    w: np.ndarray


def _blend(P: np.ndarray, w: np.ndarray, mode: str) -> np.ndarray:
    # P: (n_rows, n_models)
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s

    if mode == "prob":
        return np.clip(P @ w, 0.0, 1.0)
    if mode == "logit":
        Z = _logit(P)
        return np.clip(_sigmoid(Z @ w), 0.0, 1.0)
    raise ValueError("mode must be prob or logit")


def _dirichlet_samples(
    rng: np.random.Generator, n: int, k: int, alpha: float
) -> np.ndarray:
    # returns (n, k)
    a = np.full(k, float(alpha), dtype=np.float64)
    return rng.dirichlet(a, size=int(n))


def _local_refine(
    rng: np.random.Generator,
    best: Best,
    P: np.ndarray,
    y: np.ndarray,
    mode: str,
    steps: int,
) -> Best:
    w = best.w.copy()
    loss = best.loss
    k = len(w)
    for _ in range(int(steps)):
        # small random move on simplex
        j = rng.integers(0, k)
        delta = rng.normal(0.0, 0.05)
        w2 = w.copy()
        w2[j] = max(0.0, w2[j] + delta)
        # renorm
        s = w2.sum()
        if s <= 0:
            continue
        w2 = w2 / s
        p2 = _blend(P, w2, mode)
        l2 = _logloss(y, p2)
        if l2 < loss:
            w, loss = w2, l2
    return Best(loss=loss, w=w)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Optimize OOF blend weights")
    p.add_argument("--oof", nargs="+", type=Path, required=True)
    p.add_argument("--subs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("sub/submission_oof_opt.csv"))

    p.add_argument("--mode", type=str, default="logit", help="prob|logit")
    p.add_argument("--trials", type=int, default=5000)
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet concentration (lower => sparser)",
    )
    p.add_argument("--refine-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    if len(args.oof) != len(args.subs):
        raise ValueError("--oof and --subs must have same number of files")

    mode = str(args.mode).strip().lower()
    if mode not in {"prob", "logit"}:
        raise ValueError("--mode must be prob|logit")

    oofs = [_read_oof(pth) for pth in args.oof]
    subs = [_read_sub(pth) for pth in args.subs]

    train_ids = oofs[0]["id"].to_numpy()
    y = oofs[0]["y"].to_numpy(dtype=np.int64)

    for i, df in enumerate(oofs[1:], start=2):
        if not np.array_equal(train_ids, df["id"].to_numpy()):
            raise ValueError(f"Train IDs mismatch between oof[0] and oof[{i-1}]")
        if not np.array_equal(y, df["y"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Targets mismatch between oof[0] and oof[{i-1}]")

    test_ids = subs[0]["id"].to_numpy()
    for i, df in enumerate(subs[1:], start=2):
        if not np.array_equal(test_ids, df["id"].to_numpy()):
            raise ValueError(f"Test IDs mismatch between subs[0] and subs[{i-1}]")

    P_oof = np.vstack(
        [df["oof_pred"].to_numpy(dtype=np.float64) for df in oofs]
    ).T  # (n, m)
    P_test = np.vstack(
        [df["diagnosed_diabetes"].to_numpy(dtype=np.float64) for df in subs]
    ).T

    base_losses = []
    for i in range(P_oof.shape[1]):
        base_losses.append(_logloss(y, P_oof[:, i]))

    rng = np.random.default_rng(int(args.seed))

    # Start from uniform
    w0 = np.ones(P_oof.shape[1], dtype=np.float64) / P_oof.shape[1]
    p0 = _blend(P_oof, w0, mode)
    best = Best(loss=_logloss(y, p0), w=w0)

    # Random search
    W = _dirichlet_samples(rng, int(args.trials), P_oof.shape[1], float(args.alpha))
    for w in W:
        p = _blend(P_oof, w, mode)
        l = _logloss(y, p)
        if l < best.loss:
            best = Best(loss=l, w=w)

    # Local refinement
    if int(args.refine_steps) > 0:
        best = _local_refine(rng, best, P_oof, y, mode, int(args.refine_steps))

    # Apply to test
    p_test = _blend(P_test, best.w, mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {"id": test_ids, "diagnosed_diabetes": np.clip(p_test, 0.0, 1.0)}
    )
    out.to_csv(args.out, index=False)

    if args.verbose:
        names = [p.name for p in args.oof]
        order = np.argsort(best.w)[::-1]
        top = [
            (names[i], float(best.w[i]))
            for i in order[: min(10, len(order))]
            if best.w[i] > 1e-4
        ]
        print(
            f"[opt] mode={mode} models={len(names)} trials={int(args.trials)} alpha={float(args.alpha)} refine={int(args.refine_steps)}",
            flush=True,
        )
        print(f"[opt] best_logloss={best.loss:.6f}", flush=True)
        print(
            f"[opt] base_logloss(min/mean/max)={float(np.min(base_losses)):.6f}/{float(np.mean(base_losses)):.6f}/{float(np.max(base_losses)):.6f}",
            flush=True,
        )
        print(f"[opt] weights(top)={top}", flush=True)
        print(f"[done] wrote {args.out} rows={len(out)}", flush=True)

    if args.zip_output:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(args.out, arcname=args.out.name)
        if args.verbose:
            print(f"[done] wrote {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
