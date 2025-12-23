#!/usr/bin/env python3
"""Blend multiple Kaggle submissions using JAX math.

Assumes all inputs have columns: id, diagnosed_diabetes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blend submissions (JAX)")
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("sub/submission_blend_jax.csv"))
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--mode", type=str, default="prob", help="prob|logit")
    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args(argv)

    import jax
    import jax.numpy as jnp

    if len(args.inputs) < 2:
        raise ValueError("Provide at least 2 --inputs")

    frames = []
    for ip in args.inputs:
        df = pd.read_csv(ip)
        if set(df.columns) != {"id", "diagnosed_diabetes"}:
            raise ValueError(f"Bad columns in {ip}: {list(df.columns)}")
        frames.append(df)

    ids = frames[0]["id"].to_numpy()
    for i, df in enumerate(frames[1:], start=2):
        if not np.array_equal(ids, df["id"].to_numpy()):
            raise ValueError(f"IDs do not match between inputs[0] and inputs[{i-1}]")

    preds = np.vstack(
        [df["diagnosed_diabetes"].to_numpy(dtype=np.float64) for df in frames]
    )  # (m, n)

    if args.weights is None:
        w = np.ones(preds.shape[0], dtype=np.float64) / preds.shape[0]
    else:
        w = np.asarray(args.weights, dtype=np.float64)
        if len(w) != preds.shape[0]:
            raise ValueError("--weights must have same length as --inputs")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("--weights must be non-negative and sum > 0")
        w = w / w.sum()

    mode = str(args.mode).strip().lower()
    if mode not in {"prob", "logit"}:
        raise ValueError("--mode must be prob|logit")

    P = jnp.asarray(preds, dtype=jnp.float64)
    ww = jnp.asarray(w, dtype=jnp.float64)

    if mode == "prob":
        blended = jnp.clip(jnp.sum(P.T * ww[None, :], axis=1), 0.0, 1.0)
    else:
        z = jnp.asarray(_logit(preds), dtype=jnp.float64)
        blended = jnp.clip(jax.nn.sigmoid(jnp.sum(z.T * ww[None, :], axis=1)), 0.0, 1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"id": ids, "diagnosed_diabetes": np.asarray(blended)})
    out_df.to_csv(args.out, index=False)

    if args.verbose:
        print(f"[blend-jax] wrote {args.out} from {len(args.inputs)} files", flush=True)

    if args.zip_output:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(args.out, arcname=args.out.name)
        if args.verbose:
            print(f"[blend-jax] wrote {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
