#!/usr/bin/env python3
"""Stacking meta-learner in JAX (logistic regression on OOF preds).

Inputs:
- One or more OOF files (id,y,oof_pred)
- Matching submissions (id,diagnosed_diabetes)

Trains a regularized logistic regression in JAX and applies it to test preds.
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


def _read_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "y", "oof_pred"}
    if set(df.columns) != need:
        raise ValueError(f"Bad columns in {path}: {list(df.columns)} (need {sorted(need)})")
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(np.int64)
    df["y"] = np.where(df["y"].to_numpy() > 0, 1, 0).astype(np.int64)
    df["oof_pred"] = pd.to_numeric(df["oof_pred"], errors="coerce").fillna(0.5).clip(0, 1).astype(np.float64)
    return df


def _read_sub(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "diagnosed_diabetes"}
    if set(df.columns) != need:
        raise ValueError(f"Bad columns in {path}: {list(df.columns)} (need {sorted(need)})")
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    df["diagnosed_diabetes"] = pd.to_numeric(df["diagnosed_diabetes"], errors="coerce").fillna(0.5).clip(0, 1).astype(np.float64)
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OOF stacking (JAX logistic regression)")
    p.add_argument("--oof", nargs="+", type=Path, required=True)
    p.add_argument("--subs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("sub/submission_stack_jax.csv"))

    p.add_argument("--features", type=str, default="logit", help="prob|logit|prob,logit")
    p.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization (like sklearn)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    if len(args.oof) != len(args.subs):
        raise ValueError("--oof and --subs must have same number of files")

    import jax
    import jax.numpy as jnp
    import optax

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

    feat_set = [s.strip().lower() for s in str(args.features).split(",") if s.strip()]
    if any(s not in {"prob", "logit"} for s in feat_set):
        raise ValueError("--features must be prob|logit|prob,logit")

    P_oof = np.vstack([df["oof_pred"].to_numpy(dtype=np.float64) for df in oofs]).T  # (n, m)
    P_test = np.vstack([df["diagnosed_diabetes"].to_numpy(dtype=np.float64) for df in subs]).T

    def build_X(P: np.ndarray) -> np.ndarray:
        cols = []
        if "prob" in feat_set:
            cols.append(P)
        if "logit" in feat_set:
            cols.append(_logit(P))
        X = np.hstack(cols) if len(cols) > 1 else cols[0]
        return X.astype(np.float32)

    X = build_X(P_oof)
    X_test = build_X(P_test)

    y_f = y.astype(np.float32)

    # Params: w (d,), b scalar
    key = jax.random.PRNGKey(int(args.seed))
    w = jax.random.normal(key, (X.shape[1],), dtype=jnp.float32) * 0.01
    b = jnp.array(0.0, dtype=jnp.float32)

    Xj = jnp.asarray(X)
    yj = jnp.asarray(y_f)

    reg = 1.0 / float(args.C)

    def loss_fn(params):
        w, b = params
        logits = Xj @ w + b
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, yj))
        loss = loss + 0.5 * reg * jnp.sum(w * w)
        return loss

    opt = optax.adam(float(args.lr))
    opt_state = opt.init((w, b))

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    params = (w, b)
    last = None
    for i in range(int(args.steps)):
        params, opt_state, loss = step(params, opt_state)
        if args.verbose >= 2 and (i % 200 == 0 or i == int(args.steps) - 1):
            print(f"[stack-jax] step={i} loss={float(loss):.6f}", flush=True)
        last = loss

    w_hat, b_hat = params

    # OOF metrics
    p_oof = jax.nn.sigmoid(Xj @ w_hat + b_hat)
    p_oof_np = np.asarray(jnp.clip(p_oof, 1e-15, 1 - 1e-15), dtype=np.float64)
    ll = float((-(y_f * np.log(p_oof_np) + (1 - y_f) * np.log1p(-p_oof_np))).mean())

    Xt = jnp.asarray(X_test)
    p_test = jax.nn.sigmoid(Xt @ w_hat + b_hat)
    p_test_np = np.asarray(jnp.clip(p_test, 0.0, 1.0), dtype=np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": p_test_np})
    out.to_csv(args.out, index=False)

    if args.verbose:
        print(f"[stack-jax] models={len(args.oof)} features={feat_set} C={float(args.C)} steps={int(args.steps)} lr={float(args.lr)}", flush=True)
        print(f"[stack-jax] OOF logloss={ll:.6f}", flush=True)
        print(f"[done] wrote {args.out} rows={len(out)}", flush=True)

    if args.zip_output:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname=args.out.name)
        if args.verbose:
            print(f"[done] wrote {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
