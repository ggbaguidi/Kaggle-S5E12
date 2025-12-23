#!/usr/bin/env python3
"""Stacking meta-learner using OOF predictions.

This fits a simple regularized logistic regression on OOF predictions from
multiple base models, then applies it to their test predictions.

Inputs:
- One or more OOF files, each with columns: id, y, oof_pred
- Matching submission files, each with columns: id, diagnosed_diabetes

Example:
  python stack_oof.py \
    --oof sub/teacher_oof.csv sub/cat_oof.csv \
    --subs sub/submission_teacher.csv sub/submission_cat.csv \
    --out sub/submission_stack.csv --zip-output

Notes:
- Uses strict id alignment; errors if ids don’t match.
- You can optionally include raw probs and/or logits as features.
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
    z = np.asarray(z)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


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
    df["y"] = np.where(df["y"].to_numpy() > 0, 1, 0)
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stacking with logistic regression on OOF predictions"
    )
    p.add_argument(
        "--oof", nargs="+", type=Path, required=True, help="OOF CSVs (id,y,oof_pred)"
    )
    p.add_argument(
        "--subs",
        nargs="+",
        type=Path,
        required=True,
        help="Test submissions (id,diagnosed_diabetes)",
    )
    p.add_argument("--out", type=Path, default=Path("sub/submission_stack.csv"))

    p.add_argument(
        "--features",
        type=str,
        default="logit",
        help="Feature set: prob|logit|prob,logit",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization for LogisticRegression",
    )
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    if len(args.oof) != len(args.subs):
        raise ValueError(
            "--oof and --subs must have the same number of files (aligned order)"
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    oofs = [_read_oof(pth) for pth in args.oof]
    subs = [_read_sub(pth) for pth in args.subs]

    train_ids = oofs[0]["id"].to_numpy()
    y = oofs[0]["y"].to_numpy(dtype=np.int64)

    for i, df in enumerate(oofs[1:], start=2):
        if not np.array_equal(train_ids, df["id"].to_numpy()):
            raise ValueError(f"Train IDs do not match between oof[0] and oof[{i-1}]")
        if not np.array_equal(y, df["y"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Targets y do not match between oof[0] and oof[{i-1}]")

    test_ids = subs[0]["id"].to_numpy()
    for i, df in enumerate(subs[1:], start=2):
        if not np.array_equal(test_ids, df["id"].to_numpy()):
            raise ValueError(f"Test IDs do not match between subs[0] and subs[{i-1}]")

    feat_set = [s.strip().lower() for s in str(args.features).split(",") if s.strip()]
    if any(s not in {"prob", "logit"} for s in feat_set):
        raise ValueError("--features must be prob, logit, or prob,logit")

    def build_X(preds: np.ndarray) -> np.ndarray:
        cols = []
        if "prob" in feat_set:
            cols.append(preds)
        if "logit" in feat_set:
            cols.append(_logit(preds))
        return np.hstack(cols) if len(cols) > 1 else cols[0]

    # Shape: (n_models, n_rows)
    P_oof = np.vstack([df["oof_pred"].to_numpy(dtype=np.float64) for df in oofs]).T
    P_test = np.vstack(
        [df["diagnosed_diabetes"].to_numpy(dtype=np.float64) for df in subs]
    ).T

    X = build_X(P_oof)
    X_test = build_X(P_test)

    clf = LogisticRegression(
        C=float(args.C),
        penalty="l2",
        solver="lbfgs",
        max_iter=int(args.max_iter),
        n_jobs=None,
    )
    clf.fit(X, y)

    p_oof = np.clip(clf.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
    ll = log_loss(y, p_oof)
    auc = roc_auc_score(y, p_oof)

    p_test = np.clip(clf.predict_proba(X_test)[:, 1], 0.0, 1.0)

    out = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": p_test})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    if args.verbose:
        coefs = clf.coef_.reshape(-1)
        print(
            f"[stack] models={len(args.oof)} features={feat_set} C={float(args.C)}",
            flush=True,
        )
        print(f"[stack] OOF logloss={ll:.6f} auc={auc:.6f}", flush=True)
        print(
            f"[stack] coef={np.array2string(coefs, precision=4, max_line_width=120)}",
            flush=True,
        )
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
