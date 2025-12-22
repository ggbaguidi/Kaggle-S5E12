#!/usr/bin/env python3
"""JAX distillation (student) wrapper.

This is a thin wrapper around `train_jax_fttransformer.py` that keeps the same
spirit as `distill_student.py` but runs fully in JAX.

You can also just call `train_jax_fttransformer.py` directly with:
  --teacher-oof ... --soft-alpha ... --label-smoothing ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from train_jax_fttransformer import main as train_main

    p = argparse.ArgumentParser(description="JAX distillation student (wrapper around train_jax_fttransformer.py)")
    p.add_argument("--teacher-oof", type=Path, required=True, help="Teacher OOF CSV (id,y,oof_pred)")
    p.add_argument("--soft-alpha", type=float, default=0.7)
    p.add_argument("--label-smoothing", type=float, default=0.02)

    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("sub/submission_student_jax.csv"))
    p.add_argument("--oof-out", type=Path, default=None)

    # Model/training knobs (pass-through with reasonable defaults)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--norm", type=str, default="derf")

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=3)

    p.add_argument("--max-categories", type=int, default=2000)
    p.add_argument("--max-train-rows", type=int, default=0)

    p.add_argument("--train-weights", type=Path, default=None)

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    fwd = [
        "--train",
        str(args.train),
        "--test",
        str(args.test),
        "--out",
        str(args.out),
        "--folds",
        str(args.folds),
        "--seeds",
        str(args.seeds),
        "--norm",
        str(args.norm),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--n-layers",
        str(args.n_layers),
        "--ff-mult",
        str(args.ff_mult),
        "--dropout",
        str(args.dropout),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--early-stop",
        str(args.early_stop),
        "--max-categories",
        str(args.max_categories),
        "--max-train-rows",
        str(args.max_train_rows),
        "--teacher-oof",
        str(args.teacher_oof),
        "--soft-alpha",
        str(args.soft_alpha),
        "--label-smoothing",
        str(args.label_smoothing),
        "--verbose",
        str(args.verbose),
    ]

    if args.oof_out is not None:
        fwd += ["--oof-out", str(args.oof_out)]
    if args.train_weights is not None:
        fwd += ["--train-weights", str(args.train_weights)]
    if args.zip_output:
        fwd += ["--zip-output"]

    return int(train_main(fwd))


if __name__ == "__main__":
    raise SystemExit(main())
