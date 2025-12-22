#!/usr/bin/env python3
"""Adversarial reweighting in JAX/Flax (CPU).

Trains a classifier to distinguish train vs test rows (domain classifier), then
uses importance weights on train rows:
  w = p(test|x) / (1 - p(test|x))

Outputs:
- CSV with columns: id, weight

Notes:
- Uses K-fold cross-fitting on the combined (train+test) rows to reduce leakage.
- Uses the same preprocessing as JAX tabular trainers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jax_tabular_preproc import fit_preprocessor


def _make_stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("folds must be >= 2")
    rng = np.random.default_rng(int(seed))

    folds = np.empty(len(y), dtype=np.int32)
    folds.fill(-1)
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[ii] = j % n_splits
    if (folds < 0).any():
        idx = np.arange(len(y))
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[ii] = j % n_splits
    return folds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="JAX adversarial reweighting (train-vs-test)")
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("sub/train_weights_adv_jax.csv"))

    p.add_argument("--id-col", type=str, default="id")
    p.add_argument("--target-col", type=str, default="diagnosed_diabetes", help="Ignored (only used to drop from features)")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-categories", type=int, default=2000)
    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--max-test-rows", type=int, default=0)

    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden", type=str, default="256,128")
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=2)

    p.add_argument("--clip-min", type=float, default=0.2)
    p.add_argument("--clip-max", type=float, default=5.0)
    p.add_argument("--normalize", action="store_true", help="Normalize weights to mean=1")

    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax.training import train_state

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    id_col = str(args.id_col)
    target_col = str(args.target_col)
    if id_col not in train_df.columns:
        raise ValueError(f"id-col {id_col!r} not found in train")
    if id_col not in test_df.columns:
        raise ValueError(f"id-col {id_col!r} not found in test")

    # Features = all columns except id and target (if present)
    feature_cols = [c for c in train_df.columns if c not in {id_col, target_col}]

    if args.max_train_rows and args.max_train_rows > 0:
        train_df = train_df.sample(n=min(int(args.max_train_rows), len(train_df)), random_state=int(args.seed)).reset_index(drop=True)
    if args.max_test_rows and args.max_test_rows > 0:
        test_df = test_df.sample(n=min(int(args.max_test_rows), len(test_df)), random_state=int(args.seed)).reset_index(drop=True)

    pre = fit_preprocessor(
        train_df,
        test_df,
        id_col=id_col,
        target_col=None,
        feature_cols=feature_cols,
        max_categories=int(args.max_categories),
    )

    train_ids = pre.transform_ids(train_df)
    test_ids = pre.transform_ids(test_df)

    x_num_tr, x_cat_tr = pre.transform_X(train_df)
    x_num_te, x_cat_te = pre.transform_X(test_df)

    # Domain labels: train=0, test=1
    x_num = np.concatenate([x_num_tr, x_num_te], axis=0)
    x_cat = np.concatenate([x_cat_tr, x_cat_te], axis=0)
    d = np.concatenate([
        np.zeros(len(train_df), dtype=np.float32),
        np.ones(len(test_df), dtype=np.float32),
    ])

    fold_id = _make_stratified_folds(d.astype(np.int64), int(args.folds), int(args.seed))

    hidden = [int(s.strip()) for s in str(args.hidden).split(",") if s.strip()]
    if not hidden:
        raise ValueError("--hidden must be non-empty")

    class DomainMLP(nn.Module):
        cat_sizes: list[int]
        embed_dim: int
        hidden: list[int]
        dropout: float

        @nn.compact
        def __call__(self, x_num, x_cat, train: bool):
            parts = []
            if x_cat.shape[1] > 0:
                for i, size in enumerate(self.cat_sizes):
                    emb = nn.Embed(num_embeddings=int(max(2, size)), features=int(self.embed_dim), name=f"emb_{i}")
                    parts.append(emb(x_cat[:, i]))
            if x_num.shape[1] > 0:
                parts.append(x_num)
            x = jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

            for h in self.hidden:
                x = nn.Dense(int(h))(x)
                x = nn.relu(x)
                if self.dropout and self.dropout > 0:
                    x = nn.Dropout(rate=float(self.dropout))(x, deterministic=not train)
            x = nn.Dense(1)(x).squeeze(-1)
            return x  # logits

    model = DomainMLP(cat_sizes=pre.cat_sizes, embed_dim=int(args.embed_dim), hidden=hidden, dropout=float(args.dropout))

    def create_state(rng_key):
        params = model.init(
            {"params": rng_key, "dropout": rng_key},
            jnp.zeros((1, x_num.shape[1]), dtype=jnp.float32),
            jnp.zeros((1, x_cat.shape[1]), dtype=jnp.int32),
            train=True,
        )["params"]
        tx = optax.adamw(learning_rate=float(args.lr), weight_decay=float(args.weight_decay))
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, xnb, xcb, yb, rng_key):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, xnb, xcb, train=True, rngs={"dropout": rng_key})
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, yb))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def pred_step(params, xnb, xcb):
        logits = model.apply({"params": params}, xnb, xcb, train=False)
        return jax.nn.sigmoid(logits)

    rng = np.random.default_rng(int(args.seed))
    key = jax.random.PRNGKey(int(args.seed))

    p_oof = np.zeros(len(d), dtype=np.float64)

    bs = int(args.batch_size)

    for f in range(int(args.folds)):
        tr_idx = np.where(fold_id != f)[0]
        va_idx = np.where(fold_id == f)[0]

        state = create_state(jax.random.fold_in(key, f))

        best_loss = float("inf")
        best_params = state.params
        bad = 0

        for epoch in range(int(args.epochs)):
            perm = np.array(tr_idx, copy=True)
            rng.shuffle(perm)
            n_batches = max(1, len(perm) // bs)
            for bi in range(n_batches):
                j = perm[bi * bs : (bi + 1) * bs]
                xnb = jnp.asarray(x_num[j], dtype=jnp.float32)
                xcb = jnp.asarray(x_cat[j], dtype=jnp.int32)
                yb = jnp.asarray(d[j], dtype=jnp.float32)
                key = jax.random.fold_in(key, epoch * 10_000 + bi)
                state, _ = train_step(state, xnb, xcb, yb, key)

            # quick val loss
            preds = []
            for start in range(0, len(va_idx), bs):
                sl = va_idx[start : start + bs]
                pb = pred_step(
                    state.params,
                    jnp.asarray(x_num[sl], dtype=jnp.float32),
                    jnp.asarray(x_cat[sl], dtype=jnp.int32),
                )
                preds.append(np.asarray(pb))
            pv = np.concatenate(preds, axis=0).astype(np.float64)
            pv = np.clip(pv, 1e-6, 1 - 1e-6)
            yv = d[va_idx].astype(np.float64)
            ll = float((-(yv * np.log(pv) + (1 - yv) * np.log1p(-pv))).mean())

            if args.verbose >= 2:
                print(f"[adv fold={f}] epoch={epoch+1} val_logloss={ll:.6f}", flush=True)

            if ll + 1e-6 < best_loss:
                best_loss = ll
                best_params = state.params
                bad = 0
            else:
                bad += 1
                if bad >= int(args.early_stop):
                    break

        # OOF preds for this fold
        preds = []
        for start in range(0, len(va_idx), bs):
            sl = va_idx[start : start + bs]
            pb = pred_step(
                best_params,
                jnp.asarray(x_num[sl], dtype=jnp.float32),
                jnp.asarray(x_cat[sl], dtype=jnp.int32),
            )
            preds.append(np.asarray(pb))
        pv = np.concatenate(preds, axis=0).astype(np.float64)
        p_oof[va_idx] = pv

        if args.verbose:
            print(f"[adv] fold={f} best_val_logloss={best_loss:.6f}", flush=True)

    # Use train portion only for weights.
    p_train = np.clip(p_oof[: len(train_df)], 1e-6, 1 - 1e-6)
    w = p_train / (1.0 - p_train)

    w = np.clip(w, float(args.clip_min), float(args.clip_max)).astype(np.float64)
    if args.normalize:
        m = float(np.mean(w))
        if m > 0:
            w = w / m

    out = pd.DataFrame({"id": train_ids, "weight": w.astype(np.float32)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    if args.verbose:
        print(f"[adv] wrote {args.out} rows={len(out)} w(min/mean/max)={float(np.min(w)):.4f}/{float(np.mean(w)):.4f}/{float(np.max(w)):.4f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
