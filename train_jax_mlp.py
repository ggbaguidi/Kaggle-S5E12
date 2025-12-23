#!/usr/bin/env python3
"""JAX/Flax CPU MLP for tabular probability prediction.

This is mainly useful on synthetic-tabular competitions where a simple NN can
sometimes match the generator better than trees.

Model:
- Categorical columns -> integer codes -> embedding vectors
- Numeric columns -> standardize (train mean/std), NaNs -> 0 after standardization
- Concatenate -> MLP -> sigmoid

Validation:
- Deterministic split by hashing `id` (stable across runs)

Outputs:
- submission CSV (+ optional zip)

Dependencies:
- jax (CPU), flax, optax
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _infer_cols(
    train: pd.DataFrame, id_col: str | None, target_col: str | None
) -> tuple[str, str, list[str]]:
    id_col = id_col or ("id" if "id" in train.columns else train.columns[0])
    target_col = target_col or (
        "diagnosed_diabetes" if "diagnosed_diabetes" in train.columns else None
    )
    if target_col is None or target_col not in train.columns:
        raise ValueError("Could not infer target column; pass --target-col")
    features = [c for c in train.columns if c not in (id_col, target_col)]
    return id_col, target_col, features


def _stable_val_mask(ids: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    # Deterministic split using a stable integer hash.
    # Mix in seed so you can create different (but stable) splits.
    x = ids.astype(np.uint64, copy=False)
    # Use uint64 arithmetic; wraparound is intended.
    x ^= np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    u = (x % np.uint64(10_000)).astype(np.int64)
    return u < int(val_frac * 10_000)


@dataclass
class EncodedData:
    x_num: np.ndarray
    x_cat: np.ndarray
    y: np.ndarray


def _encode_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    max_categories: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    """Return encoded (train_num, train_cat, test_num, test_cat, cat_sizes, num_mean, num_std)."""

    cat_cols: list[str] = []
    num_cols: list[str] = []
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c].dtype):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Categorical encoding with shared categories across train+test.
    cat_sizes: list[int] = []
    train_cat_arrays: list[np.ndarray] = []
    test_cat_arrays: list[np.ndarray] = []

    for c in cat_cols:
        combo = pd.concat([train[c], test[c]], axis=0, ignore_index=True)
        combo = combo.astype("string").fillna("__MISSING__")
        vc = combo.value_counts(dropna=False)
        if len(vc) > max_categories:
            keep = set(vc.nlargest(max_categories).index.astype(str).tolist())
            combo = combo.where(combo.isin(keep), "__OTHER__")
        cats = pd.Categorical(combo)
        codes = cats.codes.astype(np.int32, copy=False)
        n_train = len(train)
        tr_codes = codes[:n_train]
        te_codes = codes[n_train:]
        # pandas uses -1 for NaN; we replaced NaN with string but keep guard:
        tr_codes = np.where(tr_codes < 0, 0, tr_codes)
        te_codes = np.where(te_codes < 0, 0, te_codes)
        # Ensure at least 1 embedding.
        size = int(max(1, len(cats.categories)))
        cat_sizes.append(size)
        train_cat_arrays.append(tr_codes)
        test_cat_arrays.append(te_codes)

    if cat_cols:
        x_cat_train = np.stack(train_cat_arrays, axis=1)
        x_cat_test = np.stack(test_cat_arrays, axis=1)
    else:
        x_cat_train = np.zeros((len(train), 0), dtype=np.int32)
        x_cat_test = np.zeros((len(test), 0), dtype=np.int32)

    # Numeric: standardize using train statistics, then fill NaN with 0.
    if num_cols:
        tr_num = (
            train[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32, copy=False)
        )
        te_num = (
            test[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32, copy=False)
        )
        mean = np.nanmean(tr_num, axis=0).astype(np.float32)
        std = np.nanstd(tr_num, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        tr_num = (tr_num - mean) / std
        te_num = (te_num - mean) / std
        tr_num = np.nan_to_num(tr_num, nan=0.0, posinf=0.0, neginf=0.0)
        te_num = np.nan_to_num(te_num, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        tr_num = np.zeros((len(train), 0), dtype=np.float32)
        te_num = np.zeros((len(test), 0), dtype=np.float32)
        mean = np.zeros((0,), dtype=np.float32)
        std = np.ones((0,), dtype=np.float32)

    return (
        tr_num,
        x_cat_train,
        te_num,
        x_cat_test,
        cat_sizes,
        mean,
        std,
        num_cols,
        cat_cols,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="JAX/Flax CPU MLP tabular trainer")
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("sub/submission_jax.csv"))
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument("--max-categories", type=int, default=2000)

    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden", type=str, default="256,128,64")
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=3)

    p.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional cap for quick experiments",
    )
    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    # Import JAX stack late so the script can still show CLI help without deps.
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax.training import train_state

    rng = np.random.default_rng(int(args.seed))

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    id_col, target_col, features = _infer_cols(
        train, id_col=args.id_col, target_col=args.target_col
    )

    if args.max_train_rows and args.max_train_rows > 0:
        # Keep deterministic subset for repeatability.
        train = train.sample(
            n=min(int(args.max_train_rows), len(train)), random_state=int(args.seed)
        ).reset_index(drop=True)

    ids = (
        pd.to_numeric(train[id_col], errors="coerce")
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )
    val_mask = _stable_val_mask(ids, float(args.val_frac), int(args.seed))

    y = (
        pd.to_numeric(train[target_col], errors="coerce")
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )
    y = np.where(y > 0, 1, 0).astype(np.float32, copy=False)

    tr_num, tr_cat, te_num, te_cat, cat_sizes, num_mean, num_std, num_cols, cat_cols = (
        _encode_features(
            train,
            test,
            features,
            max_categories=int(args.max_categories),
        )
    )

    if args.verbose:
        print(
            f"[jax] rows={len(train)} test_rows={len(test)} features={len(features)} num={len(num_cols)} cat={len(cat_cols)}",
            flush=True,
        )
        if len(cat_cols):
            print(
                f"[jax] cat_sizes(min/median/max)={int(np.min(cat_sizes))}/{int(np.median(cat_sizes))}/{int(np.max(cat_sizes))}",
                flush=True,
            )
        print(
            f"[jax] val_frac={args.val_frac} val_rows={int(val_mask.sum())}", flush=True
        )

    x_num_tr = tr_num[~val_mask]
    x_cat_tr = tr_cat[~val_mask]
    y_tr = y[~val_mask]

    x_num_va = tr_num[val_mask]
    x_cat_va = tr_cat[val_mask]
    y_va = y[val_mask]

    # Simple dataset iterator
    def batches(
        xn: np.ndarray, xc: np.ndarray, yy: np.ndarray, batch_size: int, shuffle: bool
    ):
        n = len(yy)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            j = idx[start : start + batch_size]
            yield xn[j], xc[j], yy[j]

    hidden = [int(x.strip()) for x in str(args.hidden).split(",") if x.strip()]

    class TabMLP(nn.Module):
        cat_sizes: list[int]
        embed_dim: int
        hidden: list[int]
        dropout: float

        @nn.compact
        def __call__(self, x_num, x_cat, train: bool):
            parts = []
            if x_cat.shape[1] > 0:
                for i, size in enumerate(self.cat_sizes):
                    # Slightly scale embedding dim by cardinality but keep bounded.
                    dim = int(self.embed_dim)
                    emb = nn.Embed(
                        num_embeddings=int(max(1, size)), features=dim, name=f"emb_{i}"
                    )
                    parts.append(emb(x_cat[:, i]))
            if x_num.shape[1] > 0:
                parts.append(x_num)
            x = jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

            for h in self.hidden:
                x = nn.Dense(h)(x)
                x = nn.relu(x)
                if self.dropout and self.dropout > 0:
                    x = nn.Dropout(rate=float(self.dropout))(x, deterministic=not train)
            x = nn.Dense(1)(x)
            return x.squeeze(-1)  # logits

    model = TabMLP(
        cat_sizes=cat_sizes,
        embed_dim=int(args.embed_dim),
        hidden=hidden,
        dropout=float(args.dropout),
    )

    def bce_with_logits(logits, labels):
        # stable BCE
        return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    @jax.jit
    def train_step(state, batch, rng_key):
        x_num_b, x_cat_b, y_b = batch
        x_num_b = jnp.asarray(x_num_b)
        x_cat_b = jnp.asarray(x_cat_b)
        y_b = jnp.asarray(y_b)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                x_num_b,
                x_cat_b,
                train=True,
                rngs={"dropout": rng_key},
            )
            return bce_with_logits(logits, y_b)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(params, x_num_b, x_cat_b):
        logits = model.apply(
            {"params": params}, jnp.asarray(x_num_b), jnp.asarray(x_cat_b), train=False
        )
        return jax.nn.sigmoid(logits)

    # Init
    init_key = jax.random.PRNGKey(int(args.seed))
    x_num0 = jnp.zeros((1, x_num_tr.shape[1]), dtype=jnp.float32)
    x_cat0 = jnp.zeros((1, x_cat_tr.shape[1]), dtype=jnp.int32)
    variables = model.init(
        {"params": init_key, "dropout": init_key}, x_num0, x_cat0, train=True
    )

    tx = optax.adamw(
        learning_rate=float(args.lr), weight_decay=float(args.weight_decay)
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx
    )

    best_ll = float("inf")
    bad = 0

    def logloss(y_true: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

    for epoch in range(1, int(args.epochs) + 1):
        losses = []
        for i, (xn, xc, yy) in enumerate(
            batches(x_num_tr, x_cat_tr, y_tr, int(args.batch_size), shuffle=True),
            start=1,
        ):
            drop_key = jax.random.fold_in(init_key, epoch * 100_000 + i)
            state, loss = train_step(state, (xn, xc, yy), drop_key)
            losses.append(float(loss))

        # Val
        preds = []
        for xn, xc, _yy in batches(
            x_num_va, x_cat_va, y_va, int(args.batch_size), shuffle=False
        ):
            p_batch = np.asarray(eval_step(state.params, xn, xc))
            preds.append(p_batch)
        p_va = (
            np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float32)
        )
        ll = logloss(y_va, p_va) if len(p_va) else float("nan")

        if args.verbose:
            tr_loss = float(np.mean(losses)) if losses else float("nan")
            print(
                f"[jax] epoch={epoch} train_loss={tr_loss:.6f} val_logloss={ll:.6f}",
                flush=True,
            )

        if ll < best_ll:
            best_ll = ll
            best_params = state.params
            bad = 0
        else:
            bad += 1
            if bad >= int(args.early_stop):
                if args.verbose:
                    print(
                        f"[jax] early_stop after {epoch} epochs (best_val_logloss={best_ll:.6f})",
                        flush=True,
                    )
                break

    # Predict test
    test_preds = []
    for start in range(0, len(test), int(args.batch_size)):
        xn = te_num[start : start + int(args.batch_size)]
        xc = te_cat[start : start + int(args.batch_size)]
        p_batch = np.asarray(eval_step(best_params, xn, xc))
        test_preds.append(p_batch)
    p_test = np.concatenate(test_preds, axis=0)

    sub = pd.DataFrame(
        {
            "id": pd.to_numeric(test[id_col], errors="coerce")
            .fillna(0)
            .astype(np.int64),
            "diagnosed_diabetes": np.clip(p_test, 0.0, 1.0),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    if args.verbose:
        print(f"[done] wrote {args.out} rows={len(sub)}", flush=True)

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
