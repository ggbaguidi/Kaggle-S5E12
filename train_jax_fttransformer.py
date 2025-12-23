#!/usr/bin/env python3
"""JAX/Flax FT-Transformer-style tabular trainer (CPU).

Outputs:
- Kaggle submission: columns (id, diagnosed_diabetes)
- Optional OOF: columns (id, y, oof_pred)

Features:
- Shared preprocessing via `jax_tabular_preproc.py`
- Stratified K-fold CV (numpy implementation)
- Multi-seed ensembling
- Optional sample weights (id,weight)
- Optional distillation from teacher OOF (id,oof_pred)
- Norm options: layernorm | derf | none

Derf (from arXiv:2512.10938v1) used as a norm replacement:
  Derf(x) = gamma * erf(alpha * x + s) + beta
where alpha,s are learned scalars; gamma,beta are per-channel.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import zipfile
import json

import numpy as np
import pandas as pd

from jax_tabular_preproc import fit_preprocessor, infer_id_target_features


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logloss_np(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    if w is None:
        return float(loss.mean())
    ww = np.asarray(w, dtype=np.float64)
    ww = np.clip(ww, 0.0, np.inf)
    denom = float(ww.sum())
    if denom <= 0:
        return float(loss.mean())
    return float((loss * ww).sum() / denom)


def _make_stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Return fold_id per row in [0..n_splits-1]."""
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
        # If only one class exists, fall back to random folds.
        idx = np.arange(len(y))
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[ii] = j % n_splits

    return folds


def _read_train_weights(path: Path, train_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    need = {"id", "weight"}
    if set(df.columns) != need:
        raise ValueError(f"Bad columns in {path}: {list(df.columns)} (need {sorted(need)})")
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(np.float32)

    w_map = dict(zip(df["id"].to_numpy(), df["weight"].to_numpy()))
    w = np.asarray([float(w_map.get(int(i), 1.0)) for i in train_ids], dtype=np.float32)
    w = np.clip(w, 0.0, np.inf).astype(np.float32, copy=False)
    return w


def _read_teacher_oof(path: Path, train_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    need = {"id", "y", "oof_pred"}
    if set(df.columns) != need:
        raise ValueError(f"Bad columns in {path}: {list(df.columns)} (need {sorted(need)})")
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    df["oof_pred"] = pd.to_numeric(df["oof_pred"], errors="coerce").fillna(0.5).clip(0, 1).astype(np.float32)

    p_map = dict(zip(df["id"].to_numpy(), df["oof_pred"].to_numpy()))
    p = np.asarray([float(p_map.get(int(i), 0.5)) for i in train_ids], dtype=np.float32)
    p = np.clip(p, 1e-6, 1 - 1e-6).astype(np.float32, copy=False)
    return p


@dataclass
class FoldResult:
    oof_pred: np.ndarray
    test_pred: np.ndarray
    val_logloss: float


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="JAX/Flax FT-Transformer trainer (CPU)")
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("sub/submission_jax_ftt.csv"))
    p.add_argument("--oof-out", type=Path, default=None, help="Optional OOF CSV output (id,y,oof_pred)")

    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")

    p.add_argument("--max-categories", type=int, default=2000)
    p.add_argument("--max-train-rows", type=int, default=0)

    p.add_argument("--norm", type=str, default="derf", help="layernorm|derf|none")
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

    p.add_argument("--grad-clip", type=float, default=1.0, help="Global norm clip; <=0 disables")
    p.add_argument("--lr-schedule", type=str, default="constant", help="constant|cosine")
    p.add_argument("--warmup-epochs", type=float, default=0.0)
    p.add_argument("--min-lr", type=float, default=0.0)

    p.add_argument("--train-weights", type=Path, default=None, help="CSV with columns id,weight")

    p.add_argument("--teacher-oof", type=Path, default=None, help="Teacher OOF CSV (id,y,oof_pred) for distillation")
    p.add_argument("--soft-alpha", type=float, default=0.0, help="Blend factor for teacher: y_soft = (1-a)*y + a*teacher")
    p.add_argument("--label-smoothing", type=float, default=0.0)

    p.add_argument("--zip-output", action="store_true")
    p.add_argument(
        "--save-best-dir",
        type=Path,
        default=None,
        help="Optional directory to save best per-fold params (Flax msgpack) whenever val logloss improves.",
    )
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    # Late imports so CLI works without JAX installed.
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax.training import train_state
    from flax import serialization

    def _print_runtime_info() -> None:
        import os
        import sys
        try:
            import jaxlib  # type: ignore
        except Exception:
            jaxlib = None  # type: ignore

        print(f"[runtime] python={sys.executable}", flush=True)
        try:
            jl_ver = getattr(jaxlib, "__version__", "unknown") if jaxlib is not None else "missing"
            print(f"[runtime] jax={jax.__version__} jaxlib={jl_ver}", flush=True)
        except Exception:
            pass
        if "JAX_PLATFORM_NAME" in os.environ:
            print(f"[runtime] env JAX_PLATFORM_NAME={os.environ.get('JAX_PLATFORM_NAME')}", flush=True)

        try:
            backend = jax.default_backend()
        except Exception:
            backend = "unknown"
        try:
            devs = jax.devices()
            dev_str = ", ".join([f"{d.platform}:{d.device_kind}" for d in devs])
            ndev = len(devs)
        except Exception:
            dev_str = "unknown"
            ndev = 0
        print(f"[runtime] jax_backend={backend} n_devices={ndev} devices={dev_str}", flush=True)

    _print_runtime_info()

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds is empty")

    save_best_dir: Path | None = Path(args.save_best_dir) if args.save_best_dir is not None else None

    def _save_best_params(*, seed: int, fold: int, params: dict, best_val_logloss: float) -> None:
        if save_best_dir is None:
            return
        save_best_dir.mkdir(parents=True, exist_ok=True)

        # Flax serialization is a msgpack byte payload.
        params_bytes = serialization.to_bytes(params)
        stem = f"ftt_seed{seed}_fold{fold}"
        params_path = save_best_dir / f"{stem}.msgpack"
        meta_path = save_best_dir / f"{stem}.json"

        params_path.write_bytes(params_bytes)
        meta = {
            "seed": int(seed),
            "fold": int(fold),
            "best_val_logloss": float(best_val_logloss),
            "train": str(args.train),
            "test": str(args.test),
            "folds": int(args.folds),
            "seeds": str(args.seeds),
            "norm": str(args.norm),
            "d_model": int(args.d_model),
            "n_heads": int(args.n_heads),
            "n_layers": int(args.n_layers),
            "ff_mult": int(args.ff_mult),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "early_stop": int(args.early_stop),
            "max_categories": int(args.max_categories),
            "max_train_rows": int(args.max_train_rows),
            "train_weights": str(args.train_weights) if args.train_weights is not None else None,
            "teacher_oof": str(args.teacher_oof) if args.teacher_oof is not None else None,
            "soft_alpha": float(args.soft_alpha),
            "label_smoothing": float(args.label_smoothing),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def _try_load_best_params(*, seed: int, fold: int, target_params: dict) -> tuple[dict | None, float | None]:
        """Return (params, best_val_logloss) if a checkpoint exists; else (None, None)."""
        if save_best_dir is None:
            return None, None
        stem = f"ftt_seed{seed}_fold{fold}"
        params_path = save_best_dir / f"{stem}.msgpack"
        meta_path = save_best_dir / f"{stem}.json"
        if not params_path.exists():
            return None, None

        try:
            loaded = serialization.from_bytes(target_params, params_path.read_bytes())
        except Exception as e:
            if args.verbose:
                print(f"[warn] failed to load checkpoint {params_path}: {e}", flush=True)
            return None, None

        best_ll = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if isinstance(meta, dict) and "best_val_logloss" in meta:
                    best_ll = float(meta["best_val_logloss"])
            except Exception:
                best_ll = None

        if args.verbose:
            msg = f"[ckpt] loaded {params_path}"
            if best_ll is not None:
                msg += f" (best_val_logloss={best_ll:.6f})"
            print(msg, flush=True)

        return loaded, best_ll

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    id_col, target_col, feature_cols = infer_id_target_features(train_df, id_col=args.id_col, target_col=args.target_col)

    if args.max_train_rows and args.max_train_rows > 0:
        train_df = train_df.sample(n=min(int(args.max_train_rows), len(train_df)), random_state=int(seeds[0])).reset_index(drop=True)

    pre = fit_preprocessor(
        train_df,
        test_df,
        id_col=id_col,
        target_col=target_col,
        feature_cols=feature_cols,
        max_categories=int(args.max_categories),
    )

    train_ids = pre.transform_ids(train_df)
    test_ids = pre.transform_ids(test_df)
    y_hard = pre.transform_y(train_df)

    x_num_tr, x_cat_tr = pre.transform_X(train_df)
    x_num_te, x_cat_te = pre.transform_X(test_df)

    if args.train_weights is not None:
        base_w = _read_train_weights(Path(args.train_weights), train_ids)
    else:
        base_w = np.ones(len(train_df), dtype=np.float32)

    # Distillation soft targets
    soft_alpha = float(args.soft_alpha)
    if args.teacher_oof is not None:
        teacher = _read_teacher_oof(Path(args.teacher_oof), train_ids)
        y_soft = (1.0 - soft_alpha) * y_hard + soft_alpha * teacher
        y_soft = np.clip(y_soft, 1e-6, 1 - 1e-6).astype(np.float32)
    else:
        y_soft = y_hard

    ls = float(args.label_smoothing)
    if ls > 0:
        y_soft = (1.0 - ls) * y_soft + 0.5 * ls

    folds = int(args.folds)
    fold_id = _make_stratified_folds(y_hard.astype(np.int64), folds, seed=int(seeds[0]))

    d_model = int(args.d_model)
    n_heads = int(args.n_heads)
    n_layers = int(args.n_layers)
    ff_mult = int(args.ff_mult)
    dropout = float(args.dropout)

    if d_model % n_heads != 0:
        raise ValueError("d-model must be divisible by n-heads")

    norm_kind = str(args.norm).strip().lower()
    if norm_kind not in {"layernorm", "derf", "none"}:
        raise ValueError("--norm must be layernorm|derf|none")

    class Derf(nn.Module):
        alpha_init: float = 0.5
        s_init: float = 0.0

        @nn.compact
        def __call__(self, x):
            d = x.shape[-1]
            gamma = self.param("gamma", lambda k: jnp.ones((d,), dtype=jnp.float32))
            beta = self.param("beta", lambda k: jnp.zeros((d,), dtype=jnp.float32))
            alpha = self.param("alpha", lambda k: jnp.array(self.alpha_init, dtype=jnp.float32))
            s = self.param("s", lambda k: jnp.array(self.s_init, dtype=jnp.float32))
            return gamma * jax.lax.erf(alpha * x + s) + beta

    def make_norm():
        if norm_kind == "layernorm":
            return nn.LayerNorm(use_bias=True, use_scale=True)
        if norm_kind == "derf":
            return Derf()
        return None

    class NumTokenEmbed(nn.Module):
        n_num: int
        d_model: int

        @nn.compact
        def __call__(self, x_num):
            # x_num: (B, n_num)
            w = self.param("w", nn.initializers.lecun_normal(), (self.n_num, self.d_model))
            b = self.param("b", nn.initializers.zeros, (self.n_num, self.d_model))
            return x_num[:, :, None] * w[None, :, :] + b[None, :, :]

    class TransformerBlock(nn.Module):
        d_model: int
        n_heads: int
        ff_mult: int
        dropout: float

        @nn.compact
        def __call__(self, x, *, train: bool):
            norm1 = make_norm()
            y = x
            if norm1 is not None:
                y = norm1(y)

            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
                dropout_rate=self.dropout,
                deterministic=not train,
            )(y, y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
            x = x + y

            norm2 = make_norm()
            y = x
            if norm2 is not None:
                y = norm2(y)

            y = nn.Dense(self.ff_mult * self.d_model)(y)
            y = nn.gelu(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
            y = nn.Dense(self.d_model)(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic=not train)
            x = x + y
            return x

    class FTTransformer(nn.Module):
        cat_sizes: list[int]
        n_num: int
        d_model: int
        n_heads: int
        n_layers: int
        ff_mult: int
        dropout: float

        @nn.compact
        def __call__(self, x_num, x_cat, *, train: bool):
            # Tokens
            tokens = []

            # CLS token
            cls = self.param("cls", nn.initializers.normal(0.02), (self.d_model,))
            cls_tok = jnp.broadcast_to(cls[None, None, :], (x_num.shape[0], 1, self.d_model))
            tokens.append(cls_tok)

            if x_cat.shape[1] > 0:
                cat_toks = []
                for i, size in enumerate(self.cat_sizes):
                    emb = nn.Embed(num_embeddings=int(max(2, size)), features=self.d_model, name=f"emb_{i}")
                    cat_toks.append(emb(x_cat[:, i]))
                cat_toks = jnp.stack(cat_toks, axis=1)
                tokens.append(cat_toks)

            if self.n_num > 0:
                tokens.append(NumTokenEmbed(n_num=self.n_num, d_model=self.d_model, name="numtok")(x_num))

            x = jnp.concatenate(tokens, axis=1)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

            for i in range(self.n_layers):
                x = TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    ff_mult=self.ff_mult,
                    dropout=self.dropout,
                    name=f"blk_{i}",
                )(x, train=train)

            normf = make_norm()
            if normf is not None:
                x = normf(x)

            cls_out = x[:, 0, :]
            logit = nn.Dense(1)(cls_out).squeeze(-1)
            return logit

    model = FTTransformer(
        cat_sizes=pre.cat_sizes,
        n_num=int(x_num_tr.shape[1]),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
    )

    def create_state(rng_key, *, steps_per_epoch: int, epochs: int):
        params = model.init(
            {"params": rng_key, "dropout": rng_key},
            jnp.zeros((1, x_num_tr.shape[1]), dtype=jnp.float32),
            jnp.zeros((1, x_cat_tr.shape[1]), dtype=jnp.int32),
            train=True,
        )["params"]
        lr0 = float(args.lr)
        min_lr = float(args.min_lr)
        schedule_kind = str(args.lr_schedule).strip().lower()
        if schedule_kind not in {"constant", "cosine"}:
            raise ValueError("--lr-schedule must be constant|cosine")

        if schedule_kind == "constant":
            lr_schedule = lr0
        else:
            total_steps = int(max(1, int(steps_per_epoch) * int(epochs)))
            warmup_steps = int(max(0, round(float(args.warmup_epochs) * int(steps_per_epoch))))
            warmup_steps = int(min(warmup_steps, max(0, total_steps - 1)))
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=min_lr,
                peak_value=lr0,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=min_lr,
            )

        transforms = []
        gc = float(args.grad_clip)
        if gc and gc > 0:
            transforms.append(optax.clip_by_global_norm(gc))
        transforms.append(optax.adamw(learning_rate=lr_schedule, weight_decay=float(args.weight_decay)))
        tx = optax.chain(*transforms)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def bce_with_logits(logits, y, w=None):
        y = y.astype(jnp.float32)
        # optax.sigmoid_binary_cross_entropy expects logits and labels
        per = optax.sigmoid_binary_cross_entropy(logits, y)
        if w is not None:
            per = per * w.astype(jnp.float32)
            denom = jnp.maximum(1e-6, jnp.sum(w))
            return jnp.sum(per) / denom
        return jnp.mean(per)

    @jax.jit
    def train_step(state, batch, rng_key):
        x_num_b, x_cat_b, y_b, w_b = batch

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                x_num_b,
                x_cat_b,
                train=True,
                rngs={"dropout": rng_key},
            )
            loss = bce_with_logits(logits, y_b, w_b)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def pred_step(params, x_num_b, x_cat_b):
        logits = model.apply({"params": params}, x_num_b, x_cat_b, train=False)
        return jax.nn.sigmoid(logits)

    def run_one_seed(seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
        rng = np.random.default_rng(int(seed))
        key = jax.random.PRNGKey(int(seed))

        oof = np.zeros(len(train_df), dtype=np.float64)
        test_pred = np.zeros(len(test_df), dtype=np.float64)
        fold_losses: list[float] = []

        for f in range(folds):
            tr_idx = np.where(fold_id != f)[0]
            va_idx = np.where(fold_id == f)[0]

            xnum_tr_f = x_num_tr[tr_idx]
            xcat_tr_f = x_cat_tr[tr_idx]
            y_tr_f = y_soft[tr_idx]
            w_tr_f = base_w[tr_idx]

            xnum_va_f = x_num_tr[va_idx]
            xcat_va_f = x_cat_tr[va_idx]
            y_va_f = y_hard[va_idx]
            w_va_f = base_w[va_idx]

            bs = int(args.batch_size)
            n_train = int(len(tr_idx))
            steps_per_epoch = int(max(1, (n_train + bs - 1) // bs))
            state = create_state(jax.random.fold_in(key, f), steps_per_epoch=steps_per_epoch, epochs=int(args.epochs))

            schedule_kind = str(args.lr_schedule).strip().lower()
            if schedule_kind == "constant":
                lr_fn = None
            else:
                total_steps = int(max(1, steps_per_epoch * int(args.epochs)))
                warmup_steps = int(max(0, round(float(args.warmup_epochs) * steps_per_epoch)))
                warmup_steps = int(min(warmup_steps, max(0, total_steps - 1)))
                lr_fn = optax.warmup_cosine_decay_schedule(
                    init_value=float(args.min_lr),
                    peak_value=float(args.lr),
                    warmup_steps=warmup_steps,
                    decay_steps=total_steps,
                    end_value=float(args.min_lr),
                )

            # Resume from previous best checkpoint (if present).
            loaded_params, loaded_best = _try_load_best_params(seed=seed, fold=f, target_params=state.params)
            if loaded_params is not None:
                state = state.replace(params=loaded_params)

            best_loss = float("inf") if loaded_best is None else float(loaded_best)
            best_params = state.params
            bad = 0

            for epoch in range(int(args.epochs)):
                # shuffle train indices
                perm = np.arange(len(tr_idx))
                rng.shuffle(perm)

                # train
                tr_loss_sum = 0.0
                tr_w_sum = 0.0
                for bi in range(steps_per_epoch):
                    j = perm[bi * bs : (bi + 1) * bs]
                    xb_num = jnp.asarray(xnum_tr_f[j], dtype=jnp.float32)
                    xb_cat = jnp.asarray(xcat_tr_f[j], dtype=jnp.int32)
                    yb = jnp.asarray(y_tr_f[j], dtype=jnp.float32)
                    wb = jnp.asarray(w_tr_f[j], dtype=jnp.float32)
                    key = jax.random.fold_in(key, epoch * 10_000 + bi)
                    state, loss_b = train_step(state, (xb_num, xb_cat, yb, wb), key)
                    loss_b_f = float(loss_b)
                    w_sum_b = float(np.asarray(w_tr_f[j], dtype=np.float64).sum())
                    tr_loss_sum += loss_b_f * w_sum_b
                    tr_w_sum += w_sum_b

                tr_loss = tr_loss_sum / max(1e-12, tr_w_sum)

                # eval
                preds = []
                for start in range(0, len(va_idx), bs):
                    sl = slice(start, min(len(va_idx), start + bs))
                    p_b = pred_step(
                        state.params,
                        jnp.asarray(xnum_va_f[sl], dtype=jnp.float32),
                        jnp.asarray(xcat_va_f[sl], dtype=jnp.int32),
                    )
                    preds.append(np.asarray(p_b))
                p_va = np.concatenate(preds, axis=0).astype(np.float64)
                ll = _logloss_np(y_va_f, p_va, w_va_f)

                if args.verbose >= 2:
                    if lr_fn is None:
                        lr_now = float(args.lr)
                    else:
                        lr_now = float(lr_fn(int(epoch) * int(steps_per_epoch)))
                    print(
                        f"[seed={seed} fold={f}] epoch={epoch+1} train_bce={tr_loss:.6f} val_logloss={ll:.6f} lr={lr_now:.6g}",
                        flush=True,
                    )

                if ll + 1e-6 < best_loss:
                    best_loss = ll
                    best_params = state.params
                    _save_best_params(seed=seed, fold=f, params=best_params, best_val_logloss=best_loss)
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(args.early_stop):
                        break

            # store oof
            preds = []
            for start in range(0, len(va_idx), bs):
                sl = slice(start, min(len(va_idx), start + bs))
                p_b = pred_step(
                    best_params,
                    jnp.asarray(xnum_va_f[sl], dtype=jnp.float32),
                    jnp.asarray(xcat_va_f[sl], dtype=jnp.int32),
                )
                preds.append(np.asarray(p_b))
            p_va = np.concatenate(preds, axis=0).astype(np.float64)
            oof[va_idx] = p_va
            fold_losses.append(float(best_loss))

            # predict test
            preds = []
            for start in range(0, len(test_df), bs):
                sl = slice(start, min(len(test_df), start + bs))
                p_b = pred_step(
                    best_params,
                    jnp.asarray(x_num_te[sl], dtype=jnp.float32),
                    jnp.asarray(x_cat_te[sl], dtype=jnp.int32),
                )
                preds.append(np.asarray(p_b))
            p_te = np.concatenate(preds, axis=0).astype(np.float64)
            test_pred += p_te / float(folds)

            if args.verbose:
                print(f"[seed={seed}] fold={f} best_val_logloss={best_loss:.6f}", flush=True)

        # overall metrics on hard labels
        overall_ll = _logloss_np(y_hard, oof, base_w)
        if args.verbose:
            print(f"[seed={seed}] OOF logloss={overall_ll:.6f} fold_mean={float(np.mean(fold_losses)):.6f}", flush=True)

        return oof, test_pred, fold_losses

    # Ensemble over seeds
    oof_ens = np.zeros(len(train_df), dtype=np.float64)
    test_ens = np.zeros(len(test_df), dtype=np.float64)

    for si, seed in enumerate(seeds, start=1):
        oof_s, te_s, _ = run_one_seed(int(seed))
        oof_ens += oof_s / float(len(seeds))
        test_ens += te_s / float(len(seeds))
        if args.verbose:
            ll = _logloss_np(y_hard, oof_s, base_w)
            print(f"[seed {si}/{len(seeds)}] done seed={seed} oof_logloss={ll:.6f}", flush=True)

    final_ll = _logloss_np(y_hard, oof_ens, base_w)
    if args.verbose:
        print(f"[final] ensemble_seeds={len(seeds)} oof_logloss={final_ll:.6f}", flush=True)

    # Write submission
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": np.clip(test_ens, 0.0, 1.0)})
    sub.to_csv(args.out, index=False)
    if args.verbose:
        print(f"[done] wrote {args.out} rows={len(sub)}", flush=True)

    # Write OOF
    if args.oof_out is not None:
        out_oof = Path(args.oof_out)
        out_oof.parent.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame({"id": train_ids, "y": y_hard.astype(np.int64), "oof_pred": np.clip(oof_ens, 0.0, 1.0)})
        oof_df.to_csv(out_oof, index=False)
        if args.verbose:
            print(f"[done] wrote {out_oof} rows={len(oof_df)}", flush=True)

    if args.zip_output:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname=args.out.name)
        if args.verbose:
            print(f"[done] wrote {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
