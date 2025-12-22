from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

from .data import (
    fit_encoders,
    infer_feature_types,
    infer_target_and_id,
    load_dataframes,
    transform_df,
)
from .model import ModelConfig, TabularEmbedMLP
from .adv_weights import AdvConfig, compute_adv_weights, compute_adv_weights_sklearn
from .dist_features import (
    DistFeatureConfig,
    fit_llr_estimator,
    fit_shift_estimator,
    transform_llr_features,
    transform_shift_features,
)
from .train_utils import iter_minibatches


class TrainState(train_state.TrainState):
    dropout_rng: jax.Array


def _default_monotone_spec() -> dict[str, int]:
    # sign meaning: +1 => increasing feature should (weakly) increase diabetes logit
    #              -1 => increasing feature should (weakly) decrease diabetes logit
    # This is a pragmatic prior for this dataset; override via --mono.
    return {
        "age": +1,
        "bmi": +1,
        "waist_to_hip_ratio": +1,
        "systolic_bp": +1,
        "diastolic_bp": +1,
        "cholesterol_total": +1,
        "ldl_cholesterol": +1,
        "triglycerides": +1,
        "screen_time_hours_per_day": +1,
        "physical_activity_minutes_per_week": -1,
        "diet_score": -1,
        "sleep_hours_per_day": 0,
        "hdl_cholesterol": -1,
    }


def _parse_mono_arg(mono: str) -> dict[str, int]:
    # Format: col:+1,col:-1
    out: dict[str, int] = {}
    mono = mono.strip()
    if not mono:
        return out
    for chunk in mono.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid --mono entry '{chunk}'. Expected format like age:+1")
        name, sign = chunk.split(":", 1)
        name = name.strip()
        sign_i = int(sign.strip())
        if sign_i not in (-1, 0, 1):
            raise ValueError(f"Invalid mono sign for '{name}': {sign_i} (allowed: -1,0,+1)")
        out[name] = sign_i
    return out


def bce_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return optax.sigmoid_binary_cross_entropy(logits, y)


def make_state(
    *,
    model: TabularEmbedMLP,
    params: dict,
    dropout_rng: jax.Array,
    lr: float,
    weight_decay: float,
) -> TrainState:
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, dropout_rng=dropout_rng)


def predict_logits_in_batches(
    model: TabularEmbedMLP,
    params: dict,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    @jax.jit
    def pred_step(xc, xn):
        return model.apply({"params": params}, xc, xn, train=False)

    out = np.zeros((x_cat.shape[0],), dtype=np.float32)
    for start in range(0, x_cat.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        logits = pred_step(jnp.asarray(x_cat[sl]), jnp.asarray(x_num[sl]))
        out[sl] = np.asarray(logits, dtype=np.float32)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="data/train.csv")
    p.add_argument("--test", type=str, default="data/test.csv")
    p.add_argument("--out", type=str, default="sub/submission_jax_scratch_adv.csv")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--embed-dim", type=int, default=16)

    p.add_argument("--patience", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=200)

    p.add_argument("--use-adv-weights", action="store_true")
    p.add_argument(
        "--adv-kind",
        type=str,
        default="sklearn",
        choices=["sklearn", "jax"],
        help="Backend used to learn train-vs-test weights. 'sklearn' is fastest on CPU.",
    )
    p.add_argument("--adv-epochs", type=int, default=3)
    p.add_argument("--adv-max-rows", type=int, default=250000)
    p.add_argument("--adv-clip-min", type=float, default=0.2)
    p.add_argument("--adv-clip-max", type=float, default=5.0)
    p.add_argument("--norm-kind", type=str, default="derf")

    p.add_argument(
        "--low-card-int-as-cat-threshold",
        type=int,
        default=32,
        help="Treat int/bool columns with <= this many unique values as categorical.",
    )

    p.add_argument(
        "--use-mono",
        action="store_true",
        help="Enable monotonicity constraint penalty on selected numeric columns.",
    )
    p.add_argument(
        "--mono",
        type=str,
        default="",
        help="Monotonic spec: 'col:+1,col:-1'. Overrides defaults when provided.",
    )
    p.add_argument(
        "--mono-lambda",
        type=float,
        default=0.05,
        help="Weight of monotonic penalty term.",
    )
    p.add_argument(
        "--mono-delta",
        type=float,
        default=0.5,
        help="Perturbation size (in standardized units) for monotonic penalty.",
    )
    p.add_argument(
        "--mono-k",
        type=int,
        default=4,
        help="How many monotone features to sample per step (keeps CPU cost bounded).",
    )

    p.add_argument(
        "--add-dist-features",
        action="store_true",
        help="Add distribution-derived features (cross-fitted) based on per-feature likelihood ratios.",
    )
    p.add_argument("--dist-bins", type=int, default=128)
    p.add_argument("--dist-alpha", type=float, default=0.5)
    p.add_argument("--dist-quantile-max-rows", type=int, default=200000)
    p.add_argument(
        "--dist-per-feature-llr",
        action="store_true",
        help="Include one LLR feature per original column.",
    )
    p.add_argument(
        "--dist-nb-logit",
        action="store_true",
        help="Include a single Naive-Bayes-style summed LLR feature.",
    )
    p.add_argument(
        "--dist-shift",
        action="store_true",
        help="Include per-feature log p(test)/p(train) shift features (unsupervised; uses test covariates).",
    )

    p.add_argument("--smoke", action="store_true")

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    train_df, test_df = load_dataframes(args.train, args.test)
    target_col, id_col = infer_target_and_id(train_df)

    spec = infer_feature_types(
        train_df,
        test_df,
        target_col=target_col,
        id_col=id_col,
        low_card_int_as_cat_threshold=int(args.low_card_int_as_cat_threshold),
    )
    enc = fit_encoders(train_df, test_df, spec)

    y = train_df[target_col].to_numpy(dtype=np.float32)
    ids_test = test_df[id_col].to_numpy() if id_col in test_df.columns else np.arange(len(test_df))

    x_cat_train, x_num_train = transform_df(train_df, spec, enc)
    x_cat_test, x_num_test = transform_df(test_df, spec, enc)

    # Monotonic constraints only apply to numeric columns.
    mono_map = _parse_mono_arg(args.mono) if args.mono else _default_monotone_spec()
    mono_idx = []
    mono_sign = []
    if args.use_mono and spec.num_cols:
        num_index = {c: i for i, c in enumerate(spec.num_cols)}
        for col, sgn in mono_map.items():
            if sgn == 0:
                continue
            if col not in num_index:
                continue
            mono_idx.append(num_index[col])
            mono_sign.append(sgn)

    mono_idx_arr = jnp.asarray(np.array(mono_idx, dtype=np.int32)) if mono_idx else None
    mono_sign_arr = jnp.asarray(np.array(mono_sign, dtype=np.float32)) if mono_sign else None
    mono_k = int(min(int(args.mono_k), len(mono_idx))) if mono_idx else 0

    if args.use_mono:
        kept_cols = []
        if mono_idx:
            inv_num_cols = list(spec.num_cols)
            kept_cols = [(inv_num_cols[i], int(s)) for i, s in zip(mono_idx, mono_sign)]
        print(f"monotone: enabled cols={kept_cols} lambda={args.mono_lambda} delta={args.mono_delta} k={mono_k}")

    mono_lambda = float(args.mono_lambda)
    mono_delta = float(args.mono_delta)

    # optional smoke mode: smaller training
    if args.smoke:
        max_train = min(len(train_df), 120000)
        idx = rng.choice(np.arange(len(train_df)), size=max_train, replace=False)
        x_cat_train = x_cat_train[idx]
        x_num_train = x_num_train[idx]
        y = y[idx]

        max_test = min(len(test_df), 120000)
        idx_t = rng.choice(np.arange(len(test_df)), size=max_test, replace=False)
        x_cat_test = x_cat_test[idx_t]
        x_num_test = x_num_test[idx_t]
        ids_test = ids_test[idx_t]

    cat_cards = [spec.cat_cardinalities[c] for c in spec.cat_cols]

    if args.add_dist_features and (not args.dist_per_feature_llr) and (not args.dist_nb_logit):
        # sensible default: include both when user enables dist features
        args.dist_per_feature_llr = True
        args.dist_nb_logit = True

    dist_cfg = DistFeatureConfig(
        num_bins=int(args.dist_bins),
        alpha=float(args.dist_alpha),
        quantile_max_rows=int(args.dist_quantile_max_rows),
        add_per_feature_llr=bool(args.dist_per_feature_llr),
        add_nb_logit=bool(args.dist_nb_logit),
        add_shift=bool(args.dist_shift),
    )

    # Optional unsupervised shift features computed once.
    shift_train = None
    shift_test = None
    if args.add_dist_features and dist_cfg.add_shift:
        print("dist: fitting shift estimator (train vs test)")
        sh = fit_shift_estimator(
            x_cat_train,
            x_num_train,
            x_cat_test,
            x_num_test,
            cat_cardinalities=cat_cards,
            cfg=dist_cfg,
            seed=args.seed,
        )
        shift_train = transform_shift_features(sh, x_cat_train, x_num_train, cfg=dist_cfg)
        shift_test = transform_shift_features(sh, x_cat_test, x_num_test, cfg=dist_cfg)
        print(f"dist: shift features train={shift_train.shape} test={shift_test.shape}")

    # Adversarial weights (computed once on full train)
    w_train = None
    if args.use_adv_weights:
        print(f"adv: learning weights kind={args.adv_kind} max_rows={args.adv_max_rows}")
        adv_cfg = AdvConfig(
            epochs=args.adv_epochs,
            batch_size=args.batch_size,
            max_rows=args.adv_max_rows,
            clip_min=args.adv_clip_min,
            clip_max=args.adv_clip_max,
            seed=args.seed,
        )
        if args.adv_kind == "sklearn":
            w_train = compute_adv_weights_sklearn(
                x_cat_train,
                x_num_train,
                x_cat_test,
                x_num_test,
                adv_cfg=adv_cfg,
            )
        else:
            w_train = compute_adv_weights(
                x_cat_train,
                x_num_train,
                x_cat_test,
                x_num_test,
                cat_cardinalities=cat_cards,
                adv_cfg=adv_cfg,
            )
        print("adv: weights ready")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = np.zeros((len(y),), dtype=np.float32)
    test_preds = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros_like(y), y)):
        print(f"fold={fold}: init/compile (first fold can take a while on CPU)")
        xct, xnt, yt = x_cat_train[tr], x_num_train[tr], y[tr]
        xcv, xnv, yv = x_cat_train[va], x_num_train[va], y[va]
        wt = None if w_train is None else w_train[tr]

        x_num_test_fold = x_num_test

        # Cross-fitted supervised distribution features (Naive-Bayes style): fit on fold-train only.
        if args.add_dist_features:
            print(f"fold={fold}: dist fit (LLR)")
            est = fit_llr_estimator(
                xct,
                xnt,
                yt,
                cat_cardinalities=cat_cards,
                cfg=dist_cfg,
                seed=args.seed + fold,
            )
            llr_tr, nb_tr = transform_llr_features(est, xct, xnt, cfg=dist_cfg)
            llr_va, nb_va = transform_llr_features(est, xcv, xnv, cfg=dist_cfg)
            llr_te, nb_te = transform_llr_features(est, x_cat_test, x_num_test, cfg=dist_cfg)

            extra_tr = []
            extra_va = []
            extra_te = []

            if dist_cfg.add_per_feature_llr:
                extra_tr.append(llr_tr)
                extra_va.append(llr_va)
                extra_te.append(llr_te)
            if dist_cfg.add_nb_logit:
                extra_tr.append(nb_tr)
                extra_va.append(nb_va)
                extra_te.append(nb_te)
            if dist_cfg.add_shift and shift_train is not None and shift_test is not None:
                extra_tr.append(shift_train[tr])
                extra_va.append(shift_train[va])
                extra_te.append(shift_test)

            if extra_tr:
                xnt = np.concatenate([xnt] + extra_tr, axis=1).astype(np.float32, copy=False)
                xnv = np.concatenate([xnv] + extra_va, axis=1).astype(np.float32, copy=False)
                x_num_test_fold = np.concatenate([x_num_test] + extra_te, axis=1).astype(
                    np.float32, copy=False
                )

        cfg = ModelConfig(
            cat_cardinalities=cat_cards,
            num_features=xnt.shape[1],
            width=args.width,
            blocks=args.blocks,
            dropout=args.dropout,
            embed_dim=args.embed_dim,
            norm_kind=args.norm_kind,
        )

        jax_rng = jax.random.PRNGKey(args.seed + fold)
        model = TabularEmbedMLP(cfg)

        params_rng, dropout_rng = jax.random.split(jax_rng)
        variables = model.init(
            {"params": params_rng, "dropout": dropout_rng},
            jnp.zeros((2, xct.shape[1]), dtype=jnp.int32),
            jnp.zeros((2, xnt.shape[1]), dtype=jnp.float32),
            train=True,
        )
        state = make_state(
            model=model,
            params=variables["params"],
            dropout_rng=dropout_rng,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        @jax.jit
        def train_step(state: TrainState, xc, xn, yb, wb):
            dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

            def loss_fn(params):
                logits_train = state.apply_fn(
                    {"params": params},
                    xc,
                    xn,
                    train=True,
                    rngs={"dropout": dropout_rng},
                )
                loss_vec = bce_logits(logits_train, yb)
                if wb is not None:
                    loss_vec = loss_vec * wb

                loss = jnp.mean(loss_vec)

                # PINN-like constraint: monotonicity penalty on selected numeric features.
                if mono_k > 0 and mono_idx_arr is not None and mono_sign_arr is not None:
                    # use deterministic logits for constraint to reduce dropout noise
                    logits_base = state.apply_fn({"params": params}, xc, xn, train=False)
                    mono_rng, _ = jax.random.split(new_dropout_rng)
                    sel = jax.random.choice(
                        mono_rng,
                        jnp.arange(mono_idx_arr.shape[0]),
                        shape=(mono_k,),
                        replace=False,
                    )
                    f_idx = mono_idx_arr[sel]
                    f_sign = mono_sign_arr[sel]

                    def one_pen(idx, sign):
                        xn_pert = xn.at[:, idx].add(jnp.asarray(mono_delta, dtype=xn.dtype))
                        logits_pert = state.apply_fn({"params": params}, xc, xn_pert, train=False)
                        margin = sign * (logits_pert - logits_base)
                        return jnp.mean(nn.relu(-margin))

                    mono_pen = jnp.mean(jax.vmap(one_pen)(f_idx, f_sign))
                    loss = loss + jnp.asarray(mono_lambda, dtype=loss.dtype) * mono_pen

                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(dropout_rng=new_dropout_rng)
            return state, loss

        best_auc = -1.0
        best_params = None
        bad = 0
        step = 0

        for epoch in range(int(args.epochs)):
            for batch in iter_minibatches(
                xct,
                xnt,
                yt,
                wt,
                batch_size=args.batch_size,
                rng=rng,
                shuffle=True,
            ):
                wb = None if batch.w is None else jnp.asarray(batch.w)
                state, _ = train_step(
                    state,
                    jnp.asarray(batch.x_cat),
                    jnp.asarray(batch.x_num),
                    jnp.asarray(batch.y),
                    wb,
                )
                step += 1

                if step % int(args.eval_every) == 0:
                    val_logits = predict_logits_in_batches(
                        model,
                        state.params,
                        xcv,
                        xnv,
                        batch_size=args.batch_size,
                    )
                    val_prob = 1.0 / (1.0 + np.exp(-val_logits))
                    auc = float(roc_auc_score(yv, val_prob))

                    if auc > best_auc + 1e-5:
                        best_auc = auc
                        best_params = jax.tree_util.tree_map(lambda x: x, state.params)
                        bad = 0
                    else:
                        bad += 1

                    if bad >= int(args.patience):
                        break

            if bad >= int(args.patience):
                break

        if best_params is None:
            best_params = state.params

        val_logits = predict_logits_in_batches(
            model, best_params, xcv, xnv, batch_size=args.batch_size
        )
        oof[va] = 1.0 / (1.0 + np.exp(-val_logits))

        test_logits = predict_logits_in_batches(
            model, best_params, x_cat_test, x_num_test_fold, batch_size=args.batch_size
        )
        test_preds.append(1.0 / (1.0 + np.exp(-test_logits)))

        fold_auc = float(roc_auc_score(yv, oof[va]))
        print(f"fold={fold} auc={fold_auc:.6f} best_auc={best_auc:.6f}")

    full_auc = float(roc_auc_score(y, oof))
    print(f"OOF AUC: {full_auc:.6f}")

    pred = np.mean(np.stack(test_preds, axis=0), axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({id_col: ids_test, target_col: pred.astype(np.float32)})
    sub.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
