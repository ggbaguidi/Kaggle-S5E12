#!/usr/bin/env python3
"""LightGBM CPU trainer with stratified CV for Kaggle submissions.

Why this helps vs streaming SGD:
- GBDT models (LightGBM) usually perform much better on tabular data.
- Stratified K-fold + early stopping gives a robust score estimate and reduces overfit.

Assumptions:
- `data/train.csv` contains target column `diagnosed_diabetes` and id column `id`.
- `data/test.csv` contains id column `id`.

Outputs `submission.csv` and optionally `submission.csv.zip`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd
import os


def _infer_cols(train: pd.DataFrame, id_col: str | None, target_col: str | None) -> tuple[str, str, list[str]]:
    id_col = id_col or ("id" if "id" in train.columns else train.columns[0])
    target_col = target_col or ("diagnosed_diabetes" if "diagnosed_diabetes" in train.columns else None)
    if target_col is None or target_col not in train.columns:
        raise ValueError("Could not infer target column; pass --target-col")
    features = [c for c in train.columns if c not in (id_col, target_col)]
    return id_col, target_col, features


def _prep_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    max_categories: int,
    extra: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    # LightGBM can use pandas 'category' dtype as categorical features.
    categorical_cols: list[str] = []
    for c in features:
        if not pd.api.types.is_numeric_dtype(train[c].dtype):
            categorical_cols.append(c)

    for c in categorical_cols:
        # Stabilize categories across train+test
        parts = [train[c], test[c]]
        if extra is not None and c in extra.columns:
            parts.append(extra[c])
        combo = pd.concat(parts, axis=0, ignore_index=True)
        combo = combo.astype("string").fillna("__MISSING__")
        # Reduce huge cardinalities to an '__OTHER__' bucket
        vc = combo.value_counts(dropna=False)
        if len(vc) > max_categories:
            keep = set(vc.nlargest(max_categories).index.astype(str).tolist())
            combo = combo.where(combo.isin(keep), "__OTHER__")

        cats = pd.Categorical(combo)
        n_train = len(train)
        n_test = len(test)
        n_extra = len(extra) if extra is not None and c in extra.columns else 0

        train[c] = cats[:n_train].astype("category")
        test[c] = cats[n_train : n_train + n_test].astype("category")
        if n_extra:
            start = n_train + n_test
            extra[c] = cats[start : start + n_extra].astype("category")

    # Numeric: coerce and fill missing
    for c in features:
        if c in categorical_cols:
            continue
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")
        if extra is not None and c in extra.columns:
            extra[c] = pd.to_numeric(extra[c], errors="coerce")
        # Keep NaNs: LightGBM handles missing values natively.

    return train, test, extra, categorical_cols


def _parse_seeds(seeds: str | None, seed: int) -> list[int]:
    if seeds is None or not str(seeds).strip():
        return [int(seed)]
    parts = [p.strip() for p in str(seeds).split(",") if p.strip()]
    return [int(p) for p in parts]


def train_cv_and_predict(
    *,
    train_path: Path,
    test_path: Path,
    out_path: Path,
    id_col: str | None,
    target_col: str | None,
    train_weights_path: Path | None,
    oof_out: Path | None,
    extra_train_path: Path | None,
    extra_target_col: str | None,
    extra_weight: float,
    folds: int,
    seed: int,
    seeds: str | None,
    num_boost_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    min_data_in_leaf: int,
    min_sum_hessian_in_leaf: float,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    lambda_l1: float,
    lambda_l2: float,
    cat_smooth: float,
    cat_l2: float,
    max_cat_to_onehot: int,
    min_data_per_group: int,
    max_categories: int,
    zip_output: bool,
    model_out: Path | None,
    verbose: int,
) -> None:
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_weights: np.ndarray | None = None

    extra_train: pd.DataFrame | None = None
    extra_y: np.ndarray | None = None
    if extra_train_path is not None:
        extra_train = pd.read_csv(extra_train_path)

    id_col, target_col, features = _infer_cols(train, id_col=id_col, target_col=target_col)

    if train_weights_path is not None:
        wdf = pd.read_csv(train_weights_path)
        if id_col not in wdf.columns or "weight" not in wdf.columns:
            raise ValueError(f"train weights file must have columns: {id_col}, weight")
        # Align by id to avoid relying on row order.
        tmp = pd.DataFrame({"__id": pd.to_numeric(train[id_col], errors="coerce").fillna(0).astype(np.int64)})
        wdf = wdf[[id_col, "weight"]].copy()
        wdf[id_col] = pd.to_numeric(wdf[id_col], errors="coerce").fillna(0).astype(np.int64)
        wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(1.0).astype(np.float32)
        merged = tmp.merge(wdf, left_on="__id", right_on=id_col, how="left")
        if merged["weight"].isna().any():
            missing = int(merged["weight"].isna().sum())
            raise ValueError(f"train weights missing for {missing} train ids")
        train_weights = merged["weight"].to_numpy(dtype=np.float32, copy=False)

    y = pd.to_numeric(train[target_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    y = np.where(y > 0, 1, 0).astype(np.int64, copy=False)

    if extra_train is not None:
        et = extra_target_col or target_col
        if et not in extra_train.columns:
            raise ValueError(f"extra target column '{et}' not found in {extra_train_path}")
        extra_y = pd.to_numeric(extra_train[et], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        extra_y = np.where(extra_y > 0, 1, 0).astype(np.int64, copy=False)
        # Align feature set by intersection; keep competition feature order.
        missing = [c for c in features if c not in extra_train.columns]
        if missing:
            raise ValueError(f"extra_train missing feature columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
        extra_train = extra_train[features + [et]].copy()

    train, test, extra_train, categorical_cols = _prep_frames(
        train, test, features, max_categories=max_categories, extra=extra_train
    )

    x = train[features]
    x_test = test[features]
    x_extra = extra_train[features] if extra_train is not None else None

    seed_list = _parse_seeds(seeds, seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_ens = np.zeros(len(train), dtype=np.float64)
    test_pred_ens = np.zeros(len(test), dtype=np.float64)

    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
        "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "cat_smooth": cat_smooth,
        "cat_l2": cat_l2,
        "max_cat_to_onehot": max_cat_to_onehot,
        "min_data_per_group": min_data_per_group,
        "verbosity": -1,
        "feature_pre_filter": False,
        "force_col_wise": True,
    }

    if verbose:
        print(f"[lgbm] rows={len(train)} test_rows={len(test)} features={len(features)} cats={len(categorical_cols)}", flush=True)
        if extra_train is not None:
            print(f"[lgbm] extra_rows={len(extra_train)} extra_weight={extra_weight}", flush=True)
        print(f"[lgbm] folds={folds} seeds={seed_list} num_boost_round={num_boost_round} early_stopping={early_stopping_rounds}", flush=True)

    models: list[dict[str, object]] = []

    for s_i, s in enumerate(seed_list, start=1):
        params = dict(base_params)
        params["seed"] = int(s)

        oof = np.zeros(len(train), dtype=np.float64)
        test_pred = np.zeros(len(test), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
            x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
            x_va, y_va = x.iloc[va_idx], y[va_idx]

            w_base = train_weights[tr_idx] if train_weights is not None else None
            if x_extra is not None and extra_y is not None:
                x_tr = pd.concat([x_tr, x_extra], axis=0, ignore_index=True)
                y_tr = np.concatenate([y_tr, extra_y])
                if w_base is None:
                    w_base = np.ones(len(tr_idx), dtype=np.float32)
                w_tr = np.concatenate(
                    [
                        w_base.astype(np.float32, copy=False),
                        np.full(len(extra_y), float(extra_weight), dtype=np.float32),
                    ]
                )
            else:
                w_tr = w_base
                if w_tr is not None:
                    w_tr = w_tr.astype(np.float32, copy=False)

            dtrain = lgb.Dataset(x_tr, label=y_tr, weight=w_tr, categorical_feature=categorical_cols, free_raw_data=True)
            dvalid = lgb.Dataset(x_va, label=y_va, categorical_feature=categorical_cols, reference=dtrain, free_raw_data=True)

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=bool(verbose))],
            )

            p_va = booster.predict(x_va, num_iteration=booster.best_iteration)
            p_va = np.clip(p_va, 1e-15, 1 - 1e-15)
            oof[va_idx] = p_va

            p_te = booster.predict(x_test, num_iteration=booster.best_iteration)
            test_pred += p_te / folds

            fold_ll = log_loss(y_va, p_va)
            fold_auc = roc_auc_score(y_va, p_va)
            if verbose:
                print(f"[lgbm] seed={s} fold={fold} best_iter={booster.best_iteration} logloss={fold_ll:.6f} auc={fold_auc:.6f}", flush=True)

            if model_out is not None:
                models.append(
                    {
                        "seed": int(s),
                        "fold": fold,
                        "best_iteration": int(booster.best_iteration),
                        "model_str": booster.model_to_string(),
                    }
                )

        oof_ll = log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15))
        oof_auc = roc_auc_score(y, oof)
        if verbose:
            print(f"[lgbm] seed={s} OOF logloss={oof_ll:.6f} auc={oof_auc:.6f}", flush=True)

        oof_ens += oof / len(seed_list)
        test_pred_ens += test_pred / len(seed_list)

    oof_ll = log_loss(y, np.clip(oof_ens, 1e-15, 1 - 1e-15))
    oof_auc = roc_auc_score(y, oof_ens)
    if verbose:
        print(f"[lgbm] OOF logloss={oof_ll:.6f} auc={oof_auc:.6f}", flush=True)

    sub = pd.DataFrame({"id": pd.to_numeric(test[id_col], errors="coerce").fillna(0).astype(np.int64), "diagnosed_diabetes": np.clip(test_pred_ens, 0.0, 1.0)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if verbose:
        print(f"[done] wrote {out_path} rows={len(sub)}", flush=True)

    if zip_output:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        if verbose:
            print(f"[done] wrote {zip_path}", flush=True)

    if model_out is not None:
        payload = {"params": base_params, "seeds": seed_list, "oof_logloss": float(oof_ll), "oof_auc": float(oof_auc), "models": models}
        model_out.write_text(json.dumps(payload)[:50_000_000])
        if verbose:
            print(f"[done] wrote {model_out}", flush=True)

    if oof_out is not None:
        oof_df = pd.DataFrame(
            {
                "id": pd.to_numeric(train[id_col], errors="coerce").fillna(0).astype(np.int64),
                "y": y.astype(np.int64, copy=False),
                "oof_pred": np.clip(oof_ens, 0.0, 1.0).astype(np.float32),
            }
        )
        oof_out.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(oof_out, index=False)
        if verbose:
            print(f"[done] wrote {oof_out} rows={len(oof_df)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LightGBM CPU CV trainer for diabetes probability")
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)

    p.add_argument("--train-weights", type=Path, default=None, help="Optional CSV with columns: id, weight (aligned by id)")
    p.add_argument("--oof-out", type=Path, default=None, help="Optional CSV output for OOF preds: id,y,oof_pred")

    p.add_argument("--extra-train", type=Path, default=None, help="Optional extra training CSV (e.g., original dataset)")
    p.add_argument("--extra-target-col", type=str, default=None, help="Target column name in extra train (default: same as --target-col)")
    p.add_argument("--extra-weight", type=float, default=0.3, help="Downweight extra rows vs competition rows")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for ensembling (e.g., 42,43,44)")

    p.add_argument("--num-boost-round", type=int, default=4000)
    p.add_argument("--early-stopping-rounds", type=int, default=200)

    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=128)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--min-data-in-leaf", type=int, default=53)
    p.add_argument("--min-sum-hessian-in-leaf", type=float, default=0.002357)

    p.add_argument("--feature-fraction", type=float, default=0.95)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)

    p.add_argument("--lambda-l1", type=float, default=0.0)
    p.add_argument("--lambda-l2", type=float, default=0.0)

    p.add_argument("--cat-smooth", type=float, default=17.0)
    p.add_argument("--cat-l2", type=float, default=29.0)
    p.add_argument("--max-cat-to-onehot", type=int, default=7)
    p.add_argument("--min-data-per-group", type=int, default=200)

    p.add_argument("--max-categories", type=int, default=2000, help="Cap unique categories per feature")

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--model-out", type=Path, default=None, help="Optional JSON with params + serialized models (large)")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    train_cv_and_predict(
        train_path=args.train,
        test_path=args.test,
        out_path=args.out,
        id_col=args.id_col,
        target_col=args.target_col,
        train_weights_path=args.train_weights,
        oof_out=args.oof_out,
        extra_train_path=args.extra_train,
        extra_target_col=args.extra_target_col,
        extra_weight=float(args.extra_weight),
        folds=max(2, int(args.folds)),
        seed=int(args.seed),
        seeds=args.seeds,
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_data_in_leaf=int(args.min_data_in_leaf),
        min_sum_hessian_in_leaf=float(args.min_sum_hessian_in_leaf),
        feature_fraction=float(args.feature_fraction),
        bagging_fraction=float(args.bagging_fraction),
        bagging_freq=int(args.bagging_freq),
        lambda_l1=float(args.lambda_l1),
        lambda_l2=float(args.lambda_l2),
        cat_smooth=float(args.cat_smooth),
        cat_l2=float(args.cat_l2),
        max_cat_to_onehot=int(args.max_cat_to_onehot),
        min_data_per_group=int(args.min_data_per_group),
        max_categories=int(args.max_categories),
        zip_output=bool(args.zip_output),
        model_out=args.model_out,
        verbose=int(args.verbose),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
