#!/usr/bin/env python3
"""Adversarial reweighting for synthetic/distribution-shifted tabular competitions.

Trains a classifier to distinguish competition train vs test rows, then converts the
predicted p(test|x) into importance weights for the *train* rows:

    w(x) = p(test|x) / (1 - p(test|x))

Weights are cross-fitted (OOF) to reduce overfitting.

Outputs a CSV with columns: id, weight
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    categorical_cols: list[str] = []
    for c in features:
        if not pd.api.types.is_numeric_dtype(train[c].dtype):
            categorical_cols.append(c)

    for c in categorical_cols:
        combo = pd.concat([train[c], test[c]], axis=0, ignore_index=True)
        combo = combo.astype("string").fillna("__MISSING__")
        vc = combo.value_counts(dropna=False)
        if len(vc) > max_categories:
            keep = set(vc.nlargest(max_categories).index.astype(str).tolist())
            combo = combo.where(combo.isin(keep), "__OTHER__")
        cats = pd.Categorical(combo)
        n_train = len(train)
        n_test = len(test)
        train[c] = cats[:n_train].astype("category")
        test[c] = cats[n_train : n_train + n_test].astype("category")

    for c in features:
        if c in categorical_cols:
            continue
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")

    return train, test, categorical_cols


def adversarial_weights(
    *,
    train_path: Path,
    test_path: Path,
    out_path: Path,
    id_col: str | None,
    target_col: str | None,
    folds: int,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    lambda_l2: float,
    max_categories: int,
    clip_min: float,
    clip_max: float,
    normalize: bool,
    zip_output: bool,
    verbose: int,
) -> None:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    id_col, _target, features = _infer_cols(train, id_col=id_col, target_col=target_col)
    train, test, categorical_cols = _prep_frames(train, test, features, max_categories=max_categories)

    x_train = train[features]
    x_test = test[features]

    x_all = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    d_all = np.concatenate([np.zeros(len(x_train), dtype=np.int64), np.ones(len(x_test), dtype=np.int64)])

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": int(bagging_freq),
        "lambda_l2": float(lambda_l2),
        "verbosity": -1,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "seed": int(seed),
    }

    skf = StratifiedKFold(n_splits=max(2, int(folds)), shuffle=True, random_state=int(seed))
    oof = np.zeros(len(x_all), dtype=np.float64)

    if verbose:
        print(f"[adv] train_rows={len(x_train)} test_rows={len(x_test)} features={len(features)} cats={len(categorical_cols)}", flush=True)
        print(f"[adv] folds={folds} num_boost_round={num_boost_round} early_stopping={early_stopping_rounds}", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_all, d_all), start=1):
        dtrain = lgb.Dataset(
            x_all.iloc[tr_idx],
            label=d_all[tr_idx],
            categorical_feature=categorical_cols,
            free_raw_data=True,
        )
        dvalid = lgb.Dataset(
            x_all.iloc[va_idx],
            label=d_all[va_idx],
            categorical_feature=categorical_cols,
            reference=dtrain,
            free_raw_data=True,
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            valid_sets=[dvalid],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=bool(verbose))],
        )

        p = booster.predict(x_all.iloc[va_idx], num_iteration=booster.best_iteration)
        oof[va_idx] = np.clip(p, 1e-6, 1 - 1e-6)
        if verbose:
            print(f"[adv] fold={fold} best_iter={booster.best_iteration}", flush=True)

    p_train = oof[: len(x_train)]
    w = p_train / (1.0 - p_train)
    w = np.clip(w, float(clip_min), float(clip_max))
    if normalize:
        w = w / (np.mean(w) + 1e-12)

    out = pd.DataFrame(
        {
            "id": pd.to_numeric(train[id_col], errors="coerce").fillna(0).astype(np.int64),
            "weight": w.astype(np.float32),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    if verbose:
        print(f"[done] wrote {out_path} rows={len(out)} weight_mean={float(np.mean(w)):.6f} min={float(np.min(w)):.6f} max={float(np.max(w)):.6f}", flush=True)

    if zip_output:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        if verbose:
            print(f"[done] wrote {zip_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Adversarial reweighting: train-vs-test importance weights")
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("sub/train_weights_adv.csv"))
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None, help="Only used to infer feature columns")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-boost-round", type=int, default=1500)
    p.add_argument("--early-stopping-rounds", type=int, default=100)

    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--min-data-in-leaf", type=int, default=50)

    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)

    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--max-categories", type=int, default=2000)

    p.add_argument("--clip-min", type=float, default=0.1)
    p.add_argument("--clip-max", type=float, default=10.0)
    p.add_argument("--no-normalize", action="store_true", help="Do not normalize weights to mean=1")

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    adversarial_weights(
        train_path=args.train,
        test_path=args.test,
        out_path=args.out,
        id_col=args.id_col,
        target_col=args.target_col,
        folds=int(args.folds),
        seed=int(args.seed),
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_data_in_leaf=int(args.min_data_in_leaf),
        feature_fraction=float(args.feature_fraction),
        bagging_fraction=float(args.bagging_fraction),
        bagging_freq=int(args.bagging_freq),
        lambda_l2=float(args.lambda_l2),
        max_categories=int(args.max_categories),
        clip_min=float(args.clip_min),
        clip_max=float(args.clip_max),
        normalize=not bool(args.no_normalize),
        zip_output=bool(args.zip_output),
        verbose=int(args.verbose),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
