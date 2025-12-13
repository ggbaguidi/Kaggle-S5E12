#!/usr/bin/env python3
"""Teacher→student distillation for tabular probability submissions.

Workflow:
1) Train a strong teacher with CV and export OOF predictions:
   python train_lgbm.py --oof-out sub/teacher_oof.csv --out sub/teacher_sub.csv

2) Train a student to mimic teacher probabilities (optionally blended with true labels):
   python distill_student.py --teacher-oof sub/teacher_oof.csv --out sub/student_sub.csv

The student is trained with CV LightGBM *regression* to fit soft targets in [0,1].
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


def distill_cv_and_predict(
    *,
    train_path: Path,
    test_path: Path,
    teacher_oof_path: Path,
    out_path: Path,
    id_col: str | None,
    target_col: str | None,
    train_weights_path: Path | None,
    folds: int,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    lambda_l2: float,
    max_categories: int,
    soft_alpha: float,
    label_smoothing: float,
    zip_output: bool,
    verbose: int,
) -> None:
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    id_col, target_col, features = _infer_cols(train, id_col=id_col, target_col=target_col)

    teacher = pd.read_csv(teacher_oof_path)
    if "oof_pred" not in teacher.columns:
        raise ValueError("teacher oof file must contain column: oof_pred")
    if "id" not in teacher.columns:
        raise ValueError("teacher oof file must contain column: id")

    train_id = pd.to_numeric(train[id_col], errors="coerce").fillna(0).astype(np.int64)
    teacher = teacher[["id", "oof_pred"]].copy()
    teacher["id"] = pd.to_numeric(teacher["id"], errors="coerce").fillna(0).astype(np.int64)
    teacher["oof_pred"] = pd.to_numeric(teacher["oof_pred"], errors="coerce").clip(0, 1).fillna(0.5).astype(np.float32)

    merged = pd.DataFrame({"id": train_id}).merge(teacher, on="id", how="left")
    if merged["oof_pred"].isna().any():
        missing = int(merged["oof_pred"].isna().sum())
        raise ValueError(f"teacher oof missing for {missing} train ids")

    y_hard = pd.to_numeric(train[target_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    y_hard = np.where(y_hard > 0, 1, 0).astype(np.int64, copy=False)
    t = merged["oof_pred"].to_numpy(dtype=np.float32, copy=False)

    a = float(soft_alpha)
    a = 0.0 if a < 0 else (1.0 if a > 1 else a)
    y_soft = (1.0 - a) * y_hard.astype(np.float32) + a * t

    eps = float(label_smoothing)
    eps = 0.0 if eps < 0 else (1.0 if eps > 1 else eps)
    if eps:
        y_soft = (1.0 - eps) * y_soft + eps * 0.5

    train_weights: np.ndarray | None = None
    if train_weights_path is not None:
        wdf = pd.read_csv(train_weights_path)
        if id_col not in wdf.columns or "weight" not in wdf.columns:
            raise ValueError(f"train weights file must have columns: {id_col}, weight")
        wdf = wdf[[id_col, "weight"]].copy()
        wdf[id_col] = pd.to_numeric(wdf[id_col], errors="coerce").fillna(0).astype(np.int64)
        wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(1.0).astype(np.float32)
        wmerged = pd.DataFrame({"id": train_id}).merge(wdf, on="id", how="left")
        if wmerged["weight"].isna().any():
            missing = int(wmerged["weight"].isna().sum())
            raise ValueError(f"train weights missing for {missing} train ids")
        train_weights = wmerged["weight"].to_numpy(dtype=np.float32, copy=False)

    train, test, categorical_cols = _prep_frames(train, test, features, max_categories=max_categories)

    x = train[features]
    x_test = test[features]

    skf = StratifiedKFold(n_splits=max(2, int(folds)), shuffle=True, random_state=int(seed))

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
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

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    if verbose:
        print(f"[distill] rows={len(train)} test_rows={len(test)} features={len(features)} cats={len(categorical_cols)}", flush=True)
        print(f"[distill] folds={folds} soft_alpha={a} label_smoothing={eps}", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y_hard), start=1):
        x_tr = x.iloc[tr_idx]
        x_va = x.iloc[va_idx]
        y_tr = y_soft[tr_idx]
        y_va_soft = y_soft[va_idx]

        w_tr = train_weights[tr_idx] if train_weights is not None else None

        dtrain = lgb.Dataset(x_tr, label=y_tr, weight=w_tr, categorical_feature=categorical_cols, free_raw_data=True)
        dvalid = lgb.Dataset(x_va, label=y_va_soft, categorical_feature=categorical_cols, reference=dtrain, free_raw_data=True)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            valid_sets=[dvalid],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=bool(verbose))],
        )

        p_va = booster.predict(x_va, num_iteration=booster.best_iteration)
        p_va = np.clip(p_va, 0.0, 1.0)
        oof[va_idx] = p_va

        p_te = booster.predict(x_test, num_iteration=booster.best_iteration)
        test_pred += np.clip(p_te, 0.0, 1.0) / max(2, int(folds))

        ll = log_loss(y_hard[va_idx], np.clip(p_va, 1e-15, 1 - 1e-15))
        auc = roc_auc_score(y_hard[va_idx], p_va)
        if verbose:
            print(f"[distill] fold={fold} best_iter={booster.best_iteration} logloss={ll:.6f} auc={auc:.6f}", flush=True)

    oof_ll = log_loss(y_hard, np.clip(oof, 1e-15, 1 - 1e-15))
    oof_auc = roc_auc_score(y_hard, oof)
    if verbose:
        print(f"[distill] OOF logloss={oof_ll:.6f} auc={oof_auc:.6f}", flush=True)

    sub = pd.DataFrame(
        {
            "id": pd.to_numeric(test[id_col], errors="coerce").fillna(0).astype(np.int64),
            "diagnosed_diabetes": np.clip(test_pred, 0.0, 1.0),
        }
    )
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Teacher→student distillation (LightGBM regression student)")
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--teacher-oof", type=Path, required=True, help="CSV with columns: id,oof_pred")
    p.add_argument("--out", type=Path, default=Path("sub/submission_student.csv"))
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)

    p.add_argument("--train-weights", type=Path, default=None, help="Optional CSV with columns: id, weight")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-boost-round", type=int, default=4000)
    p.add_argument("--early-stopping-rounds", type=int, default=200)

    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--min-data-in-leaf", type=int, default=40)

    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)

    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--max-categories", type=int, default=2000)

    p.add_argument("--soft-alpha", type=float, default=1.0, help="Blend factor: 0=hard labels, 1=teacher only")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Mix targets towards 0.5 (0..1)")

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    distill_cv_and_predict(
        train_path=args.train,
        test_path=args.test,
        teacher_oof_path=args.teacher_oof,
        out_path=args.out,
        id_col=args.id_col,
        target_col=args.target_col,
        train_weights_path=args.train_weights,
        folds=int(args.folds),
        seed=int(args.seed),
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_data_in_leaf=int(args.min_data_in_leaf),
        feature_fraction=float(args.feature_fraction),
        bagging_fraction=float(args.bagging_fraction),
        bagging_freq=int(args.bagging_freq),
        lambda_l2=float(args.lambda_l2),
        max_categories=int(args.max_categories),
        soft_alpha=float(args.soft_alpha),
        label_smoothing=float(args.label_smoothing),
        zip_output=bool(args.zip_output),
        verbose=int(args.verbose),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
