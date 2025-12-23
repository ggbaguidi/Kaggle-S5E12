#!/usr/bin/env python3
"""CatBoost CPU trainer with stratified CV (strong tabular baseline).

- Handles categorical columns automatically (object/string/category).
- StratifiedKFold CV + early stopping.
- Optional multi-seed ensembling via --seeds.
- Writes Kaggle submission + optional zip.

Requires: catboost (already installed in this venv).
"""

from __future__ import annotations

import argparse
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


def _cat_features(df: pd.DataFrame, features: list[str]) -> list[int]:
    cat_idx: list[int] = []
    for i, c in enumerate(features):
        if not pd.api.types.is_numeric_dtype(df[c].dtype):
            cat_idx.append(i)
    return cat_idx


def _parse_seeds(seeds: str | None, seed: int) -> list[int]:
    if seeds is None or not str(seeds).strip():
        return [int(seed)]
    return [int(x.strip()) for x in str(seeds).split(",") if x.strip()]


def train_cv_and_predict(
    *,
    train_path: Path,
    test_path: Path,
    out_path: Path,
    id_col: str | None,
    target_col: str | None,
    oof_out: Path | None,
    extra_train_path: Path | None,
    extra_target_col: str | None,
    extra_weight: float,
    folds: int,
    seed: int,
    seeds: str | None,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_strength: float,
    bagging_temperature: float,
    border_count: int,
    od_wait: int,
    thread_count: int,
    zip_output: bool,
    verbose: int,
) -> None:
    from catboost import CatBoostClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    extra_train: pd.DataFrame | None = None
    extra_y: np.ndarray | None = None
    if extra_train_path is not None:
        extra_train = pd.read_csv(extra_train_path)

    id_col, target_col, features = _infer_cols(
        train, id_col=id_col, target_col=target_col
    )

    y = (
        pd.to_numeric(train[target_col], errors="coerce")
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )
    y = np.where(y > 0, 1, 0).astype(np.int64, copy=False)

    if extra_train is not None:
        et = extra_target_col or target_col
        if et not in extra_train.columns:
            raise ValueError(
                f"extra target column '{et}' not found in {extra_train_path}"
            )
        missing = [c for c in features if c not in extra_train.columns]
        if missing:
            raise ValueError(
                f"extra_train missing feature columns: {missing[:10]}{'...' if len(missing)>10 else ''}"
            )
        extra_y = (
            pd.to_numeric(extra_train[et], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        extra_y = np.where(extra_y > 0, 1, 0).astype(np.int64, copy=False)
        extra_train = extra_train[features + [et]].copy()

    x = train[features].copy()
    x_test = test[features].copy()
    x_extra = extra_train[features].copy() if extra_train is not None else None

    # Keep NaNs; CatBoost handles missing. Ensure categoricals are strings.
    for c in features:
        if not pd.api.types.is_numeric_dtype(x[c].dtype):
            x[c] = x[c].astype("string").fillna("__MISSING__")
            x_test[c] = x_test[c].astype("string").fillna("__MISSING__")
            if x_extra is not None and c in x_extra.columns:
                x_extra[c] = x_extra[c].astype("string").fillna("__MISSING__")

    cat_idx = _cat_features(x, features)
    seed_list = _parse_seeds(seeds, seed)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_ens = np.zeros(len(train), dtype=np.float64)
    test_pred_ens = np.zeros(len(test), dtype=np.float64)

    if verbose:
        print(
            f"[cat] rows={len(train)} test_rows={len(test)} features={len(features)} cat_features={len(cat_idx)}",
            flush=True,
        )
        if x_extra is not None:
            print(
                f"[cat] extra_rows={len(x_extra)} extra_weight={extra_weight}",
                flush=True,
            )
        print(
            f"[cat] folds={folds} seeds={seed_list} iterations={iterations} od_wait={od_wait} threads={thread_count}",
            flush=True,
        )

    for s in seed_list:
        oof = np.zeros(len(train), dtype=np.float64)
        test_pred = np.zeros(len(test), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
            x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
            x_va, y_va = x.iloc[va_idx], y[va_idx]

            if x_extra is not None and extra_y is not None:
                x_tr = pd.concat([x_tr, x_extra], axis=0, ignore_index=True)
                y_tr = np.concatenate([y_tr, extra_y])
                w_tr = np.concatenate(
                    [
                        np.ones(len(tr_idx), dtype=np.float32),
                        np.full(len(extra_y), float(extra_weight), dtype=np.float32),
                    ]
                )
            else:
                w_tr = None

            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                bagging_temperature=bagging_temperature,
                border_count=border_count,
                od_type="Iter",
                od_wait=od_wait,
                random_seed=int(s),
                task_type="CPU",
                thread_count=thread_count,
                allow_writing_files=False,
                verbose=(100 if verbose else False),
            )

            model.fit(
                x_tr,
                y_tr,
                cat_features=cat_idx,
                eval_set=(x_va, y_va),
                use_best_model=True,
                sample_weight=w_tr,
            )

            p_va = model.predict_proba(x_va)[:, 1]
            p_va = np.clip(p_va, 1e-15, 1 - 1e-15)
            oof[va_idx] = p_va

            p_te = model.predict_proba(x_test)[:, 1]
            test_pred += p_te / folds

            ll = log_loss(y_va, p_va)
            auc = roc_auc_score(y_va, p_va)
            if verbose:
                print(
                    f"[cat] seed={s} fold={fold} logloss={ll:.6f} auc={auc:.6f}",
                    flush=True,
                )

        oof_ll = log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15))
        oof_auc = roc_auc_score(y, oof)
        if verbose:
            print(
                f"[cat] seed={s} OOF logloss={oof_ll:.6f} auc={oof_auc:.6f}", flush=True
            )

        oof_ens += oof / len(seed_list)
        test_pred_ens += test_pred / len(seed_list)

    oof_ll = log_loss(y, np.clip(oof_ens, 1e-15, 1 - 1e-15))
    oof_auc = roc_auc_score(y, oof_ens)
    if verbose:
        print(f"[cat] OOF logloss={oof_ll:.6f} auc={oof_auc:.6f}", flush=True)

    sub = pd.DataFrame(
        {
            "id": pd.to_numeric(test[id_col], errors="coerce")
            .fillna(0)
            .astype(np.int64),
            "diagnosed_diabetes": np.clip(test_pred_ens, 0.0, 1.0),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if verbose:
        print(f"[done] wrote {out_path} rows={len(sub)}", flush=True)

    if zip_output:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(out_path, arcname=out_path.name)
        if verbose:
            print(f"[done] wrote {zip_path}", flush=True)

    if oof_out is not None:
        oof_df = pd.DataFrame(
            {
                "id": pd.to_numeric(train[id_col], errors="coerce")
                .fillna(0)
                .astype(np.int64),
                "y": y.astype(np.int64, copy=False),
                "oof_pred": np.clip(oof_ens, 0.0, 1.0).astype(np.float32),
            }
        )
        oof_out.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(oof_out, index=False)
        if verbose:
            print(f"[done] wrote {oof_out} rows={len(oof_df)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CatBoost CPU CV trainer for diabetes probability"
    )
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("submission_cat.csv"))
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)

    p.add_argument(
        "--oof-out",
        type=Path,
        default=None,
        help="Optional CSV output for OOF preds: id,y,oof_pred",
    )

    p.add_argument(
        "--extra-train",
        type=Path,
        default=None,
        help="Optional extra training CSV (e.g., original dataset)",
    )
    p.add_argument(
        "--extra-target-col",
        type=str,
        default=None,
        help="Target column name in extra train (default: same as --target-col)",
    )
    p.add_argument(
        "--extra-weight",
        type=float,
        default=0.3,
        help="Downweight extra rows vs competition rows",
    )

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default=None)

    p.add_argument("--iterations", type=int, default=8000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2-leaf-reg", type=float, default=6.0)
    p.add_argument("--random-strength", type=float, default=1.0)
    p.add_argument("--bagging-temperature", type=float, default=0.8)
    p.add_argument("--border-count", type=int, default=128)
    p.add_argument("--od-wait", type=int, default=300)
    p.add_argument("--thread-count", type=int, default=-1)

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    train_cv_and_predict(
        train_path=args.train,
        test_path=args.test,
        out_path=args.out,
        id_col=args.id_col,
        target_col=args.target_col,
        oof_out=args.oof_out,
        extra_train_path=args.extra_train,
        extra_target_col=args.extra_target_col,
        extra_weight=float(args.extra_weight),
        folds=max(2, int(args.folds)),
        seed=int(args.seed),
        seeds=args.seeds,
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_strength=float(args.random_strength),
        bagging_temperature=float(args.bagging_temperature),
        border_count=int(args.border_count),
        od_wait=int(args.od_wait),
        thread_count=int(args.thread_count),
        zip_output=bool(args.zip_output),
        verbose=int(args.verbose),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
