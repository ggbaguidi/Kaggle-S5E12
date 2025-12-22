#!/usr/bin/env python3
"""CPU-friendly RandomForest / GradientBoosting trainer for Kaggle S5E12.

Models:
- RandomForestClassifier (rf)
- HistGradientBoostingClassifier (hgb)

Outputs:
- submission CSV: id, diagnosed_diabetes
- optional OOF CSV: id, y, oof_pred

Preprocessing (memory-friendly):
- Numeric: median impute
- Categorical: most-frequent impute + OrdinalEncoder (unknown -> -1)

This avoids one-hot expansion (keeps RAM bounded on 16GB).
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train RandomForest or sklearn GBM on S5E12 (CPU)")

    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("sub/submission_rf_gbm.csv"))
    p.add_argument("--oof-out", type=Path, default=None)

    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)

    p.add_argument("--model", type=str, default="hgb", help="rf|hgb")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")

    p.add_argument("--max-train-rows", type=int, default=0)

    p.add_argument("--train-weights", type=Path, default=None, help="CSV with columns id,weight")

    # RF params
    p.add_argument("--rf-n-estimators", type=int, default=600)
    p.add_argument("--rf-max-depth", type=int, default=0, help="0 means None")
    p.add_argument("--rf-min-samples-leaf", type=int, default=2)
    p.add_argument("--rf-max-features", type=str, default="sqrt")

    # HGB params
    p.add_argument("--hgb-max-iter", type=int, default=2500)
    p.add_argument("--hgb-learning-rate", type=float, default=0.05)
    p.add_argument("--hgb-max-leaf-nodes", type=int, default=63)
    p.add_argument("--hgb-max-depth", type=int, default=0, help="0 means None")
    p.add_argument("--hgb-min-samples-leaf", type=int, default=20)
    p.add_argument("--hgb-l2", type=float, default=0.0)
    p.add_argument("--hgb-max-bins", type=int, default=255)
    p.add_argument("--hgb-early-stopping", action="store_true")
    p.add_argument("--hgb-validation-fraction", type=float, default=0.1)
    p.add_argument("--hgb-n-iter-no-change", type=int, default=50)

    p.add_argument("--zip-output", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    args = p.parse_args(argv)

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

    model_kind = str(args.model).strip().lower()
    if model_kind not in {"rf", "hgb"}:
        raise ValueError("--model must be rf or hgb")

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds is empty")

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    id_col, target_col, feature_cols = _infer_cols(train_df, args.id_col, args.target_col)

    if args.max_train_rows and args.max_train_rows > 0:
        train_df = train_df.sample(n=min(int(args.max_train_rows), len(train_df)), random_state=int(seeds[0])).reset_index(drop=True)

    train_ids = pd.to_numeric(train_df[id_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    test_ids = pd.to_numeric(test_df[id_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()

    y = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    y = np.where(y > 0, 1, 0).astype(np.int64, copy=False)

    if args.train_weights is not None:
        sample_weight = _read_train_weights(Path(args.train_weights), train_ids)
    else:
        sample_weight = np.ones(len(train_df), dtype=np.float32)

    # Identify feature types based on train dtypes.
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_bool_dtype(train_df[c].dtype):
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(train_df[c].dtype):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Preprocess without one-hot to keep memory bounded.
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1, dtype=np.int32)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    # Cast final matrix to float32 for less RAM.
    cast32 = FunctionTransformer(lambda X: X.astype(np.float32, copy=False), feature_names_out="one-to-one")

    def make_estimator(seed: int):
        if model_kind == "rf":
            from sklearn.ensemble import RandomForestClassifier

            max_depth = None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth)
            return RandomForestClassifier(
                n_estimators=int(args.rf_n_estimators),
                max_depth=max_depth,
                min_samples_leaf=int(args.rf_min_samples_leaf),
                max_features=str(args.rf_max_features),
                n_jobs=-1,
                random_state=int(seed),
                class_weight=None,
            )

        from sklearn.ensemble import HistGradientBoostingClassifier

        max_depth = None if int(args.hgb_max_depth) <= 0 else int(args.hgb_max_depth)
        return HistGradientBoostingClassifier(
            max_iter=int(args.hgb_max_iter),
            learning_rate=float(args.hgb_learning_rate),
            max_leaf_nodes=int(args.hgb_max_leaf_nodes),
            max_depth=max_depth,
            min_samples_leaf=int(args.hgb_min_samples_leaf),
            l2_regularization=float(args.hgb_l2),
            max_bins=int(args.hgb_max_bins),
            early_stopping=bool(args.hgb_early_stopping),
            validation_fraction=float(args.hgb_validation_fraction),
            n_iter_no_change=int(args.hgb_n_iter_no_change),
            random_state=int(seed),
        )

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    oof_ens = np.zeros(len(train_df), dtype=np.float64)
    test_ens = np.zeros(len(test_df), dtype=np.float64)

    for si, seed in enumerate(seeds, start=1):
        skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(seed))
        oof = np.zeros(len(train_df), dtype=np.float64)
        test_accum = np.zeros(len(test_df), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            est = make_estimator(int(seed))

            pipe = Pipeline(steps=[("pre", pre), ("cast32", cast32), ("model", est)])

            sw_tr = sample_weight[tr_idx]

            pipe.fit(X_train.iloc[tr_idx], y[tr_idx], model__sample_weight=sw_tr)

            p_va = pipe.predict_proba(X_train.iloc[va_idx])[:, 1].astype(np.float64)
            p_te = pipe.predict_proba(X_test)[:, 1].astype(np.float64)

            oof[va_idx] = p_va
            test_accum += p_te / float(args.folds)

            ll = log_loss(y[va_idx], np.clip(p_va, 1e-15, 1 - 1e-15), sample_weight=sample_weight[va_idx])
            auc = roc_auc_score(y[va_idx], p_va)
            if args.verbose:
                print(f"[seed={seed}] fold={fold}/{args.folds} logloss={ll:.6f} auc={auc:.6f}", flush=True)

        ll_all = log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15), sample_weight=sample_weight)
        auc_all = roc_auc_score(y, oof)
        if args.verbose:
            print(f"[seed {si}/{len(seeds)}] done seed={seed} OOF logloss={ll_all:.6f} auc={auc_all:.6f}", flush=True)

        oof_ens += oof / float(len(seeds))
        test_ens += test_accum / float(len(seeds))

    ll_final = log_loss(y, np.clip(oof_ens, 1e-15, 1 - 1e-15), sample_weight=sample_weight)
    auc_final = roc_auc_score(y, oof_ens)
    if args.verbose:
        print(f"[final] model={model_kind} seeds={len(seeds)} OOF logloss={ll_final:.6f} auc={auc_final:.6f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test_ids, "diagnosed_diabetes": np.clip(test_ens, 0.0, 1.0)})
    sub.to_csv(args.out, index=False)
    if args.verbose:
        print(f"[done] wrote {args.out} rows={len(sub)}", flush=True)

    if args.oof_out is not None:
        oof_path = Path(args.oof_out)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame({"id": train_ids, "y": y.astype(np.int64), "oof_pred": np.clip(oof_ens, 0.0, 1.0)})
        oof_df.to_csv(oof_path, index=False)
        if args.verbose:
            print(f"[done] wrote {oof_path} rows={len(oof_df)}", flush=True)

    if args.zip_output:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname=args.out.name)
        if args.verbose:
            print(f"[done] wrote {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
