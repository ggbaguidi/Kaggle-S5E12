#!/usr/bin/env python3
"""Train a RAM-friendly diabetes probability model (CPU-only).

Design:
- Reads CSVs in chunks (streaming) so it scales beyond RAM.
- Numeric: incremental standardization via a first streaming pass.
- Categorical: hashing trick via HashingVectorizer on "col=value" tokens.
- Model: SGDClassifier with log-loss + partial_fit.

Outputs a Kaggle-ready submission with columns: id, diagnosed_diabetes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Schema:
    id_col: str
    target_col: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


def _infer_schema(
    train_path: Path, id_col: str | None, target_col: str | None
) -> Schema:
    sample = pd.read_csv(train_path, nrows=500)

    inferred_id = id_col or ("id" if "id" in sample.columns else sample.columns[0])
    inferred_target = target_col or (
        "diagnosed_diabetes" if "diagnosed_diabetes" in sample.columns else None
    )
    if inferred_target is None or inferred_target not in sample.columns:
        raise ValueError(
            "Could not infer target column. Pass --target-col (e.g., diagnosed_diabetes)."
        )

    feature_cols = [
        c for c in sample.columns if c not in (inferred_id, inferred_target)
    ]

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for c in feature_cols:
        dt = sample[c].dtype
        if pd.api.types.is_numeric_dtype(dt):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return Schema(
        id_col=inferred_id,
        target_col=inferred_target,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def _read_csv_in_chunks(
    path: Path, chunk_size: int, usecols: list[str] | None = None
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=chunk_size, usecols=usecols)


def _stable_val_mask(ids: np.ndarray, val_frac: float, val_seed: int) -> np.ndarray:
    """Deterministic per-row validation mask based on id hashing.

    This avoids shuffling or holding data in RAM.
    """
    if val_frac <= 0:
        return np.zeros(len(ids), dtype=bool)
    if val_frac >= 1:
        return np.ones(len(ids), dtype=bool)

    x = ids.astype(np.int64, copy=False)
    # Handle NaNs or non-numeric ids coerced to <NA> earlier.
    x = np.where(np.isfinite(x), x, 0).astype(np.int64, copy=False)
    u = x.astype(np.uint64)
    u ^= np.uint64(val_seed)
    u *= np.uint64(11400714819323198485)  # 64-bit multiplicative hash
    # Map to 0..999999 buckets
    buckets = ((u >> np.uint64(32)) % np.uint64(1_000_000)).astype(np.uint32)
    return buckets < int(val_frac * 1_000_000)


def _categorical_text_frame(
    df: pd.DataFrame, categorical_cols: list[str]
) -> np.ndarray:
    """Return an array of strings, one per row, containing 'col=value' tokens."""
    if not categorical_cols:
        return np.asarray([""] * len(df), dtype=object)

    txt = pd.Series([""] * len(df), index=df.index, dtype="string")
    for col in categorical_cols:
        vals = df[col].astype("string").fillna("__MISSING__")
        txt = txt + (col + "=" + vals + " ")
    return txt.to_numpy(dtype=object)


def _numeric_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    if not numeric_cols:
        return np.zeros((len(df), 0), dtype=np.float32)

    x = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0)
    return x.to_numpy(dtype=np.float32, copy=False)


def _build_features(
    df: pd.DataFrame,
    schema: Schema,
    vectorizer: HashingVectorizer,
    scaler: StandardScaler | None,
) -> sparse.csr_matrix:
    x_num = _numeric_matrix(df, schema.numeric_cols)
    if scaler is not None and x_num.shape[1] > 0:
        x_num = scaler.transform(x_num)

    x_cat_txt = _categorical_text_frame(df, schema.categorical_cols)
    x_cat = vectorizer.transform(x_cat_txt)  # sparse

    x_num_sp = (
        sparse.csr_matrix(x_num)
        if x_num.shape[1] > 0
        else sparse.csr_matrix((len(df), 0))
    )
    return sparse.hstack([x_num_sp, x_cat], format="csr")


def _stream_fit_scaler(
    train_path: Path,
    schema: Schema,
    chunk_size: int,
    *,
    val_frac: float = 0.0,
    val_seed: int = 0,
    train_only: bool = False,
) -> StandardScaler:
    scaler = StandardScaler(with_mean=True, with_std=True)
    if not schema.numeric_cols:
        return scaler

    usecols = [schema.id_col, *schema.numeric_cols]
    for chunk in _read_csv_in_chunks(
        train_path, chunk_size=chunk_size, usecols=usecols
    ):
        if train_only and val_frac > 0:
            ids = (
                pd.to_numeric(chunk[schema.id_col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            val_mask = _stable_val_mask(ids, val_frac=val_frac, val_seed=val_seed)
            if val_mask.all():
                continue
            chunk = chunk.loc[~val_mask]

        x_num = _numeric_matrix(chunk, schema.numeric_cols)
        if x_num.shape[0] == 0:
            continue
        scaler.partial_fit(x_num)
    return scaler


def _stream_class_counts(
    *,
    train_path: Path,
    schema: Schema,
    chunk_size: int,
    val_frac: float = 0.0,
    val_seed: int = 0,
    train_only: bool = False,
    max_rows: int | None = None,
) -> np.ndarray:
    """Count class occurrences in a streaming pass.

    Returns counts for classes [0, 1].
    """
    counts = np.zeros(2, dtype=np.int64)
    seen = 0

    usecols = [schema.id_col, schema.target_col]
    for chunk in _read_csv_in_chunks(
        train_path, chunk_size=chunk_size, usecols=usecols
    ):
        if val_frac > 0:
            ids = (
                pd.to_numeric(chunk[schema.id_col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            val_mask = _stable_val_mask(ids, val_frac=val_frac, val_seed=val_seed)
            chunk = chunk.loc[~val_mask] if train_only else chunk.loc[val_mask]

        if len(chunk) == 0:
            continue

        y = (
            pd.to_numeric(chunk[schema.target_col], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        # Ensure binary 0/1
        y = np.where(y > 0, 1, 0).astype(np.int64, copy=False)
        binc = np.bincount(y, minlength=2)
        counts[:2] += binc[:2].astype(np.int64)

        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            break

    return counts


def _balanced_sample_weight(y: np.ndarray, class_counts: np.ndarray) -> np.ndarray:
    """Compute sklearn-style 'balanced' weights for a binary label array."""
    n = int(class_counts.sum())
    # Avoid division by zero in degenerate splits
    w0 = n / (2.0 * max(1, int(class_counts[0])))
    w1 = n / (2.0 * max(1, int(class_counts[1])))
    return np.where(y > 0, w1, w0).astype(np.float64, copy=False)


def _make_sgd_model(seed: int, cfg: dict[str, Any]) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty=cfg.get("penalty", "l2"),
        alpha=float(cfg.get("alpha", 1e-5)),
        l1_ratio=cfg.get("l1_ratio", 0.15),
        learning_rate=cfg.get("learning_rate", "optimal"),
        eta0=cfg.get("eta0", 0.0),
        fit_intercept=True,
        average=True,
        # NOTE: SGDClassifier.partial_fit does NOT support class_weight='balanced'.
        # We implement balanced weighting via per-sample weights in _stream_train_sgd.
        class_weight=None,
        random_state=seed,
    )


def _stream_train_sgd(
    *,
    model: SGDClassifier,
    train_path: Path,
    schema: Schema,
    vectorizer: HashingVectorizer,
    scaler: StandardScaler,
    chunk_size: int,
    passes: int,
    val_frac: float = 0.0,
    val_seed: int = 0,
    train_only: bool = False,
    max_train_rows: int | None = None,
    class_counts: np.ndarray | None = None,
) -> None:
    classes = np.array([0, 1], dtype=np.int64)
    fitted = False
    seen = 0

    for epoch in range(passes):
        for chunk in _read_csv_in_chunks(
            train_path,
            chunk_size=chunk_size,
            usecols=[schema.id_col, schema.target_col, *schema.feature_cols],
        ):
            if train_only and val_frac > 0:
                ids = (
                    pd.to_numeric(chunk[schema.id_col], errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                    .to_numpy()
                )
                val_mask = _stable_val_mask(ids, val_frac=val_frac, val_seed=val_seed)
                if val_mask.all():
                    continue
                chunk = chunk.loc[~val_mask]

            if len(chunk) == 0:
                continue

            y = (
                pd.to_numeric(chunk[schema.target_col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            y = np.where(y > 0, 1, 0).astype(np.int64, copy=False)
            x = _build_features(
                chunk, schema=schema, vectorizer=vectorizer, scaler=scaler
            )

            sw = None
            if class_counts is not None:
                sw = _balanced_sample_weight(y, class_counts=class_counts)

            if not fitted:
                model.partial_fit(x, y, classes=classes, sample_weight=sw)
                fitted = True
            else:
                model.partial_fit(x, y, sample_weight=sw)

            seen += len(chunk)
            if max_train_rows is not None and seen >= max_train_rows:
                return


def _stream_val_logloss(
    *,
    model: SGDClassifier,
    train_path: Path,
    schema: Schema,
    vectorizer: HashingVectorizer,
    scaler: StandardScaler,
    chunk_size: int,
    val_frac: float,
    val_seed: int,
) -> float:
    eps = 1e-15
    loss_sum = 0.0
    n = 0

    for chunk in _read_csv_in_chunks(
        train_path,
        chunk_size=chunk_size,
        usecols=[schema.id_col, schema.target_col, *schema.feature_cols],
    ):
        ids = (
            pd.to_numeric(chunk[schema.id_col], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        val_mask = _stable_val_mask(ids, val_frac=val_frac, val_seed=val_seed)
        if not val_mask.any():
            continue

        v = chunk.loc[val_mask]
        if len(v) == 0:
            continue

        y = (
            pd.to_numeric(v[schema.target_col], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        x = _build_features(v, schema=schema, vectorizer=vectorizer, scaler=scaler)
        p = model.predict_proba(x)[:, 1]
        p = np.clip(p, eps, 1 - eps)
        loss_sum += float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).sum())
        n += int(len(y))

    return loss_sum / max(1, n)


def train_and_predict(
    train_path: Path,
    test_path: Path,
    out_path: Path,
    chunk_size: int,
    n_hash_features: int,
    seed: int,
    passes: int,
    id_col: str | None,
    target_col: str | None,
    *,
    val_frac: float = 0.0,
    val_seed: int = 42,
    tune: bool = False,
    tune_limit: int = 0,
    tune_passes: int = 1,
    tune_max_train_rows: int = 0,
    tune_out: Path | None = None,
    verbose: int = 1,
    zip_output: bool = False,
) -> None:
    schema = _infer_schema(train_path, id_col=id_col, target_col=target_col)

    vectorizer = HashingVectorizer(
        n_features=n_hash_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
        token_pattern=r"[^ ]+",
    )

    if tune and val_frac <= 0:
        raise ValueError("--tune requires --val-frac > 0")

    chosen_cfg: dict[str, Any] = {
        "penalty": "l2",
        "alpha": 1e-5,
        "learning_rate": "optimal",
        "class_weight": None,
    }

    if tune:
        if verbose:
            print(
                f"[tune] val_frac={val_frac} val_seed={val_seed} tune_passes={tune_passes}",
                flush=True,
            )
        # Fit scaler on train-only portion (avoid leakage into validation).
        scaler_tv = _stream_fit_scaler(
            train_path,
            schema=schema,
            chunk_size=chunk_size,
            val_frac=val_frac,
            val_seed=val_seed,
            train_only=True,
        )

        # If we want balanced weights, we need a stable estimate of class frequencies.
        counts_tv = _stream_class_counts(
            train_path=train_path,
            schema=schema,
            chunk_size=chunk_size,
            val_frac=val_frac,
            val_seed=val_seed,
            train_only=True,
            max_rows=(
                int(tune_max_train_rows)
                if tune_max_train_rows and tune_max_train_rows > 0
                else None
            ),
        )

        grid: list[dict[str, Any]] = []
        for alpha in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
            for cw in (None, "balanced"):
                grid.append(
                    {
                        "penalty": "l2",
                        "alpha": alpha,
                        "learning_rate": "optimal",
                        "class_weight": cw,
                    }
                )
        for alpha in (1e-6, 1e-5, 3e-5):
            for l1_ratio in (0.15, 0.5):
                for cw in (None, "balanced"):
                    grid.append(
                        {
                            "penalty": "elasticnet",
                            "alpha": alpha,
                            "l1_ratio": l1_ratio,
                            "learning_rate": "optimal",
                            "class_weight": cw,
                        }
                    )
        # A couple of constant-LR candidates (sometimes helps with convergence)
        grid.append(
            {
                "penalty": "l2",
                "alpha": 1e-5,
                "learning_rate": "constant",
                "eta0": 0.02,
                "class_weight": None,
            }
        )
        grid.append(
            {
                "penalty": "l2",
                "alpha": 1e-5,
                "learning_rate": "constant",
                "eta0": 0.05,
                "class_weight": "balanced",
            }
        )

        if tune_limit and tune_limit > 0:
            grid = grid[: int(tune_limit)]

        best_loss = float("inf")
        best_cfg: dict[str, Any] | None = None
        results: list[dict[str, Any]] = []

        max_rows = (
            int(tune_max_train_rows)
            if tune_max_train_rows and tune_max_train_rows > 0
            else None
        )
        for i, cfg in enumerate(grid, start=1):
            m = _make_sgd_model(seed=seed, cfg=cfg)

            cw = cfg.get("class_weight", None)
            use_counts = counts_tv if cw == "balanced" else None

            _stream_train_sgd(
                model=m,
                train_path=train_path,
                schema=schema,
                vectorizer=vectorizer,
                scaler=scaler_tv,
                chunk_size=chunk_size,
                passes=max(1, int(tune_passes)),
                val_frac=val_frac,
                val_seed=val_seed,
                train_only=True,
                max_train_rows=max_rows,
                class_counts=use_counts,
            )
            loss = _stream_val_logloss(
                model=m,
                train_path=train_path,
                schema=schema,
                vectorizer=vectorizer,
                scaler=scaler_tv,
                chunk_size=chunk_size,
                val_frac=val_frac,
                val_seed=val_seed,
            )
            row = {"val_logloss": loss, **cfg}
            results.append(row)
            if loss < best_loss:
                best_loss = loss
                best_cfg = cfg

            if verbose:
                print(
                    f"[tune] {i:02d}/{len(grid)} val_logloss={loss:.6f} cfg={cfg}",
                    flush=True,
                )

        if best_cfg is None:
            raise RuntimeError("Tuning failed: no configs evaluated")

        chosen_cfg = dict(best_cfg)
        if verbose:
            print(
                f"[tune] best_val_logloss={best_loss:.6f} best_cfg={chosen_cfg}",
                flush=True,
            )
        if tune_out is not None:
            tune_out.parent.mkdir(parents=True, exist_ok=True)
            tune_out.write_text(
                json.dumps(
                    {
                        "best": {"val_logloss": best_loss, **chosen_cfg},
                        "trials": results,
                    },
                    indent=2,
                )
            )
            if verbose:
                print(f"[tune] wrote {tune_out}", flush=True)

    # Final fit on ALL training data with chosen hyperparameters.
    scaler = _stream_fit_scaler(train_path, schema=schema, chunk_size=chunk_size)
    model = _make_sgd_model(seed=seed, cfg=chosen_cfg)

    cw_final = chosen_cfg.get("class_weight", None)
    counts_full = None
    if cw_final == "balanced":
        counts_full = _stream_class_counts(
            train_path=train_path, schema=schema, chunk_size=chunk_size
        )

    if verbose:
        print(
            f"[fit] passes={passes} hash_dim={n_hash_features} cfg={chosen_cfg}",
            flush=True,
        )

    _stream_train_sgd(
        model=model,
        train_path=train_path,
        schema=schema,
        vectorizer=vectorizer,
        scaler=scaler,
        chunk_size=chunk_size,
        passes=max(1, passes),
        class_counts=counts_full,
    )

    # Stream predictions over test
    ids: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for chunk in _read_csv_in_chunks(
        test_path, chunk_size=chunk_size, usecols=[schema.id_col, *schema.feature_cols]
    ):
        x = _build_features(chunk, schema=schema, vectorizer=vectorizer, scaler=scaler)
        p = model.predict_proba(x)[:, 1].astype(np.float64)
        probs.append(p)
        ids.append(
            pd.to_numeric(chunk[schema.id_col], errors="coerce")
            .astype(np.int64)
            .to_numpy()
        )

    sub = pd.DataFrame(
        {schema.id_col: np.concatenate(ids), schema.target_col: np.concatenate(probs)}
    )
    # Match Kaggle sample submission naming
    if schema.id_col != "id":
        sub = sub.rename(columns={schema.id_col: "id"})
    if schema.target_col != "diagnosed_diabetes":
        sub = sub.rename(columns={schema.target_col: "diagnosed_diabetes"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    if verbose:
        print(f"[done] wrote {out_path} rows={len(sub)}", flush=True)

    if zip_output:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, out_path.name)
        if verbose:
            print(f"[done] zipped to {zip_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Streaming CPU-only trainer for diabetes probability prediction"
    )
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--test", type=Path, default=Path("data/test.csv"))
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--hash-dim", type=int, default=2**18)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument("--tune", action="store_true")
    p.add_argument(
        "--tune-limit",
        type=int,
        default=0,
        help="Limit number of configs evaluated (0 = all)",
    )
    p.add_argument("--tune-passes", type=int, default=1)
    p.add_argument(
        "--tune-max-train-rows",
        type=int,
        default=0,
        help="Cap rows used for tuning (0 = no cap)",
    )
    p.add_argument("--tune-out", type=Path, default=Path("tuning_results.json"))
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument(
        "--zip-output",
        action="store_true",
        help="Zip the output CSV file after writing",
    )
    args = p.parse_args(argv)

    train_and_predict(
        train_path=args.train,
        test_path=args.test,
        out_path=args.out,
        chunk_size=args.chunk_size,
        n_hash_features=args.hash_dim,
        seed=args.seed,
        passes=max(1, args.passes),
        id_col=args.id_col,
        target_col=args.target_col,
        val_frac=float(args.val_frac),
        val_seed=int(args.val_seed),
        tune=bool(args.tune),
        tune_limit=int(args.tune_limit),
        tune_passes=int(args.tune_passes),
        tune_max_train_rows=int(args.tune_max_train_rows),
        tune_out=(args.tune_out if args.tune else None),
        verbose=int(args.verbose),
        zip_output=bool(args.zip_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
