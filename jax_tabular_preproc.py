#!/usr/bin/env python3
"""Shared tabular preprocessing for JAX/Flax models.

Design goals:
- Deterministic encoding shared across train/test (and optional extra frames).
- Categorical columns: string -> top-K vocab (train+test) + __MISSING__/__OTHER__.
- Numeric columns: to float32, impute with train mean, standardize by train std.

This module is intentionally numpy/pandas-only so it can be reused by multiple
scripts without pulling JAX just to parse data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"


def infer_id_target_features(
    train: pd.DataFrame,
    *,
    id_col: str | None,
    target_col: str | None,
) -> tuple[str, str, list[str]]:
    id_col = id_col or ("id" if "id" in train.columns else train.columns[0])
    target_col = target_col or ("diagnosed_diabetes" if "diagnosed_diabetes" in train.columns else None)
    if target_col is None or target_col not in train.columns:
        raise ValueError("Could not infer target column; pass --target-col")
    features = [c for c in train.columns if c not in (id_col, target_col)]
    return id_col, target_col, features


def _is_numeric_series(s: pd.Series) -> bool:
    # Treat bool as categorical (often behaves like category), everything else numeric if numeric dtype.
    if pd.api.types.is_bool_dtype(s.dtype):
        return False
    return pd.api.types.is_numeric_dtype(s.dtype)


@dataclass
class TabularPreprocessor:
    id_col: str
    target_col: str | None
    feature_cols: list[str]

    num_cols: list[str]
    cat_cols: list[str]

    num_mean: np.ndarray  # float32, shape (n_num,)
    num_std: np.ndarray  # float32, shape (n_num,)

    # For each cat col: keep list and mapping
    cat_keep: dict[str, list[str]]
    cat_to_index: dict[str, dict[str, int]]

    @property
    def cat_sizes(self) -> list[int]:
        return [len(self.cat_keep[c]) for c in self.cat_cols]

    def transform_ids(self, df: pd.DataFrame) -> np.ndarray:
        ids = pd.to_numeric(df[self.id_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        return ids

    def transform_y(self, df: pd.DataFrame) -> np.ndarray:
        if self.target_col is None:
            raise ValueError("No target_col set on this preprocessor")
        y = pd.to_numeric(df[self.target_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        y = np.where(y > 0, 1, 0).astype(np.float32, copy=False)
        return y

    def transform_X(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Categorical -> int32 codes
        if self.cat_cols:
            cat_arrays: list[np.ndarray] = []
            for c in self.cat_cols:
                keep = self.cat_to_index[c]
                s = df[c].astype("string").fillna(MISSING_TOKEN)
                # Map unknowns to OTHER
                arr = s.to_numpy(dtype=object, copy=False)
                codes = np.empty(len(arr), dtype=np.int32)
                other_idx = int(keep[OTHER_TOKEN])
                miss_idx = int(keep[MISSING_TOKEN])
                for i, v in enumerate(arr):
                    if v is None:
                        codes[i] = miss_idx
                        continue
                    vv = str(v)
                    if vv == "<NA>":
                        codes[i] = miss_idx
                        continue
                    codes[i] = int(keep.get(vv, other_idx))
                cat_arrays.append(codes)
            x_cat = np.stack(cat_arrays, axis=1)
        else:
            x_cat = np.zeros((len(df), 0), dtype=np.int32)

        # Numeric -> float32 standardized
        if self.num_cols:
            x = df[self.num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
            x = np.where(np.isfinite(x), x, np.nan).astype(np.float32, copy=False)
            x = np.where(np.isnan(x), self.num_mean[None, :], x)
            x = (x - self.num_mean[None, :]) / self.num_std[None, :]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x_num = x.astype(np.float32, copy=False)
        else:
            x_num = np.zeros((len(df), 0), dtype=np.float32)

        return x_num, x_cat

    def to_json(self) -> dict[str, Any]:
        return {
            "id_col": self.id_col,
            "target_col": self.target_col,
            "feature_cols": self.feature_cols,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "num_mean": self.num_mean.tolist(),
            "num_std": self.num_std.tolist(),
            "cat_keep": self.cat_keep,
        }

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "TabularPreprocessor":
        cat_keep: dict[str, list[str]] = {k: list(v) for k, v in dict(obj["cat_keep"]).items()}
        cat_to_index = {k: {vv: i for i, vv in enumerate(v)} for k, v in cat_keep.items()}
        return TabularPreprocessor(
            id_col=str(obj["id_col"]),
            target_col=(None if obj.get("target_col") in (None, "") else str(obj.get("target_col"))),
            feature_cols=list(obj["feature_cols"]),
            num_cols=list(obj["num_cols"]),
            cat_cols=list(obj["cat_cols"]),
            num_mean=np.asarray(obj["num_mean"], dtype=np.float32),
            num_std=np.asarray(obj["num_std"], dtype=np.float32),
            cat_keep=cat_keep,
            cat_to_index=cat_to_index,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False))

    @staticmethod
    def load(path: Path) -> "TabularPreprocessor":
        return TabularPreprocessor.from_json(json.loads(path.read_text()))


def fit_preprocessor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str,
    target_col: str | None,
    feature_cols: list[str],
    max_categories: int,
) -> TabularPreprocessor:
    # Infer numeric vs categorical using train dtypes.
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in feature_cols:
        if _is_numeric_series(train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Numeric stats from train only.
    if num_cols:
        tr_num = train[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
        tr_num = np.where(np.isfinite(tr_num), tr_num, np.nan).astype(np.float32, copy=False)
        mean = np.nanmean(tr_num, axis=0).astype(np.float32)
        std = np.nanstd(tr_num, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    else:
        mean = np.zeros((0,), dtype=np.float32)
        std = np.ones((0,), dtype=np.float32)

    # Categorical vocab from train+test to avoid train/test mismatch.
    cat_keep: dict[str, list[str]] = {}
    cat_to_index: dict[str, dict[str, int]] = {}

    max_categories = int(max_categories)
    if max_categories < 4:
        raise ValueError("max_categories must be >= 4")

    for c in cat_cols:
        combo = pd.concat([train[c], test[c]], axis=0, ignore_index=True)
        combo = combo.astype("string").fillna(MISSING_TOKEN)
        vc = combo.value_counts(dropna=False)
        # Reserve 2 slots for MISSING and OTHER
        k = max_categories - 2
        top = vc.nlargest(k).index.astype(str).tolist() if len(vc) > k else vc.index.astype(str).tolist()

        # Ensure special tokens exist and have stable indices.
        keep_list = [MISSING_TOKEN, OTHER_TOKEN]
        for t in top:
            if t in (MISSING_TOKEN, OTHER_TOKEN):
                continue
            keep_list.append(t)

        cat_keep[c] = keep_list
        cat_to_index[c] = {v: i for i, v in enumerate(keep_list)}

    return TabularPreprocessor(
        id_col=str(id_col),
        target_col=(None if target_col is None else str(target_col)),
        feature_cols=list(feature_cols),
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_mean=mean,
        num_std=std,
        cat_keep=cat_keep,
        cat_to_index=cat_to_index,
    )
