from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TabularSpec:
    target_col: str
    id_col: str
    cat_cols: List[str]
    num_cols: List[str]
    cat_cardinalities: Dict[str, int]


@dataclass
class Encoders:
    cat_maps: Dict[str, Dict[object, int]]
    num_mean: np.ndarray
    num_std: np.ndarray


def infer_target_and_id(train_df: pd.DataFrame) -> Tuple[str, str]:
    target_candidates = [
        c
        for c in train_df.columns
        if c.lower() in {"diagnosed_diabetes", "target", "label", "y"}
    ]
    if not target_candidates:
        raise ValueError(
            "Could not infer target column. Expected one of: diagnosed_diabetes/target/label/y"
        )
    target_col = target_candidates[0]

    id_candidates = [c for c in train_df.columns if c.lower() == "id"]
    id_col = id_candidates[0] if id_candidates else "id"
    return target_col, id_col


def infer_feature_types(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    id_col: str,
    low_card_int_as_cat_threshold: int = 16,
) -> TabularSpec:
    all_cols = [c for c in train_df.columns if c not in {target_col}]
    feature_cols = [c for c in all_cols if c != id_col]

    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in feature_cols:
        if train_df[c].dtype == "object" or test_df[c].dtype == "object":
            cat_cols.append(c)
            continue

        # treat low-cardinality integer/bool columns as categorical
        s = train_df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
            u = int(s.nunique(dropna=False))
            if u <= low_card_int_as_cat_threshold:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            num_cols.append(c)

    # Build cardinalities from combined train+test
    cat_cardinalities: Dict[str, int] = {}
    for c in cat_cols:
        combined = pd.concat([train_df[c], test_df[c]], axis=0)
        # reserve 0 for unknown
        cat_cardinalities[c] = int(combined.astype("object").nunique(dropna=False)) + 1

    return TabularSpec(
        target_col=target_col,
        id_col=id_col,
        cat_cols=cat_cols,
        num_cols=num_cols,
        cat_cardinalities=cat_cardinalities,
    )


def fit_encoders(
    train_df: pd.DataFrame, test_df: pd.DataFrame, spec: TabularSpec
) -> Encoders:
    cat_maps: Dict[str, Dict[object, int]] = {}
    for c in spec.cat_cols:
        combined = pd.concat([train_df[c], test_df[c]], axis=0).astype("object")
        uniques = pd.Index(combined.unique())
        # 0 is unknown; start from 1
        mapping = {k: i + 1 for i, k in enumerate(uniques.tolist())}
        cat_maps[c] = mapping

    if spec.num_cols:
        num = train_df[spec.num_cols].to_numpy(dtype=np.float32)
        mean = np.nanmean(num, axis=0).astype(np.float32)
        std = np.nanstd(num, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    else:
        mean = np.zeros((0,), dtype=np.float32)
        std = np.ones((0,), dtype=np.float32)

    return Encoders(cat_maps=cat_maps, num_mean=mean, num_std=std)


def _encode_cat_column(values: pd.Series, mapping: Dict[object, int]) -> np.ndarray:
    # pandas map is fast enough for these cardinalities
    encoded = values.astype("object").map(mapping).fillna(0).astype(np.int32).to_numpy()
    return encoded


def transform_df(
    df: pd.DataFrame, spec: TabularSpec, enc: Encoders
) -> Tuple[np.ndarray, np.ndarray]:
    if spec.cat_cols:
        X_cat = np.stack(
            [_encode_cat_column(df[c], enc.cat_maps[c]) for c in spec.cat_cols], axis=1
        ).astype(np.int32, copy=False)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.int32)

    if spec.num_cols:
        X_num = df[spec.num_cols].to_numpy(dtype=np.float32)
        X_num = (X_num - enc.num_mean) / enc.num_std
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
    else:
        X_num = np.zeros((len(df), 0), dtype=np.float32)

    return X_cat, X_num


def load_dataframes(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
