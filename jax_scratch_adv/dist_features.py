from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DistFeatureConfig:
    num_bins: int = 128
    alpha: float = 0.5
    quantile_max_rows: int = 200_000
    add_per_feature_llr: bool = True
    add_nb_logit: bool = True
    add_shift: bool = False


@dataclass(frozen=True)
class LlrEstimator:
    # categorical counts: shape (n_cat_cols, max_card) stored ragged via list
    cat_count0: list[np.ndarray]
    cat_count1: list[np.ndarray]
    # numeric bin edges and counts per bin
    num_edges: np.ndarray  # (n_num_cols, num_bins-1)
    num_count0: np.ndarray  # (n_num_cols, num_bins)
    num_count1: np.ndarray  # (n_num_cols, num_bins)
    n0: float
    n1: float


@dataclass(frozen=True)
class ShiftEstimator:
    cat_train_counts: list[np.ndarray]
    cat_test_counts: list[np.ndarray]
    num_edges: np.ndarray
    num_train_counts: np.ndarray
    num_test_counts: np.ndarray
    n_train: float
    n_test: float


def _safe_choice(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return rng.choice(np.arange(n), size=k, replace=False)


def _digitize_1d(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # edges: (num_bins-1,), returns bin idx in [0, num_bins-1]
    # np.searchsorted is fast and stable.
    return np.searchsorted(edges, x, side="right").astype(np.int32, copy=False)


def fit_llr_estimator(
    x_cat: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray,
    *,
    cat_cardinalities: list[int],
    cfg: DistFeatureConfig,
    seed: int,
) -> LlrEstimator:
    """Fit per-feature conditional distributions p(x_j|y=0/1).

    Uses smoothed frequency estimates.
    - Categoricals: per-category counts.
    - Numerics: quantile binning from a subsample, then per-bin counts.

    This is intended for cross-fitting: fit on fold-train, transform fold-val.
    """

    rng = np.random.default_rng(seed)
    y01 = (y > 0.5).astype(np.int32, copy=False)
    n1 = float(y01.sum())
    n0 = float(len(y01) - y01.sum())

    # categorical counts
    cat_count0: list[np.ndarray] = []
    cat_count1: list[np.ndarray] = []
    for j, card in enumerate(cat_cardinalities):
        col = x_cat[:, j]
        c1 = np.bincount(col, weights=y01, minlength=card).astype(np.float32)
        c0 = np.bincount(col, weights=1.0 - y01, minlength=card).astype(np.float32)
        cat_count0.append(c0)
        cat_count1.append(c1)

    # numeric bin edges
    n_num = x_num.shape[1]
    num_bins = int(cfg.num_bins)
    if n_num > 0:
        take = _safe_choice(rng, x_num.shape[0], int(cfg.quantile_max_rows))
        xs = x_num[take]
        # compute edges per feature: (num_bins-1) quantiles between (0,1)
        qs = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)[1:-1]
        edges = np.quantile(xs, qs, axis=0).T.astype(np.float32, copy=False)
        # Ensure monotone edges (quantile degeneracy can create repeats)
        edges = np.maximum.accumulate(edges, axis=1)

        num_count0 = np.zeros((n_num, num_bins), dtype=np.float32)
        num_count1 = np.zeros((n_num, num_bins), dtype=np.float32)
        for j in range(n_num):
            b = _digitize_1d(x_num[:, j], edges[j])
            num_count1[j] = np.bincount(b, weights=y01, minlength=num_bins).astype(np.float32)
            num_count0[j] = np.bincount(b, weights=1.0 - y01, minlength=num_bins).astype(np.float32)
    else:
        edges = np.zeros((0, num_bins - 1), dtype=np.float32)
        num_count0 = np.zeros((0, num_bins), dtype=np.float32)
        num_count1 = np.zeros((0, num_bins), dtype=np.float32)

    return LlrEstimator(
        cat_count0=cat_count0,
        cat_count1=cat_count1,
        num_edges=edges,
        num_count0=num_count0,
        num_count1=num_count1,
        n0=n0,
        n1=n1,
    )


def transform_llr_features(
    est: LlrEstimator,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    *,
    cfg: DistFeatureConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (per_feature_llr, nb_logit).

    per_feature_llr has shape (n_rows, n_cat_cols + n_num_cols) if enabled, else (n_rows, 0).
    nb_logit has shape (n_rows, 1) if enabled, else (n_rows, 0).
    """

    alpha = float(cfg.alpha)
    n_rows = x_cat.shape[0]

    llrs = []

    # categorical llr per column
    for j, (c0, c1) in enumerate(zip(est.cat_count0, est.cat_count1)):
        card = c0.shape[0]
        # p(x|y) with smoothing
        p1 = (c1 + alpha) / (est.n1 + alpha * card)
        p0 = (c0 + alpha) / (est.n0 + alpha * card)
        col = x_cat[:, j]
        llr = np.log(p1[col]) - np.log(p0[col])
        llrs.append(llr.astype(np.float32, copy=False))

    # numeric llr per column
    for j in range(x_num.shape[1]):
        num_bins = est.num_count0.shape[1]
        edges = est.num_edges[j]
        b = _digitize_1d(x_num[:, j], edges)
        c1 = est.num_count1[j]
        c0 = est.num_count0[j]
        p1 = (c1 + alpha) / (est.n1 + alpha * num_bins)
        p0 = (c0 + alpha) / (est.n0 + alpha * num_bins)
        llr = np.log(p1[b]) - np.log(p0[b])
        llrs.append(llr.astype(np.float32, copy=False))

    if cfg.add_per_feature_llr and llrs:
        per_feature = np.stack(llrs, axis=1).astype(np.float32, copy=False)
    else:
        per_feature = np.zeros((n_rows, 0), dtype=np.float32)

    if cfg.add_nb_logit and llrs:
        nb = np.sum(np.stack(llrs, axis=1), axis=1, keepdims=True).astype(np.float32, copy=False)
    else:
        nb = np.zeros((n_rows, 0), dtype=np.float32)

    return per_feature, nb


def fit_shift_estimator(
    x_cat_train: np.ndarray,
    x_num_train: np.ndarray,
    x_cat_test: np.ndarray,
    x_num_test: np.ndarray,
    *,
    cat_cardinalities: list[int],
    cfg: DistFeatureConfig,
    seed: int,
) -> ShiftEstimator:
    """Fit per-feature marginal distributions p(x|train) and p(x|test)."""

    rng = np.random.default_rng(seed)

    # categorical counts
    cat_train_counts: list[np.ndarray] = []
    cat_test_counts: list[np.ndarray] = []
    for j, card in enumerate(cat_cardinalities):
        col_tr = x_cat_train[:, j]
        col_te = x_cat_test[:, j]
        cat_train_counts.append(np.bincount(col_tr, minlength=card).astype(np.float32))
        cat_test_counts.append(np.bincount(col_te, minlength=card).astype(np.float32))

    # numeric edges from (subsampled) train
    n_num = x_num_train.shape[1]
    num_bins = int(cfg.num_bins)
    if n_num > 0:
        take = _safe_choice(rng, x_num_train.shape[0], int(cfg.quantile_max_rows))
        xs = x_num_train[take]
        qs = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)[1:-1]
        edges = np.quantile(xs, qs, axis=0).T.astype(np.float32, copy=False)
        edges = np.maximum.accumulate(edges, axis=1)

        num_train_counts = np.zeros((n_num, num_bins), dtype=np.float32)
        num_test_counts = np.zeros((n_num, num_bins), dtype=np.float32)
        for j in range(n_num):
            b_tr = _digitize_1d(x_num_train[:, j], edges[j])
            b_te = _digitize_1d(x_num_test[:, j], edges[j])
            num_train_counts[j] = np.bincount(b_tr, minlength=num_bins).astype(np.float32)
            num_test_counts[j] = np.bincount(b_te, minlength=num_bins).astype(np.float32)
    else:
        edges = np.zeros((0, num_bins - 1), dtype=np.float32)
        num_train_counts = np.zeros((0, num_bins), dtype=np.float32)
        num_test_counts = np.zeros((0, num_bins), dtype=np.float32)

    return ShiftEstimator(
        cat_train_counts=cat_train_counts,
        cat_test_counts=cat_test_counts,
        num_edges=edges,
        num_train_counts=num_train_counts,
        num_test_counts=num_test_counts,
        n_train=float(x_cat_train.shape[0]),
        n_test=float(x_cat_test.shape[0]),
    )


def transform_shift_features(
    est: ShiftEstimator,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    *,
    cfg: DistFeatureConfig,
) -> np.ndarray:
    """Return per-feature log p(test)/p(train) (one value per original feature)."""

    alpha = float(cfg.alpha)
    out = []

    # categorical
    for j, (ctr, cte) in enumerate(zip(est.cat_train_counts, est.cat_test_counts)):
        card = ctr.shape[0]
        p_te = (cte + alpha) / (est.n_test + alpha * card)
        p_tr = (ctr + alpha) / (est.n_train + alpha * card)
        col = x_cat[:, j]
        out.append((np.log(p_te[col]) - np.log(p_tr[col])).astype(np.float32, copy=False))

    # numeric
    for j in range(x_num.shape[1]):
        num_bins = est.num_train_counts.shape[1]
        b = _digitize_1d(x_num[:, j], est.num_edges[j])
        ctr = est.num_train_counts[j]
        cte = est.num_test_counts[j]
        p_te = (cte + alpha) / (est.n_test + alpha * num_bins)
        p_tr = (ctr + alpha) / (est.n_train + alpha * num_bins)
        out.append((np.log(p_te[b]) - np.log(p_tr[b])).astype(np.float32, copy=False))

    if not out:
        return np.zeros((x_cat.shape[0], 0), dtype=np.float32)

    return np.stack(out, axis=1).astype(np.float32, copy=False)
