from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Batch:
    x_cat: np.ndarray  # int32
    x_num: np.ndarray  # float32
    y: Optional[np.ndarray] = None  # float32
    w: Optional[np.ndarray] = None  # float32


def iter_minibatches(
    x_cat: np.ndarray,
    x_num: np.ndarray,
    y: Optional[np.ndarray],
    w: Optional[np.ndarray],
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[Batch]:
    n = x_cat.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield Batch(
            x_cat=x_cat[b],
            x_num=x_num[b],
            y=None if y is None else y[b],
            w=None if w is None else w[b],
        )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
