from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax.training import train_state

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from .model import ModelConfig, TabularEmbedMLP
from .train_utils import iter_minibatches


@dataclass(frozen=True)
class AdvConfig:
    epochs: int = 3
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 1e-4
    width: int = 96
    blocks: int = 2
    dropout: float = 0.05
    embed_dim: int = 8
    max_rows: int = 250_000
    seed: int = 0
    clip_min: float = 0.2
    clip_max: float = 5.0


def compute_adv_weights_sklearn(
    x_cat_train: np.ndarray,
    x_num_train: np.ndarray,
    x_cat_test: np.ndarray,
    x_num_test: np.ndarray,
    *,
    adv_cfg: AdvConfig,
) -> np.ndarray:
    """Fast CPU adversarial weights via a linear classifier.

    Trains a domain classifier to distinguish train(0) vs test(1), then returns
    clipped importance weights ~ p(test|x) / p(train|x).
    """
    rng = np.random.default_rng(adv_cfg.seed)

    n_train = x_cat_train.shape[0]
    n_test = x_cat_test.shape[0]

    train_idx = np.arange(n_train)
    test_idx = np.arange(n_test)
    if n_train > adv_cfg.max_rows:
        train_idx = rng.choice(train_idx, size=adv_cfg.max_rows, replace=False)
    if n_test > adv_cfg.max_rows:
        test_idx = rng.choice(test_idx, size=adv_cfg.max_rows, replace=False)

    x_cat_s = np.concatenate([x_cat_train[train_idx], x_cat_test[test_idx]], axis=0)
    x_num_s = np.concatenate([x_num_train[train_idx], x_num_test[test_idx]], axis=0)
    y_s = np.concatenate(
        [np.zeros((len(train_idx),), dtype=np.int32), np.ones((len(test_idx),), dtype=np.int32)],
        axis=0,
    )

    perm = rng.permutation(len(y_s))
    x_cat_s = x_cat_s[perm]
    x_num_s = x_num_s[perm]
    y_s = y_s[perm]

    # One-hot encode categorical ints (small cardinalities here; sparse is safe)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x_cat_s_ohe = ohe.fit_transform(x_cat_s)
    x_num_s_sp = sparse.csr_matrix(x_num_s.astype(np.float32, copy=False))
    X_s = sparse.hstack([x_num_s_sp, x_cat_s_ohe], format="csr")

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=20,
        tol=1e-3,
        random_state=adv_cfg.seed,
    )
    clf.fit(X_s, y_s)

    # Predict p(test|x) on full train
    x_cat_ohe = ohe.transform(x_cat_train)
    x_num_sp = sparse.csr_matrix(x_num_train.astype(np.float32, copy=False))
    X_train = sparse.hstack([x_num_sp, x_cat_ohe], format="csr")
    probs = clf.predict_proba(X_train)[:, 1].astype(np.float32)

    eps = 1e-4
    probs = np.clip(probs, eps, 1.0 - eps)
    w = probs / (1.0 - probs)
    w = np.clip(w, adv_cfg.clip_min, adv_cfg.clip_max).astype(np.float32)
    w = (w / np.mean(w)).astype(np.float32)
    return w


class TrainState(train_state.TrainState):
    dropout_rng: jax.Array


def _bce_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return optax.sigmoid_binary_cross_entropy(logits, y)


def _make_state(
    rng: jax.Array, model: TabularEmbedMLP, learning_rate: float, weight_decay: float
) -> TrainState:
    params_rng, dropout_rng = jax.random.split(rng)
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.zeros((2, 0), dtype=jnp.int32),
        jnp.zeros((2, 0), dtype=jnp.float32),
        train=True,
    )
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, dropout_rng=dropout_rng)


def _init_state(rng: jax.Array, cfg: ModelConfig, n_cat: int) -> tuple[TabularEmbedMLP, TrainState]:
    model = TabularEmbedMLP(cfg)
    params_rng, dropout_rng = jax.random.split(rng)
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.zeros((2, n_cat), dtype=jnp.int32),
        jnp.zeros((2, cfg.num_features), dtype=jnp.float32),
        train=True,
    )
    tx = optax.adamw(learning_rate=2e-3, weight_decay=1e-4)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, dropout_rng=dropout_rng)
    return model, state


def compute_adv_weights(
    x_cat_train: np.ndarray,
    x_num_train: np.ndarray,
    x_cat_test: np.ndarray,
    x_num_test: np.ndarray,
    *,
    cat_cardinalities: list[int],
    adv_cfg: AdvConfig,
) -> np.ndarray:
    rng = np.random.default_rng(adv_cfg.seed)

    n_train = x_cat_train.shape[0]
    n_test = x_cat_test.shape[0]

    # Subsample for speed (balanced)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_test)
    if n_train > adv_cfg.max_rows:
        train_idx = rng.choice(train_idx, size=adv_cfg.max_rows, replace=False)
    if n_test > adv_cfg.max_rows:
        test_idx = rng.choice(test_idx, size=adv_cfg.max_rows, replace=False)

    x_cat = np.concatenate([x_cat_train[train_idx], x_cat_test[test_idx]], axis=0)
    x_num = np.concatenate([x_num_train[train_idx], x_num_test[test_idx]], axis=0)
    y = np.concatenate(
        [np.zeros((len(train_idx),), dtype=np.float32), np.ones((len(test_idx),), dtype=np.float32)],
        axis=0,
    )

    # Shuffle
    perm = rng.permutation(len(y))
    x_cat = x_cat[perm]
    x_num = x_num[perm]
    y = y[perm]

    cfg = ModelConfig(
        cat_cardinalities=cat_cardinalities,
        num_features=x_num.shape[1],
        width=adv_cfg.width,
        blocks=adv_cfg.blocks,
        dropout=adv_cfg.dropout,
        embed_dim=adv_cfg.embed_dim,
    )

    jax_rng = jax.random.PRNGKey(adv_cfg.seed)
    model = TabularEmbedMLP(cfg)
    params_rng, dropout_rng = jax.random.split(jax_rng)
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.zeros((2, x_cat.shape[1]), dtype=jnp.int32),
        jnp.zeros((2, x_num.shape[1]), dtype=jnp.float32),
        train=True,
    )
    tx = optax.adamw(learning_rate=adv_cfg.lr, weight_decay=adv_cfg.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, dropout_rng=dropout_rng)

    @jax.jit
    def train_step(state: TrainState, x_cat_b, x_num_b, y_b):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, x_cat_b, x_num_b, train=True, rngs={"dropout": dropout_rng}
            )
            loss = jnp.mean(_bce_logits(logits, y_b))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(dropout_rng=new_dropout_rng)
        return state, loss

    # Train
    for _ in range(int(adv_cfg.epochs)):
        for batch in iter_minibatches(
            x_cat, x_num, y, None, batch_size=adv_cfg.batch_size, rng=rng, shuffle=True
        ):
            state, _ = train_step(
                state,
                jnp.asarray(batch.x_cat),
                jnp.asarray(batch.x_num),
                jnp.asarray(batch.y),
            )

    # Predict p(test|x) on full train
    @jax.jit
    def pred_logits(params, x_cat_b, x_num_b):
        return model.apply({"params": params}, x_cat_b, x_num_b, train=False)

    probs = np.zeros((n_train,), dtype=np.float32)
    bs = adv_cfg.batch_size
    for start in range(0, n_train, bs):
        sl = slice(start, start + bs)
        logits = pred_logits(state.params, jnp.asarray(x_cat_train[sl]), jnp.asarray(x_num_train[sl]))
        p = jax.nn.sigmoid(logits)
        probs[sl] = np.asarray(p, dtype=np.float32)

    eps = 1e-4
    probs = np.clip(probs, eps, 1.0 - eps)

    # importance weights ~ p(test|x) / p(train|x)
    w = probs / (1.0 - probs)
    w = np.clip(w, adv_cfg.clip_min, adv_cfg.clip_max).astype(np.float32)

    # Normalize to mean 1 (keeps loss scale stable)
    w = (w / np.mean(w)).astype(np.float32)
    return w
