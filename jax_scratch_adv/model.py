from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ModelConfig:
    cat_cardinalities: Sequence[int]
    num_features: int
    width: int = 256
    blocks: int = 4
    dropout: float = 0.1
    embed_dim: int = 16
    norm_kind: str | None = 'derf'

class Derf(nn.Module):
        alpha_init: float = 0.2357
        s_init: float = 0.001

        @nn.compact
        def __call__(self, x):
            d = x.shape[-1]
            gamma = self.param("gamma", lambda k: jnp.ones((d,), dtype=jnp.float32))
            beta = self.param("beta", lambda k: jnp.zeros((d,), dtype=jnp.float32))
            alpha = self.param("alpha", lambda k: jnp.array(self.alpha_init, dtype=jnp.float32))
            s = self.param("s", lambda k: jnp.array(self.s_init, dtype=jnp.float32))
            return gamma * jax.lax.erf(alpha * x + s) + beta


def make_norm(norm_kind: str | None = 'layernorm') -> nn.Module | None:
        if norm_kind == "layernorm":
            return nn.LayerNorm(use_bias=True, use_scale=True)
        if norm_kind == "derf":
            return Derf()
        return None


class ResidualBlock(nn.Module):
    width: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, norm_kind: str | None = 'layernorm', train: bool) -> jnp.ndarray:
        h = make_norm(norm_kind)(x)
        h = nn.Dense(self.width)(h)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Dense(self.width)(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        return x + h


class TabularEmbedMLP(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(
        self, x_cat: jnp.ndarray, x_num: jnp.ndarray, *, train: bool
    ) -> jnp.ndarray:
        parts: List[jnp.ndarray] = []

        if self.cfg.cat_cardinalities:
            for i, card in enumerate(self.cfg.cat_cardinalities):
                emb = nn.Embed(num_embeddings=int(card), features=int(self.cfg.embed_dim))
                parts.append(emb(x_cat[:, i]))

        if self.cfg.num_features > 0:
            parts.append(x_num)

        if not parts:
            raise ValueError("No input features provided")

        x = jnp.concatenate(parts, axis=1)
        x = nn.Dense(self.cfg.width)(x)
        x = nn.gelu(x)

        for _ in range(int(self.cfg.blocks)):
            x = ResidualBlock(width=self.cfg.width,  dropout=self.cfg.dropout)(x, norm_kind=self.cfg.norm_kind, train=train)

        x = make_norm(self.cfg.norm_kind)(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)  # logits


def init_model(rng: jax.Array, cfg: ModelConfig, cat_cols: int) -> tuple[TabularEmbedMLP, dict]:
    model = TabularEmbedMLP(cfg)
    dummy_cat = jnp.zeros((2, cat_cols), dtype=jnp.int32)
    dummy_num = jnp.zeros((2, cfg.num_features), dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy_cat, dummy_num, train=True)
    return model, variables
