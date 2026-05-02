# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Indexer K cache for DSA (Dynamic Sparse Attention) decode.

During decode, the indexer needs to score the current Q against ALL historical K
vectors, but only the current token's hidden_states is available. This cache
stores the projected+normed+RoPE'd indexer K for each historical token.

Shape: [max_seq_len, indexer_head_dim] per sequence per layer.
Memory: 128K × 128 × 2B (BF16) × 61 layers = ~1.9 GB for v7x-8.
"""

import jax
import jax.numpy as jnp


def create_indexer_cache(
    max_seq_len: int,
    indexer_head_dim: int,
    num_layers: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> list[jax.Array]:
    """Allocate flat indexer K cache for each layer.

    Args:
        max_seq_len: Maximum sequence length (e.g. 131072 for 128K).
        indexer_head_dim: Indexer K dimension (128 for DeepSeek V3.2).
        num_layers: Number of transformer layers.
        dtype: Cache dtype (BF16 default).

    Returns:
        List of [max_seq_len, indexer_head_dim] arrays, one per layer.
    """
    shape = (max_seq_len, indexer_head_dim)
    return [jnp.zeros(shape, dtype=dtype) for _ in range(num_layers)]


def update_indexer_cache_prefill(
    cache: jax.Array,
    indexer_k: jax.Array,
    seq_len: int,
) -> jax.Array:
    """Batch-fill indexer cache during prefill.

    Args:
        cache: [max_seq_len, D] — existing indexer cache for this layer.
        indexer_k: [seq_len, D] — projected+normed+RoPE'd K for all prefill tokens.
        seq_len: Number of tokens to write.

    Returns:
        Updated cache with indexer_k written at positions [0:seq_len].
    """
    return cache.at[:seq_len].set(indexer_k[:seq_len])


def update_indexer_cache_decode(
    cache: jax.Array,
    indexer_k: jax.Array,
    position: jax.Array,
) -> jax.Array:
    """Append a single token's indexer K during decode.

    Args:
        cache: [max_seq_len, D] — existing indexer cache for this layer.
        indexer_k: [1, D] or [D] — current token's projected+normed+RoPE'd K.
        position: scalar int — the sequence position to write at.

    Returns:
        Updated cache with indexer_k written at the given position.
    """
    k = indexer_k.reshape(-1)  # [D]
    return cache.at[position].set(k)


def gather_indexer_k(
    cache: jax.Array,
    seq_len: int,
) -> jax.Array:
    """Read all historical indexer K from cache.

    Args:
        cache: [max_seq_len, D] — indexer cache for this layer.
        seq_len: Number of valid tokens in the cache.

    Returns:
        [seq_len, D] — all historical indexer K vectors.
    """
    return jax.lax.dynamic_slice(cache, (0, 0), (seq_len, cache.shape[1]))
