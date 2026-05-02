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
"""DSA (Dynamic Sparse Attention) Indexer for DeepSeek V3.2.

Ported from MaxText's attention_mla.py Indexer class (inference-only, no backward).
Reference: https://arxiv.org/pdf/2512.02556
"""

from dataclasses import InitVar, dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxLayerNorm
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.layers.jax.rope import RotaryEmbedding
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(kw_only=True)
class DSAIndexer(JaxModule):
    """Lightning Indexer for Dynamic Sparse Attention.

    Computes relevance scores to select top-K KV positions for sparse attention.
    Each layer independently selects which KV tokens each query should attend to.

    Config: indexer_n_heads=64, indexer_head_dim=128, indexer_topk=2048
    """
    n_heads: int
    head_dim: int
    q_lora_rank: int
    emb_dim: int
    qk_rope_head_dim: int
    topk: int
    dtype: jnp.dtype
    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""

    rngs: InitVar[nnx.Rngs]

    def __post_init__(self, rngs: nnx.Rngs):
        weight_init = nnx.initializers.lecun_normal()

        # Q projection: q_lora_rank → n_heads * head_dim (FP8 in V3.2 weights)
        self.wq_b = JaxEinsum(
            einsum_str="TA,AO->TO",
            kernel_shape=(self.q_lora_rank, self.n_heads * self.head_dim),
            rngs=rngs,
            param_dtype=self.dtype,
            kernel_init=weight_init,
            quant_config=self.quant_config,
            prefix=self.prefix + ".wq_b",
        )

        # K projection: emb_dim → head_dim, MQA-style (FP8 in V3.2 weights)
        self.wk = JaxEinsum(
            einsum_str="SD,DH->SH",
            kernel_shape=(self.emb_dim, self.head_dim),
            rngs=rngs,
            param_dtype=self.dtype,
            kernel_init=weight_init,
            quant_config=self.quant_config,
            prefix=self.prefix + ".wk",
        )

        # K normalization: LayerNorm with bias (NOT RMSNorm)
        self.k_norm = JaxLayerNorm(
            num_features=self.head_dim,
            use_bias=True,
            param_dtype=self.dtype,
            prefix=self.prefix + ".k_norm",
            rngs=rngs,
        )

        # Head importance projection: emb_dim → n_heads, BF16 (no FP8)
        self.weights_proj = JaxEinsum(
            einsum_str="SD,DN->SN",
            kernel_shape=(self.emb_dim, self.n_heads),
            rngs=rngs,
            param_dtype=jnp.float32,
            kernel_init=weight_init,
            prefix=self.prefix + ".weights_proj",
        )

        self.softmax_scale = self.head_dim ** -0.5

    def _apply_partial_rope(
        self,
        x: jax.Array,
        positions: jax.Array,
        rope: RotaryEmbedding,
    ) -> jax.Array:
        """Apply RoPE to first qk_rope_head_dim dims, passthrough the rest.

        Indexer splits as [rope, nope] (opposite of MLA's [nope, rope]).
        Uses concatenate layout (same as tpu-inference's RotaryEmbedding).
        """
        rope_dim = self.qk_rope_head_dim
        x_pe = x[..., :rope_dim]
        x_nope = x[..., rope_dim:]
        x_pe = rope.apply_rope(positions, x_pe)
        return jnp.concatenate([x_pe, x_nope], axis=-1)

    def compute_k(
        self,
        hidden_states: jax.Array,
        positions: jax.Array,
        rope: RotaryEmbedding,
    ) -> jax.Array:
        """Compute indexer K for given tokens (for cache storage).

        Args:
            hidden_states: [T, D] — layer input.
            positions: [T] — token positions.
            rope: RotaryEmbedding instance.

        Returns:
            [T, head_dim] — projected + normed + RoPE'd K.
        """
        k = self.wk(hidden_states)  # [T, head_dim]
        k = self.k_norm(k)
        k = k[:, None, :]  # [T, 1, head_dim] — add head dim for partial_rope
        k = self._apply_partial_rope(k, positions, rope)
        k = k[:, 0, :]  # [T, head_dim] — remove head dim
        return k

    def _compute_q(
        self,
        q_compressed: jax.Array,
        positions: jax.Array,
        rope: RotaryEmbedding,
    ) -> jax.Array:
        """Compute indexer Q from compressed query.

        Args:
            q_compressed: [T, q_lora_rank] — compressed query.
            positions: [T] — token positions.
            rope: RotaryEmbedding instance.

        Returns:
            [T, n_heads, head_dim] — projected + RoPE'd Q.
        """
        seq_len = q_compressed.shape[0]
        q = self.wq_b(q_compressed)  # [T, n_heads * head_dim]
        q = q.reshape(seq_len, self.n_heads, self.head_dim)  # [T, H, D]
        q = self._apply_partial_rope(q, positions, rope)
        return q

    def _score_and_topk(
        self,
        q: jax.Array,
        k: jax.Array,
        hidden_states: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Score Q against K and select top-K indices.

        Args:
            q: [T, n_heads, head_dim] — indexer Q.
            k: [S, head_dim] — indexer K (all history).
            hidden_states: [T, D] — for weights_proj (head importance).

        Returns:
            (topk_indices, indexer_score):
                topk_indices: [T, topk]
                indexer_score: [T, S]
        """
        # QK similarity: relu(Q @ K.T), MQA-style
        logits = jnp.einsum("THD,SD->TSH", q, k)  # [T, S, H]
        logits = jax.nn.relu(logits)

        # Head importance weights (FP32 for stability)
        weights = self.weights_proj(hidden_states.astype(jnp.float32))  # [T, H]
        weights = weights * (self.n_heads ** -0.5) * self.softmax_scale

        # Aggregate across heads
        indexer_score = jnp.einsum("TSH,TH->TS", logits, weights)  # [T, S]

        # Top-K selection
        _, topk_indices = jax.lax.top_k(indexer_score, k=self.topk)
        return topk_indices, indexer_score

    def __call__(
        self,
        hidden_states: jax.Array,
        q_compressed: jax.Array,
        positions: jax.Array,
        rope: RotaryEmbedding,
    ) -> Tuple[Optional[jax.Array], Optional[jax.Array]]:
        """Compute top-K KV indices for sparse attention (prefill mode).

        All tokens' hidden_states are available. Computes K from hidden_states
        directly (no cache needed for prefill).

        Args:
            hidden_states: [T, D] — current layer input (all tokens).
            q_compressed: [T, q_lora_rank] — compressed query.
            positions: [T] — token positions.
            rope: RotaryEmbedding instance.

        Returns:
            (topk_indices, indexer_score) or (None, None) if seq_len <= topk.
        """
        seq_len = hidden_states.shape[0]
        if seq_len <= self.topk:
            return None, None

        q = self._compute_q(q_compressed, positions, rope)
        k = self.compute_k(hidden_states, positions, rope)
        return self._score_and_topk(q, k, hidden_states)

    def forward_decode(
        self,
        hidden_states: jax.Array,
        q_compressed: jax.Array,
        positions: jax.Array,
        rope: RotaryEmbedding,
        indexer_k_cache: jax.Array,
        seq_len: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Compute top-K indices for decode (single token, uses cache).

        JIT-compatible: scores against the full cache (static shape) and masks
        invalid positions to -inf. When seq_len <= topk, top-K returns all
        valid positions plus some masked ones (harmless for attention since
        the gathered KV at masked positions is zero and gets near-zero weight
        after softmax).

        Args:
            hidden_states: [1, D] — current token only.
            q_compressed: [1, q_lora_rank] — compressed query for current token.
            positions: [1] — current token position.
            rope: RotaryEmbedding instance.
            indexer_k_cache: [max_seq_len, head_dim] — cached historical K.
            seq_len: scalar — total sequence length including current token
                (may be a traced JAX value).

        Returns:
            (topk_indices, indexer_score, updated_cache):
                topk_indices: [1, topk] — always returned (no None).
                indexer_score: [1, max_seq_len] — scores with -inf for
                    invalid positions.
                updated_cache: [max_seq_len, head_dim].
        """
        cur_k = self.compute_k(hidden_states, positions, rope)  # [1, head_dim]
        indexer_k_cache = indexer_k_cache.at[positions[0]].set(cur_k[0])

        q = self._compute_q(q_compressed, positions, rope)  # [1, H, D]

        # Score against full cache (static shape for JIT)
        logits = jnp.einsum("THD,SD->TSH", q, indexer_k_cache)
        logits = jax.nn.relu(logits)

        weights = self.weights_proj(hidden_states.astype(jnp.float32))
        weights = weights * (self.n_heads ** -0.5) * self.softmax_scale
        indexer_score = jnp.einsum("TSH,TH->TS", logits, weights)

        # Mask invalid positions (>= seq_len)
        max_seq_len = indexer_k_cache.shape[0]
        valid_mask = jnp.arange(max_seq_len) < seq_len
        indexer_score = jnp.where(
            valid_mask[None, :], indexer_score,
            jnp.finfo(jnp.float32).min)

        _, topk_indices = jax.lax.top_k(indexer_score, k=self.topk)

        return topk_indices, indexer_score, indexer_k_cache
