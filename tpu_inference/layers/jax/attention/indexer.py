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

    def __call__(
        self,
        hidden_states: jax.Array,
        q_compressed: jax.Array,
        positions: jax.Array,
        rope: RotaryEmbedding,
    ) -> Tuple[Optional[jax.Array], Optional[jax.Array]]:
        """Compute top-K KV indices for sparse attention.

        Args:
            hidden_states: [T, D] — current layer input.
            q_compressed: [T, q_lora_rank] — compressed query (after q_a_layernorm).
            positions: [T] — token positions.
            rope: RotaryEmbedding instance for partial RoPE.

        Returns:
            (topk_indices, indexer_score) or (None, None) if seq_len <= topk.
            topk_indices: [T, topk] — selected KV position indices.
            indexer_score: [T, S] — per-position relevance scores.
        """
        seq_len = hidden_states.shape[0]

        if seq_len <= self.topk:
            return None, None

        # Q: project from compressed query, reshape to [T, H, D], apply partial RoPE
        q = self.wq_b(q_compressed)  # [T, H*D]
        q = q.reshape(seq_len, self.n_heads, self.head_dim)  # [T, H, D]
        q = self._apply_partial_rope(q, positions, rope)

        # K: project from hidden states, normalize, apply partial RoPE
        k = self.wk(hidden_states)  # [T, D]
        k = self.k_norm(k)
        k = k[:, None, :]  # [T, 1, D] — add head dim for partial_rope
        k = self._apply_partial_rope(k, positions, rope)
        k = k[:, 0, :]  # [T, D] — remove head dim

        # QK similarity: relu(Q @ K.T), MQA-style (K shared across heads)
        logits = jnp.einsum("THD,SD->TSH", q, k)  # [T, S, H]
        logits = jax.nn.relu(logits)

        # Head importance weights (FP32 for stability)
        weights = self.weights_proj(hidden_states.astype(jnp.float32))  # [T, H]
        weights = weights * (self.n_heads ** -0.5) * self.softmax_scale

        # Aggregate across heads: [T, S, H] @ [T, H] → [T, S]
        # Broadcasting: weights[T, H] → weights[T, 1, H] for batch matmul
        indexer_score = jnp.einsum("TSH,TH->TS", logits, weights)

        # Top-K selection
        _, topk_indices = jax.lax.top_k(indexer_score, k=self.topk)  # [T, topk]

        return topk_indices, indexer_score
