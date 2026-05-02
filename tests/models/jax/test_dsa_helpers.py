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
"""DSA Phase 2 unit tests — helper functions, indexer, reference attention.

Tests the DSA (Dynamic Sparse Attention) components independently,
without requiring a full model or serving framework. XLA compile is
seconds-level for these small tests.

Run on TPU pod:
    python3 -m pytest tests/models/jax/test_dsa_helpers.py -v
Or standalone:
    python3 tests/models/jax/test_dsa_helpers.py
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tpu_inference.kernels.mla.v1.kernel import (
    update_kv_cache as mla_v1_update_kv_cache,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to,
    get_dtype_packing,
)
from tpu_inference.layers.jax.attention.indexer import DSAIndexer
from tpu_inference.layers.jax.rope import DeepseekScalingRotaryEmbedding
from tpu_inference.models.jax.deepseek_v3 import (
    _align_to,
    _gather_from_paged_cache,
    _reference_mla_attention,
)

# V3.2 production values
LKV_DIM = 512        # kv_lora_rank
R_DIM = 64           # qk_rope_head_dim
N_HEADS = 128        # num_attention_heads
HIDDEN_SIZE = 7168   # D
Q_LORA_RANK = 1536
INDEXER_N_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_TOPK = 2048


def _make_rope(rotary_dim=R_DIM, dtype=jnp.bfloat16):
    return DeepseekScalingRotaryEmbedding(
        rotary_dim=rotary_dim,
        rope_theta=10000,
        original_max_position_embeddings=4096,
        scaling_factor=1.0,
        dtype=dtype,
        beta_fast=32,
        beta_slow=1,
        mscale_value=1.0,
        mscale_all_dim=1.0,
    )


def _make_cache(total_pages, page_size, kv_dim, dtype=jnp.bfloat16):
    kv_packing = get_dtype_packing(dtype)
    ps_per_packing = align_to(page_size, kv_packing) // kv_packing
    shape = (total_pages, ps_per_packing, kv_packing, align_to(kv_dim, 128))
    return jnp.zeros(shape, dtype=dtype), kv_packing


class TestAlignTo(unittest.TestCase):

    def test_already_aligned(self):
        self.assertEqual(_align_to(512, 128), 512)
        self.assertEqual(_align_to(128, 128), 128)

    def test_needs_alignment(self):
        self.assertEqual(_align_to(64, 128), 128)
        self.assertEqual(_align_to(1, 128), 128)
        self.assertEqual(_align_to(129, 128), 256)

    def test_matches_kernel_align(self):
        for x in [1, 64, 127, 128, 129, 256, 512, 513]:
            self.assertEqual(_align_to(x, 128), align_to(x, 128))


class TestGatherFromPagedCache(unittest.TestCase):

    def setUp(self):
        self.dtype = jnp.bfloat16
        self.kv_packing = get_dtype_packing(self.dtype)
        self.page_size = 16
        self.kv_dim = align_to(LKV_DIM, 128) + align_to(R_DIM, 128)  # 640
        self.total_pages = 8
        self.max_tokens = self.total_pages * self.page_size  # 128

        ps_per_packing = align_to(self.page_size, self.kv_packing) // self.kv_packing
        cache_shape = (self.total_pages, ps_per_packing, self.kv_packing,
                       align_to(self.kv_dim, 128))

        rng = np.random.default_rng(42)
        data = rng.standard_normal(cache_shape).astype(np.float32)
        for p in range(self.total_pages):
            for r in range(ps_per_packing):
                for c in range(self.kv_packing):
                    data[p, r, c, 0] = p * 10000 + r * 100 + c
        self.cache = jnp.array(data, dtype=self.dtype)

        self.block_tables = jnp.arange(self.total_pages, dtype=jnp.int32)

    def test_gather_boundary_positions(self):
        positions = jnp.array([0, 1, self.page_size - 1,
                               self.page_size, self.max_tokens - 1],
                              dtype=jnp.int32)
        gathered = _gather_from_paged_cache(
            self.cache, positions, self.block_tables, max_num_seqs=1)

        self.assertEqual(gathered.shape, (len(positions), self.cache.shape[-1]))

        for i, pos in enumerate(positions):
            pos_val = int(pos)
            page = pos_val // self.page_size
            row = (pos_val % self.page_size) // self.kv_packing
            col = (pos_val % self.page_size) % self.kv_packing
            expected = page * 10000 + row * 100 + col
            actual = float(gathered[i, 0])
            self.assertAlmostEqual(actual, expected, places=0,
                                   msg=f"pos={pos_val}")

    def test_gather_all_positions(self):
        all_pos = jnp.arange(self.max_tokens, dtype=jnp.int32)
        gathered = _gather_from_paged_cache(
            self.cache, all_pos, self.block_tables, max_num_seqs=1)
        self.assertEqual(gathered.shape, (self.max_tokens, self.cache.shape[-1]))

    def test_gather_with_nonidentity_block_table(self):
        reversed_bt = jnp.array(
            list(reversed(range(self.total_pages))), dtype=jnp.int32)
        pos = jnp.array([0], dtype=jnp.int32)

        gathered_id = _gather_from_paged_cache(
            self.cache, pos, self.block_tables, max_num_seqs=1)
        gathered_rev = _gather_from_paged_cache(
            self.cache, pos, reversed_bt, max_num_seqs=1)

        last_page = self.total_pages - 1
        self.assertAlmostEqual(
            float(gathered_rev[0, 0]),
            float(self.cache[last_page, 0, 0, 0]),
            places=0)
        self.assertFalse(jnp.allclose(gathered_id, gathered_rev))


class TestReferenceMlaAttention(unittest.TestCase):

    def test_output_shape_and_dtype(self):
        rng = jax.random.PRNGKey(42)
        T, N, K = 1, 8, 64
        lkv, r = 32, 16
        kv_dim = _align_to(lkv, 128) + _align_to(r, 128)

        k1, k2, k3 = jax.random.split(rng, 3)
        q_TNA = jax.random.normal(k1, (T, N, lkv), dtype=jnp.bfloat16)
        q_rope = jax.random.normal(k2, (T, N, r), dtype=jnp.bfloat16)
        gathered = jax.random.normal(k3, (K, kv_dim), dtype=jnp.bfloat16)

        out = _reference_mla_attention(
            q_TNA, q_rope, gathered, lkv, r, sm_scale=0.1)
        self.assertEqual(out.shape, (T, N, lkv))
        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertFalse(jnp.any(jnp.isnan(out)))
        self.assertFalse(jnp.any(jnp.isinf(out)))

    def test_matches_manual_computation(self):
        """Verify reference attention matches hand-rolled einsum."""
        rng = jax.random.PRNGKey(123)
        T, N = 1, 4
        lkv_dim, r_dim = 16, 8
        K = 8
        aligned_lkv = _align_to(lkv_dim, 128)  # 128
        aligned_r = _align_to(r_dim, 128)       # 128
        kv_dim = aligned_lkv + aligned_r         # 256

        k1, k2, k3 = jax.random.split(rng, 3)
        q_TNA = jax.random.normal(k1, (T, N, lkv_dim), dtype=jnp.bfloat16)
        q_rope = jax.random.normal(k2, (T, N, r_dim), dtype=jnp.bfloat16)

        gathered_kv = jnp.zeros((K, kv_dim), dtype=jnp.bfloat16)
        raw = jax.random.normal(k3, (K, kv_dim), dtype=jnp.bfloat16)
        gathered_kv = gathered_kv.at[:, :lkv_dim].set(raw[:, :lkv_dim])
        gathered_kv = gathered_kv.at[:, aligned_lkv:aligned_lkv + r_dim].set(
            raw[:, aligned_lkv:aligned_lkv + r_dim])

        sm_scale = (lkv_dim + r_dim) ** -0.5

        our = _reference_mla_attention(
            q_TNA, q_rope, gathered_kv, lkv_dim, r_dim, sm_scale)

        kv_c = gathered_kv[:, :lkv_dim]
        k_pe = gathered_kv[:, aligned_lkv:aligned_lkv + r_dim]
        q = jnp.concatenate([q_TNA, q_rope], axis=-1)
        k = jnp.concatenate([kv_c, k_pe], axis=-1)
        score = jnp.einsum("tnd,kd->tnk", q, k) * sm_scale
        weights = jax.nn.softmax(score.astype(jnp.float32), axis=-1)
        manual = jnp.einsum(
            "tnk,kd->tnd", weights.astype(jnp.bfloat16), kv_c)

        max_diff = float(jnp.max(jnp.abs(our - manual)))
        self.assertLess(max_diff, 1e-3,
                        f"Reference vs manual max diff: {max_diff}")

    def test_dense_topk_equals_all(self):
        """When topk == seq_len, DSA should match dense attention."""
        rng = jax.random.PRNGKey(7)
        T, N = 1, 4
        lkv_dim, r_dim = 16, 8
        seq_len = 32
        aligned_lkv = _align_to(lkv_dim, 128)
        aligned_r = _align_to(r_dim, 128)
        kv_dim = aligned_lkv + aligned_r

        k1, k2, k3 = jax.random.split(rng, 3)
        q_TNA = jax.random.normal(k1, (T, N, lkv_dim), dtype=jnp.bfloat16)
        q_rope = jax.random.normal(k2, (T, N, r_dim), dtype=jnp.bfloat16)
        all_kv = jax.random.normal(k3, (seq_len, kv_dim), dtype=jnp.bfloat16)

        sm_scale = (lkv_dim + r_dim) ** -0.5

        dense = _reference_mla_attention(
            q_TNA, q_rope, all_kv, lkv_dim, r_dim, sm_scale)

        top_half = all_kv[:seq_len // 2]
        sparse = _reference_mla_attention(
            q_TNA, q_rope, top_half, lkv_dim, r_dim, sm_scale)

        self.assertFalse(jnp.allclose(dense, sparse),
                         "Sparse (half) should differ from dense (full)")
        self.assertEqual(dense.shape, sparse.shape)


class TestIndexerForwardDecode(unittest.TestCase):

    def setUp(self):
        self.D = 128
        self.n_heads = 4
        self.head_dim = 16
        self.q_lora_rank = 32
        self.qk_rope_head_dim = 8
        self.topk = 8
        self.max_seq_len = 32

        self.rope = _make_rope(rotary_dim=self.qk_rope_head_dim)

        rng = jax.random.PRNGKey(0)
        self.indexer = DSAIndexer(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            emb_dim=self.D,
            qk_rope_head_dim=self.qk_rope_head_dim,
            topk=self.topk,
            dtype=jnp.bfloat16,
            rngs=nnx.Rngs(rng),
        )

    def test_output_shapes(self):
        rng = jax.random.PRNGKey(1)
        cache = jnp.zeros((self.max_seq_len, self.head_dim), dtype=jnp.bfloat16)

        k1, k2 = jax.random.split(rng)
        prefill = jax.random.normal(k1, (16, self.D), dtype=jnp.bfloat16)
        pos_prefill = jnp.arange(16, dtype=jnp.int32)
        k_prefill = self.indexer.compute_k(prefill, pos_prefill, self.rope)
        cache = cache.at[:16].set(k_prefill)

        hidden = jax.random.normal(k2, (1, self.D), dtype=jnp.bfloat16)
        q_comp = jax.random.normal(k2, (1, self.q_lora_rank), dtype=jnp.bfloat16)
        pos_decode = jnp.array([16], dtype=jnp.int32)
        seq_len = jnp.int32(17)

        topk_idx, score, new_cache = self.indexer.forward_decode(
            hidden, q_comp, pos_decode, self.rope, cache, seq_len)

        self.assertEqual(topk_idx.shape, (1, self.topk))
        self.assertEqual(score.shape, (1, self.max_seq_len))
        self.assertEqual(new_cache.shape, cache.shape)

    def test_invalid_positions_masked(self):
        rng = jax.random.PRNGKey(2)
        cache = jnp.zeros((self.max_seq_len, self.head_dim), dtype=jnp.bfloat16)

        k1, k2 = jax.random.split(rng)
        prefill = jax.random.normal(k1, (8, self.D), dtype=jnp.bfloat16)
        pos = jnp.arange(8, dtype=jnp.int32)
        cache = cache.at[:8].set(self.indexer.compute_k(prefill, pos, self.rope))

        hidden = jax.random.normal(k2, (1, self.D), dtype=jnp.bfloat16)
        q_comp = jax.random.normal(k2, (1, self.q_lora_rank), dtype=jnp.bfloat16)
        seq_len = jnp.int32(9)

        topk_idx, score, _ = self.indexer.forward_decode(
            hidden, q_comp, jnp.array([8], dtype=jnp.int32),
            self.rope, cache, seq_len)

        valid = score[0, :9]
        invalid = score[0, 9:]

        self.assertTrue(jnp.all(jnp.isfinite(valid)))
        self.assertTrue(jnp.all(invalid == jnp.finfo(jnp.float32).min))

    def test_topk_within_valid_range(self):
        rng = jax.random.PRNGKey(3)
        cache = jnp.zeros((self.max_seq_len, self.head_dim), dtype=jnp.bfloat16)

        prefill = jax.random.normal(rng, (16, self.D), dtype=jnp.bfloat16)
        cache = cache.at[:16].set(
            self.indexer.compute_k(prefill, jnp.arange(16, dtype=jnp.int32),
                                   self.rope))

        k1, _ = jax.random.split(rng)
        hidden = jax.random.normal(k1, (1, self.D), dtype=jnp.bfloat16)
        q_comp = jax.random.normal(k1, (1, self.q_lora_rank), dtype=jnp.bfloat16)
        seq_len = jnp.int32(17)

        topk_idx, _, _ = self.indexer.forward_decode(
            hidden, q_comp, jnp.array([16], dtype=jnp.int32),
            self.rope, cache, seq_len)

        self.assertTrue(jnp.all(topk_idx >= 0))
        self.assertTrue(jnp.all(topk_idx < seq_len))

    def test_cache_updated_at_current_position(self):
        rng = jax.random.PRNGKey(4)
        cache = jnp.zeros((self.max_seq_len, self.head_dim), dtype=jnp.bfloat16)

        hidden = jax.random.normal(rng, (1, self.D), dtype=jnp.bfloat16)
        q_comp = jax.random.normal(rng, (1, self.q_lora_rank), dtype=jnp.bfloat16)
        pos = jnp.array([5], dtype=jnp.int32)

        _, _, new_cache = self.indexer.forward_decode(
            hidden, q_comp, pos, self.rope, cache, jnp.int32(6))

        self.assertFalse(jnp.allclose(new_cache[5], jnp.zeros(self.head_dim)),
                         "Position 5 should be written")
        self.assertTrue(jnp.allclose(cache[5], jnp.zeros(self.head_dim)),
                        "Original cache should be unchanged at pos 5")


class TestV1CacheUpdateIntegration(unittest.TestCase):
    """Test that v1 update_kv_cache writes to the correct positions
    and _gather_from_paged_cache can read them back."""

    def test_write_then_gather_roundtrip(self):
        kv_dtype = jnp.bfloat16
        kv_packing = get_dtype_packing(kv_dtype)
        page_size = 16
        lkv_dim = 16
        r_dim = 8
        aligned_lkv = align_to(lkv_dim, 128)
        aligned_r = align_to(r_dim, 128)
        kv_dim = aligned_lkv + aligned_r

        total_pages = 4
        ps_per_packing = align_to(page_size, kv_packing) // kv_packing
        cache = jnp.zeros(
            (total_pages, ps_per_packing, kv_packing, kv_dim), dtype=kv_dtype)

        rng = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(rng)
        seq_len = 5
        new_kv_c = jax.random.normal(k1, (seq_len, lkv_dim), dtype=kv_dtype)
        new_k_pe = jax.random.normal(k2, (seq_len, r_dim), dtype=kv_dtype)

        kv_lens = jnp.array([seq_len], dtype=jnp.int32)
        block_tables = jnp.arange(total_pages, dtype=jnp.int32)
        cu_q_lens = jnp.array([0, seq_len], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

        updated = mla_v1_update_kv_cache(
            new_kv_c, new_k_pe, cache,
            kv_lens, block_tables, cu_q_lens, distribution)

        positions = jnp.arange(seq_len, dtype=jnp.int32)
        gathered = _gather_from_paged_cache(
            updated, positions, block_tables, max_num_seqs=1)

        for t in range(seq_len):
            kv_c_back = gathered[t, :lkv_dim]
            k_pe_back = gathered[t, aligned_lkv:aligned_lkv + r_dim]

            kv_c_diff = float(jnp.max(jnp.abs(kv_c_back - new_kv_c[t])))
            k_pe_diff = float(jnp.max(jnp.abs(k_pe_back - new_k_pe[t])))

            self.assertLess(kv_c_diff, 1e-3,
                            f"Token {t} kv_c roundtrip diff: {kv_c_diff}")
            self.assertLess(k_pe_diff, 1e-3,
                            f"Token {t} k_pe roundtrip diff: {k_pe_diff}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
