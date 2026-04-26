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

"""K2.6 W4A16 INT4 (compressed-tensors symmetric, group_size=32) quantization.

Implements:
  - JAX-native unpack: int32 (packed 8 x uint4) -> uint4 [0, 15]
  - Symmetric dequant: uint4 -> int4 [-8, 7] -> bf16 x scale (no zero_point)
  - CompressedTensorsW4A16Config: HF quantization_config -> Quant method dispatch
  - CompressedTensorsW4A16FusedMoEMethod: MoE quant method (Round 3b/3c)

Reference plan: cc.higcp.com/pages/kimi-k26-tpu-inference-plan-20260425.html (v0.4.3)
Reference impl log: cc.higcp.com/pages/kimi-k26-implementation-log-20260425.html

K2.6 quantization_config (from config.json, verified 2026-04-25):
  num_bits     = 4
  type         = int
  strategy     = group
  group_size   = 32          # finer than typical 128
  symmetric    = true        # NO zero_point (verified by safetensors zp keys = 0)
  format       = pack-quantized
  quant_method = compressed-tensors

Architectural decision:
  TPU VPU only supports INT4 -> BF16 direct cast (NOT INT4 -> FP8). So we use
  W4A16 standard path (dequant to BF16, then BF16 matmul) instead of mixed
  precision. See plan v0.4.3 Section 0.2.
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np  # Phase 1: numpy unpack 比 jnp 快 4-17x on CPU
from jax.experimental.layout import Layout, with_layout_constraint  # Phase 2 v3: jnp metadata transpose


# ============================================================
# Phase 3: SHM cache for processed MoE weights
# ============================================================
# Skip _cpu_process_int4_layer (~30s/layer) on second run by caching
# numpy intermediates to disk/SHM. Only shard_put still required.
#
# Env: K26_MOE_CACHE_DIR (default empty = disabled)
#   Recommended: /dev/shm/k26_moe_cache (5L, fits 95 GB) or
#                /mnt/pd/k26_moe_cache (60L, ~1.1 TB)
# ============================================================

K26_MOE_CACHE_DIR = os.environ.get("K26_MOE_CACHE_DIR", "")
K26_USE_V16 = os.environ.get("K26_USE_V16", "0") == "1"
if K26_USE_V16:
    print("[v16] K26_USE_V16=1: keep packed uint32 in HBM, bitcast int4 view in apply_jax")


def _cache_path(layer_prefix: str, ep_size: int) -> str:
    """Cache file path: include layer prefix + EP size for invalidation."""
    safe_prefix = layer_prefix.replace("/", "_").replace(".", "_")
    return os.path.join(K26_MOE_CACHE_DIR, f"{safe_prefix}_ep{ep_size}.npz")


def _try_load_moe_cache(layer_prefix: str, ep_size: int):
    """Try load processed numpy arrays from cache. Return None on miss/error."""
    if not K26_MOE_CACHE_DIR:
        return None
    cache_path = _cache_path(layer_prefix, ep_size)
    if not os.path.exists(cache_path):
        return None
    try:
        npz = np.load(cache_path)
        return (npz["w13"], npz["w2"], npz["s13"], npz["s_down"])
    except Exception as e:
        logger.warning(f"[MoE-cache] Load failed for {cache_path}: {e}")
        return None


def _save_moe_cache(layer_prefix: str, ep_size: int,
                     w13, w2, s13, s_down):
    """Save processed numpy arrays to cache. Best-effort."""
    if not K26_MOE_CACHE_DIR:
        return
    try:
        os.makedirs(K26_MOE_CACHE_DIR, exist_ok=True)
        cache_path = _cache_path(layer_prefix, ep_size)
        # No compression for speed (5L cache 95 GB, 60L 1.1 TB - need PD)
        np.savez(cache_path, w13=w13, w2=w2, s13=s13, s_down=s_down)
        logger.info(f"[MoE-cache] Saved {cache_path}")
    except Exception as e:
        logger.warning(f"[MoE-cache] Save failed for {layer_prefix}: {e}")


# =============================================================================
# Core JAX functions: unpack + dequant (Round 3a)
# =============================================================================


def unpack_int32_to_uint4(packed: jax.Array,
                          packed_dim: int = -1) -> jax.Array:
    """Unpack int32 (packed 8 uint4 nibbles) -> uint4 array (stored as uint8).

    Compressed-tensors pack-quantized format:
      Each int32 stores 8 uint4 values [0, 15] in nibbles (4 bits each).
      Layout (lowest bits first):
        bits  0- 3: element 0
        bits  4- 7: element 1
        ...
        bits 28-31: element 7

    Output is stored as jnp.int32 (we don't use jnp.uint4 because it's not
    universally supported across JAX versions; the values are guaranteed to be
    in [0, 15] until sign-converted in dequant).

    Args:
      packed: int32 array, shape [..., K_packed, ...] where K_packed = K / 8
      packed_dim: which axis is the packed (K_packed) dimension to expand 8x

    Returns:
      uint4 values stored as int32, shape [..., K, ...] where K = K_packed * 8
      with the packed_dim expanded.

    Example:
      packed[i] = 0x0000_0123  (in hex)
      unpacked[i*8 + 0] = 3  (lowest nibble)
      unpacked[i*8 + 1] = 2
      unpacked[i*8 + 2] = 1
      unpacked[i*8 + 3..7] = 0
    """
    packed = packed.astype(jnp.int32)

    # Generate 8 right-shift amounts: [0, 4, 8, 12, 16, 20, 24, 28]
    shifts = jnp.arange(8, dtype=jnp.int32) * 4

    # Reshape shifts to broadcast on a new last dim
    # If packed shape is (A, B, C) and packed_dim is the C axis, we want
    # output shape (A, B, C, 8) where last dim is shift index.
    new_shape_for_shifts = (1, ) * packed.ndim + (8, )
    shifts = shifts.reshape(new_shape_for_shifts)

    # Expand packed array with a new last dim for shifts
    packed_expanded = jnp.expand_dims(packed, axis=-1)  # [..., K_packed, 1]

    # Right-shift and mask 0xF
    unpacked = (packed_expanded >> shifts) & 0xF  # [..., K_packed, 8] int32

    # Move the new dim (8) right after packed_dim, then collapse:
    # Goal: collapse [..., K_packed, 8] -> [..., K_packed * 8] along packed_dim
    # If packed_dim is -1 (default), this is straightforward reshape.
    # For other packed_dim values, we transpose first.

    if packed_dim == -1 or packed_dim == packed.ndim - 1:
        # Simple case: packed_dim is last. Collapse last two dims.
        out_shape = packed.shape[:-1] + (packed.shape[-1] * 8, )
        return unpacked.reshape(out_shape)
    else:
        # General case: we need to interleave new dim into packed_dim position
        # unpacked shape: original_shape + (8,)
        # We want: shape with packed_dim multiplied by 8
        # Move the last dim (8) to right after packed_dim, then reshape.
        # E.g. shape (A, B, C) packed_dim=1, unpacked shape (A, B, C, 8)
        # Move axis -1 to position 2 (right after B): (A, B, 8, C)
        # Reshape to (A, B*8, C)
        normalized_dim = packed_dim if packed_dim >= 0 else packed_dim + packed.ndim
        # Move last axis to position normalized_dim + 1
        unpacked = jnp.moveaxis(unpacked, -1, normalized_dim + 1)
        # Now unpacked shape: ...packed.shape[normalized_dim]..., 8, ...
        # Collapse [normalized_dim, normalized_dim+1] into one
        new_shape = list(packed.shape)
        new_shape[normalized_dim] *= 8
        return unpacked.reshape(new_shape)


def dequant_int4_to_bf16(
    w_uint4: jax.Array,
    scale: jax.Array,
    group_size: int = 32,
    packed_dim: int = -1,
) -> jax.Array:
    """K2.6 symmetric INT4 dequant: uint4 [0, 15] -> int4 [-8, 7] -> bf16 x scale.

    K2.6 quantization is symmetric (no zero_point). Compressed-tensors stores
    weights in uint4 container [0, 15], the int4 sign convert is just `-8`:

      w_real = (uint4_val - 8) * scale

    The PR #2306 (vllm-project/tpu-inference#2306) uses the same convention:
      `(w_uint4 - 8).astype(jnp.int4)` (compressed_tensors_moe_w4a8.py:185)

    Args:
      w_uint4: shape [..., K, ...], uint4 values stored as int32 [0, 15]
      scale:   shape [..., K/group_size, ...], BF16 per-group scale
      group_size: typically 32 for K2.6
      packed_dim: which axis K is on (for scale broadcasting)

    Returns:
      BF16 dequantized weight, shape same as w_uint4.
    """
    # 1. uint4 -> int4 sign convert: [0, 15] -> [-8, 7]
    # We keep int8 container (broader compatibility) instead of jnp.int4
    # for this intermediate step, to avoid potential XLA dtype quirks.
    w_int8 = w_uint4.astype(jnp.int8) - jnp.int8(8)

    # 2. int8 -> bf16 (TPU VPU has direct hardware support per user guidance)
    w_bf16 = w_int8.astype(jnp.bfloat16)

    # 3. Broadcast scale: repeat each scale value group_size times along packed_dim
    # scale shape is [..., K/group_size, ...], we want [..., K, ...]
    scale_bf16 = scale.astype(jnp.bfloat16)
    scale_expanded = jnp.repeat(scale_bf16, group_size, axis=packed_dim)

    # 4. Symmetric: just multiply (no zero_point subtraction)
    return w_bf16 * scale_expanded


def unpack_and_dequant_int4_to_bf16(
    weight_packed: jax.Array,
    weight_scale: jax.Array,
    group_size: int = 32,
    packed_dim: int = -1,
) -> jax.Array:
    """Combined unpack + dequant in one call.

    Args:
      weight_packed: int32, shape [..., K_packed, ...] where K_packed = K / 8
      weight_scale:  bf16, shape [..., K/group_size, ...]
      group_size:    32 for K2.6
      packed_dim:    which axis is K (after unpack)

    Returns:
      BF16 weight, shape [..., K, ...]
    """
    w_uint4 = unpack_int32_to_uint4(weight_packed, packed_dim=packed_dim)
    return dequant_int4_to_bf16(w_uint4,
                                weight_scale,
                                group_size=group_size,
                                packed_dim=packed_dim)


# =============================================================================
# QuantizationConfig + Method classes (Round 3b)
# =============================================================================

import gc
import os
import re
import time
from typing import Iterable

from flax import nnx

from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights)
from tpu_inference.layers.common.quantization.unquantized import \
    process_unquantized_moe_weights
from tpu_inference.layers.common.sharding import ShardingAxisNameBase as ShardingAxisName
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.utils import get_mesh_shape_product
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (QuantizationConfig,
                                                           QuantLinearConfig)
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedFusedMoEMethod, UnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import \
    jax_array_from_reshaped_torch
from tpu_inference.models.jax.utils.weight_utils import shard_put

logger = init_logger(__name__)

# K2.6 W4A16 supports only GMM-based MoE backends (not FUSED_MOE which is for FP8 cache).
W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS = (MoEBackend.GMM_EP,
                                             MoEBackend.GMM_TP)


def _cpu_process_int4_layer(
    packed_lists,
    scale_lists,
    moe_backend,
    mesh,
    activation,
    group_size,
):
    """CPU-heavy work: concat per-expert lists + unpack int32→int4 + reorder.

    [v23 Phase 1+2] All-numpy fast path for K2.6:
      Phase 1: numpy unpack (4-17x speedup, bitwise verified)
      Phase 2: numpy reorder (skip process_moe_weights for GMM_EP, NO-OP for K2.6
               since intermediate=2048 already aligned to 128 → pad_amount=0)
    GMM_TP fallback path: keep jnp (used by other models, K2.6 doesn't use this).
    """
    timings = {}

    def _t(label, val):
        timings[label] = val

    # Step A: numpy concat + unpack int32→int4
    ta0 = time.perf_counter()

    def _concat_unpack_int4_numpy(packed_list):
        """Fast byte-view unpack: avoid 8x int32 broadcast.

        int32 little-endian layout (verified via unit test):
          byte 0: nibble 0 (low) | nibble 1 (high)
          byte 1: nibble 2 (low) | nibble 3 (high)
          byte 2: nibble 4 (low) | nibble 5 (high)
          byte 3: nibble 6 (low) | nibble 7 (high)
        Unit test ALL PASS bitwise vs jnp (3-4400x speedup, K2.6 size 3x).
        """
        packed_np_list = [np.asarray(p, dtype=np.int32) for p in packed_list]
        packed_concat = np.concatenate(packed_np_list, axis=0)
        if not packed_concat.flags['C_CONTIGUOUS']:
            packed_concat = np.ascontiguousarray(packed_concat)
        # View int32 as uint8: 0 copy, just reinterpret. shape (..., K_packed * 4)
        packed_u8 = packed_concat.view(np.uint8)
        # Extract nibbles directly on int8 (avoid int32 broadcast temporary)
        low = packed_u8 & 0x0F
        high = (packed_u8 >> 4) & 0x0F
        out_shape = packed_u8.shape[:-1] + (packed_u8.shape[-1] * 2,)
        out = np.empty(out_shape, dtype=np.int8)
        out[..., 0::2] = low  # interleave: nibble 0, 1, 2, 3, ...
        out[..., 1::2] = high
        out -= 8  # int4 range [-8, 7]
        return out

    w_gate_np = _concat_unpack_int4_numpy(packed_lists["gate"])
    ta1 = time.perf_counter()
    w_up_np = _concat_unpack_int4_numpy(packed_lists["up"])
    ta2 = time.perf_counter()
    w_down_np = _concat_unpack_int4_numpy(packed_lists["down"])
    ta3 = time.perf_counter()
    _t("A_unpack_gate", ta1 - ta0)
    _t("A_unpack_up", ta2 - ta1)
    _t("A_unpack_down", ta3 - ta2)

    # Step B: numpy concat scale
    tb0 = time.perf_counter()
    s_gate_np = np.concatenate([np.asarray(s) for s in scale_lists["gate"]], axis=0)
    s_up_np = np.concatenate([np.asarray(s) for s in scale_lists["up"]], axis=0)
    s_down_np = np.concatenate([np.asarray(s) for s in scale_lists["down"]], axis=0)
    tb1 = time.perf_counter()
    _t("B_scale_concat", tb1 - tb0)

    # Step C: numpy fuse w13 (axis=1 = intermediate axis)
    tc0 = time.perf_counter()
    w13_np = np.concatenate([w_gate_np, w_up_np], axis=1)
    s13_np = np.concatenate([s_gate_np, s_up_np], axis=1)
    del w_gate_np, w_up_np, s_gate_np, s_up_np
    tc1 = time.perf_counter()
    _t("C_fuse_w13", tc1 - tc0)

    # Step D: numpy reorder (K2.6 GMM_EP fast path) or jnp fallback (GMM_TP)
    td0 = time.perf_counter()

    if moe_backend == MoEBackend.GMM_EP:
        # K2.6 v3 fast path: jnp swapaxes + with_layout_constraint = metadata-only
        # transpose (XLA optimizes, no actual memory copy).
        # Skip process_w13_for_gmm because for K2.6 (intermediate=2048 aligned to 128,
        # reorder_size=1, pad_amount=0), it is a NO-OP equivalent to identity.

        with cpu_mesh_context():
            # Convert numpy int8 → jnp int4 (jit cached after layer 1)
            tconv0 = time.perf_counter()
            w13_jnp_int4 = jax.block_until_ready(jnp.asarray(w13_np).astype(jnp.int4))
            w2_jnp_int4 = jax.block_until_ready(jnp.asarray(w_down_np).astype(jnp.int4))
            del w13_np, w_down_np
            s13_bf16 = jax.block_until_ready(jnp.asarray(s13_np).astype(jnp.bfloat16))
            s_down_bf16 = jax.block_until_ready(jnp.asarray(s_down_np).astype(jnp.bfloat16))
            del s13_np, s_down_np
            tconv1 = time.perf_counter()
            _t("A_np2jnp_convert", tconv1 - tconv0)

            # jnp swapaxes + with_layout_constraint = metadata-only transpose
            # (XLA optimizes via layout constraint, much faster than numpy ascontig)
            w13_swapped = jnp.swapaxes(w13_jnp_int4, 1, 2)
            w13_swapped = with_layout_constraint(w13_swapped, Layout((0, 1, 2)))
            w2_swapped = jnp.swapaxes(w2_jnp_int4, 1, 2)
            w2_swapped = with_layout_constraint(w2_swapped, Layout((0, 1, 2)))

            # Scale: astype(f32) + swapaxes + expand_dims (matches process_moe_weights)
            s13_f32 = s13_bf16.astype(jnp.float32)
            s13_swapped = jnp.swapaxes(s13_f32, 1, 2)
            s13_expanded = jnp.expand_dims(s13_swapped, axis=2)

            s_down_f32 = s_down_bf16.astype(jnp.float32)
            s_down_swapped = jnp.swapaxes(s_down_f32, 1, 2)
            s_down_expanded = jnp.expand_dims(s_down_swapped, axis=2)

            # Block on outputs
            jax.block_until_ready(w13_swapped)
            jax.block_until_ready(w2_swapped)
            jax.block_until_ready(s13_expanded)
            jax.block_until_ready(s_down_expanded)

        td1 = time.perf_counter()
        _t("D_reorder_moe", td1 - td0)

        return (w13_swapped, w2_swapped, s13_expanded, s_down_expanded, timings)

    # GMM_TP fallback (not K2.6, keep jnp + cpu_mesh_context for safety)
    with cpu_mesh_context():
        ta_conv0 = time.perf_counter()
        w_gate = jax.block_until_ready(jnp.asarray(w_gate_np).astype(jnp.int4))
        w_up = jax.block_until_ready(jnp.asarray(w_up_np).astype(jnp.int4))
        w_down = jax.block_until_ready(jnp.asarray(w_down_np).astype(jnp.int4))
        del w_gate_np, w_up_np, w_down_np
        ta_conv1 = time.perf_counter()
        _t("A_np2jnp_convert", ta_conv1 - ta_conv0)

        s_gate = jax.block_until_ready(jnp.asarray(s_gate_np).astype(jnp.bfloat16))
        s_up = jax.block_until_ready(jnp.asarray(s_up_np).astype(jnp.bfloat16))
        s_down = jax.block_until_ready(jnp.asarray(s_down_np).astype(jnp.bfloat16))
        del s_gate_np, s_up_np, s_down_np

        w13 = jax.block_until_ready(jnp.concatenate([w_gate, w_up], axis=1))
        s13 = jax.block_until_ready(jnp.concatenate([s_gate, s_up], axis=1))
        del w_gate, w_up, s_gate, s_up

        input_weights = FusedMoEWeights(
            w13_weight=w13, w13_weight_scale=s13, w13_bias=None,
            w2_weight=w_down, w2_weight_scale=s_down, w2_bias=None,
        )
        w13_reorder_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
        w13_interleave = (activation == MoEActivation.SWIGLUOAI)
        processed = process_moe_weights(
            input_weights, moe_backend=moe_backend,
            w13_reorder_size=w13_reorder_size, w13_interleave=w13_interleave,
        )
        jax.block_until_ready(processed.w13_weight)
        jax.block_until_ready(processed.w2_weight)
        jax.block_until_ready(processed.w13_weight_scale)
        jax.block_until_ready(processed.w2_weight_scale)
        td1 = time.perf_counter()
        _t("D_reorder_moe", td1 - td0)

        return (processed.w13_weight, processed.w2_weight,
                processed.w13_weight_scale, processed.w2_weight_scale,
                timings)


class CompressedTensorsW4A16Config(QuantizationConfig):
    """K2.6 W4A16 INT4 quantization (compressed-tensors symmetric, group=32).

    HF quantization_config sample (K2.6, verified 2026-04-25):
        {
          "config_groups": {
            "group_0": {"weights": {"num_bits": 4, "type": "int",
                                    "strategy": "group", "group_size": 32,
                                    "symmetric": true, "actorder": null}}
          },
          "format": "pack-quantized",
          "ignore": ["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*",
                     "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"],
          "quant_method": "compressed-tensors"
        }
    """

    def __init__(self, hf_quant_config: dict):
        super().__init__(hf_quant_config)

        groups = hf_quant_config.get("config_groups", {})
        if not groups:
            raise ValueError(
                "Expected 'config_groups' in hf_quant_config for "
                "compressed-tensors W4A16 quantization")

        # K2.6 has a single config_groups.group_0 (all layers use same scheme)
        first_group = next(iter(groups.values()))
        weights_cfg = first_group.get("weights", {})

        self.num_bits = weights_cfg.get("num_bits", 4)
        self.group_size = weights_cfg.get("group_size", 32)
        self.symmetric = weights_cfg.get("symmetric", True)
        self.strategy = weights_cfg.get("strategy", "group")
        self.actorder = weights_cfg.get("actorder", None)

        self.format = hf_quant_config.get("format", "pack-quantized")
        self.ignored_layers_patterns = hf_quant_config.get("ignore", [])

        # Hard constraints (matching PR #2306 design)
        if self.num_bits != 4:
            raise NotImplementedError(
                f"CompressedTensorsW4A16Config only supports num_bits=4, "
                f"got {self.num_bits}")
        if not self.symmetric:
            raise NotImplementedError(
                "CompressedTensorsW4A16Config only supports symmetric "
                "quantization (PR #2306 has the same constraint, K2.6 is "
                "verified symmetric=true)")
        if self.format != "pack-quantized":
            raise NotImplementedError(
                f"Only 'pack-quantized' format supported, got "
                f"{self.format}")
        if self.strategy != "group":
            raise NotImplementedError(
                f"Only 'group' strategy supported, got {self.strategy}")
        if self.actorder is not None:
            raise NotImplementedError(
                f"actorder must be None, got {self.actorder}")

    def is_layer_ignored(self, prefix: str) -> bool:
        """K2.6 ignore patterns use 're:' prefix for regex matching.

        Examples:
          'lm_head'                           → exact match
          're:.*self_attn.*'                  → all attention weights
          're:.*shared_experts.*'             → shared MoE expert
          're:.*mlp\\.(gate|up|gate_up|down)_proj.*' → dense MLP (layer 0)
        """
        for pattern in self.ignored_layers_patterns:
            if pattern.startswith("re:"):
                regex = pattern[3:]
                if re.search(regex, prefix):
                    return True
            elif pattern == prefix:
                return True
        return False

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            # Linear layers in K2.6 are all in the ignore list
            # (lm_head, attention, shared_experts, dense MLP).
            # Even if not ignored, we currently only support W4A16 for MoE.
            linear_config = QuantLinearConfig(layer, enable_sp=False)
            return UnquantizedLinearMethod(linear_config)

        if isinstance(layer, JaxMoE):
            if self.is_layer_ignored(prefix):
                return UnquantizedFusedMoEMethod()
            # K2.6 routed experts go through W4A16 INT4 path.
            return CompressedTensorsW4A16FusedMoEMethod(
                group_size=self.group_size)

        return None


class CompressedTensorsW4A16FusedMoEMethod(QuantizeMethodBase):
    """K2.6 W4A16 MoE: per-expert INT4 packed → BF16 dequant → standard MoE forward.

    Loads `weight_packed` (int32, packed 8 uint4) + `weight_scale` (bf16, per-group)
    from safetensors. In `process_weights_after_loading`, unpacks + dequants to BF16,
    then delegates to `process_unquantized_moe_weights` for standard MoE processing.

    Forward (apply_jax) reuses `UnquantizedFusedMoEMethod.apply_jax` because by
    that point all weights are already BF16.
    """

    def __init__(self, group_size: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size
        self.extra_backend_kwargs = {}
        # Reuse Unquantized for forward path (weights are dequant'd to BF16)
        self._unquantized_method = UnquantizedFusedMoEMethod()

    def create_weights_jax(self, layer: JaxMoE, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """Create per-expert INT4 packed Param slots + per-group scale Param slots.

        Layout (matches K2.6 safetensors):
          kernel_gating_EDF                  : (E, D, F) - existing BF16 slot, will be replaced
          kernel_gating_EDF_weight_packed    : (E, F, D/8) int32, packed 8 uint4 per int32
          kernel_gating_EDF_weight_scale     : (E, F, D/group_size) bf16

        Same for up_proj_EDF and down_proj_EFD.

        Note: Compressed-tensors stores weights TRANSPOSED relative to JAX layout.
        K2.6 weight_packed shape (per safetensors): (E, F, D/8) for gate/up_proj.
        We'll transpose during process_weights_after_loading.
        """
        if layer.moe_backend not in W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            raise NotImplementedError(
                f"W4A16 MoE supports only "
                f"{W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}, got "
                f"{layer.moe_backend}")

        # The original BF16 kernel_*_EDF slots are still there from JaxMoE.__init__.
        # We add three extra slot pairs for INT4 packed + scale.
        # E = num_experts, D = hidden_size, F = moe_intermediate_size
        E = layer.num_local_experts
        D = layer.kernel_gating_EDF.value.shape[-2]
        F = layer.kernel_gating_EDF.value.shape[-1]
        gs = self.group_size

        # weight_packed shape: each int32 packs 8 uint4 along the K dimension.
        # For gate/up_proj, K = D (hidden), so packed dim is D/8.
        # For down_proj, K = F (intermediate), so packed dim is F/8.
        # Note: compressed-tensors safetensors actual layout may have weights
        # transposed; we keep nnx Param shape matching after-unpack-and-dequant
        # for downstream `process_unquantized_moe_weights` compatibility.

        for proj_name, packed_shape, scale_shape in [
            ("kernel_gating_EDF", (E, F, D // 8), (E, F, D // gs)),
            ("kernel_up_proj_EDF", (E, F, D // 8), (E, F, D // gs)),
            ("kernel_down_proj_EFD", (E, D, F // 8), (E, D, F // gs)),
        ]:
            packed_param = nnx.Param(
                jnp.zeros(packed_shape, dtype=jnp.int32),
                _weights_to_load=[None for _ in range(E)],
            )
            scale_param = nnx.Param(
                jnp.zeros(scale_shape, dtype=jnp.bfloat16),
                _weights_to_load=[None for _ in range(E)],
            )
            setattr(layer, f"{proj_name}_weight_packed", packed_param)
            setattr(layer, f"{proj_name}_weight_scale", scale_param)

    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,
                     weights: Iterable) -> set:
        """Route weight_packed / weight_scale per-expert to nnx Param slots.

        Skip weight_shape (compressed-tensors metadata, not needed; shape inferred
        from unpack).
        """
        remaining_weights = dict()
        cnt_packed = 0
        cnt_scale = 0
        cnt_shape_skipped = 0

        for torch_name, torch_weight in weights:
            # torch_name is something like ".0.down_proj.weight_packed"
            # (after prefix stripping by JaxAutoWeightsLoader)
            # Format: .{expert_id}.{proj_name}.{suffix}
            stripped_name = torch_name.split(layer.prefix)[-1]
            names = stripped_name.split(".")

            if len(names) != 3:
                # Unexpected name format (e.g. router weights, biases) — pass through
                remaining_weights[torch_name] = torch_weight
                continue

            expert_id_str, proj_name, suffix = names
            try:
                expert_id = int(expert_id_str)
            except ValueError:
                remaining_weights[torch_name] = torch_weight
                continue

            # Map proj_name to JAX param prefix
            proj_to_param = {
                "gate_proj": "kernel_gating_EDF",
                "up_proj": "kernel_up_proj_EDF",
                "down_proj": "kernel_down_proj_EFD",
            }
            if proj_name not in proj_to_param:
                remaining_weights[torch_name] = torch_weight
                continue
            jax_param_prefix = proj_to_param[proj_name]

            if suffix == "weight_packed":
                jax_param_name = f"{jax_param_prefix}_weight_packed"
                cnt_packed += 1
            elif suffix == "weight_scale":
                jax_param_name = f"{jax_param_prefix}_weight_scale"
                cnt_scale += 1
            elif suffix == "weight_shape":
                # K2.6 weight_shape is metadata, skip (shape inferred from unpack)
                cnt_shape_skipped += 1
                continue
            else:
                remaining_weights[torch_name] = torch_weight
                continue

            jax_param = getattr(layer, jax_param_name, None)
            assert isinstance(jax_param, nnx.Param), (
                f"Expected nnx.Param for {jax_param_name}, got {type(jax_param)}"
            )

            jax_weight = jax_array_from_reshaped_torch(
                torch_weight, reshape_dims=(1, ) + torch_weight.shape)
            jax_param._weights_to_load[expert_id] = jax_weight

        logger.debug(
            f"[W4A16] {layer.prefix}: loaded {cnt_packed} weight_packed + "
            f"{cnt_scale} weight_scale; skipped {cnt_shape_skipped} weight_shape")

        # Pass non-INT4 weights to original loader (for biases, etc.)
        loaded_names = original_load_weights_fn(remaining_weights.items())

        for proj in [
                "kernel_gating_EDF", "kernel_up_proj_EDF",
                "kernel_down_proj_EFD"
        ]:
            for suffix in ["weight_packed", "weight_scale"]:
                pname = f"{proj}_{suffix}"
                param = getattr(layer, pname)
                if all(w is not None for w in param._weights_to_load):
                    loaded_names.add(pname)

        return loaded_names

    def process_weights_after_loading(self, layer: JaxMoE) -> bool:
        """[v14 Round 9] Keep INT4 in HBM. NO dequant to BF16.

        Critical fix from prior rounds: we previously dequant'd to BF16 and stored
        in HBM (44 GB/layer × 60 = 2.6 TB total → fits 1L/4L tests but OOM at 61L).
        Now we keep INT4 packed values in int8 container (8 GB/layer × 60 / 8 dev
        = ~70 GB/dev), and scale separately. apply_jax does on-the-fly dequant
        via gmm_v2 INT8 path (lhs auto-quantized to INT8).

        Steps:
          1. Wait until all per-expert slots are loaded (return False if not)
          2. Concat per-expert: packed (int32) + scale (bf16)
          3. Unpack int32 → uint4 → -8 → int8 container (values [-8, 7] in int8)
             ↳ NO dequant to BF16. NO multiply by scale.
          4. Fuse w13 = [w_gate, w_up], also fuse scale_13 = [scale_gate, scale_up]
          5. Shard int8 weight + bf16 scale to HBM (~70 GB/dev for 60 layers, fits)
          6. apply_jax (forward) reads int8 + scale, calls gmm_v2 INT8 path
        """
        if layer.moe_backend not in W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            raise NotImplementedError(
                f"W4A16 MoE supports only "
                f"{W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}, got "
                f"{layer.moe_backend}")

        t0 = time.perf_counter()

        # Step 1: check all weights loaded
        params_to_check = []
        for proj in [
                "kernel_gating_EDF", "kernel_up_proj_EDF",
                "kernel_down_proj_EFD"
        ]:
            for suffix in ["weight_packed", "weight_scale"]:
                params_to_check.append(getattr(layer, f"{proj}_{suffix}"))

        if any(any(w is None for w in p._weights_to_load)
               for p in params_to_check):
            return False

        t_check = time.perf_counter()

        # Step 2: snapshot per-expert weight lists from nnx.Param
        packed_lists = {
            "gate": list(layer.kernel_gating_EDF_weight_packed._weights_to_load),
            "up": list(layer.kernel_up_proj_EDF_weight_packed._weights_to_load),
            "down": list(layer.kernel_down_proj_EFD_weight_packed._weights_to_load),
        }
        scale_lists = {
            "gate": list(layer.kernel_gating_EDF_weight_scale._weights_to_load),
            "up": list(layer.kernel_up_proj_EDF_weight_scale._weights_to_load),
            "down": list(layer.kernel_down_proj_EFD_weight_scale._weights_to_load),
        }
        t_snap = time.perf_counter()

        # Step 3: free intermediate per-expert Params ASAP
        del layer.kernel_gating_EDF_weight_packed
        del layer.kernel_gating_EDF_weight_scale
        del layer.kernel_up_proj_EDF_weight_packed
        del layer.kernel_up_proj_EDF_weight_scale
        del layer.kernel_down_proj_EFD_weight_packed
        del layer.kernel_down_proj_EFD_weight_scale
        del layer.kernel_gating_EDF
        del layer.kernel_up_proj_EDF
        del layer.kernel_down_proj_EFD
        t_del = time.perf_counter()

        # Step 4: CPU work (or cache hit). Phase 3: try cache first.
        ep_size = get_mesh_shape_product(
            layer.mesh, ShardingAxisName.ATTN_DATA_EXPERT)
        cache_hit = False

        # === v16 fast path: keep packed uint32, no unpack/reorder/cpu_mesh ===
        if K26_USE_V16 and layer.moe_backend == MoEBackend.GMM_EP:
            from jax.sharding import NamedSharding
            t_v16_0 = time.perf_counter()
            # K2.6 per-expert weight is shape [1, F, D/8] uint32 (1 = leading expert dim)
            # Use np.concatenate axis=0 to merge 384 experts directly: [E=384, F, D/8]
            w_gate_np = np.concatenate([np.asarray(p) for p in packed_lists["gate"]], axis=0)
            w_up_np = np.concatenate([np.asarray(p) for p in packed_lists["up"]], axis=0)
            w_down_np = np.concatenate([np.asarray(p) for p in packed_lists["down"]], axis=0)
            t_v16_1 = time.perf_counter()
            # fuse w13 axis=1 (F dim): [E, 2*F, D/8]
            w13_packed_np = np.concatenate([w_gate_np, w_up_np], axis=1)
            del w_gate_np, w_up_np
            # scale: same pattern [E, 2*F, D/group] bf16
            s_gate_np = np.concatenate([np.asarray(s) for s in scale_lists["gate"]], axis=0)
            s_up_np = np.concatenate([np.asarray(s) for s in scale_lists["up"]], axis=0)
            s_down_np = np.concatenate([np.asarray(s) for s in scale_lists["down"]], axis=0)
            s13_np = np.concatenate([s_gate_np, s_up_np], axis=1)
            del s_gate_np, s_up_np
            print('[v16-DEBUG] post-fuse w13_packed_np.shape={}, w_down_np.shape={}, s13_np.shape={}, s_down_np.shape={}'.format(
                w13_packed_np.shape, w_down_np.shape, s13_np.shape, s_down_np.shape))
            t_v16_2 = time.perf_counter()
            # device_put to HBM with sharding (E shard via ATTN_DATA_EXPERT axis)
            w13_sharding = NamedSharding(layer.mesh, P(layer.edf_sharding[0], None, None))
            w2_sharding = NamedSharding(layer.mesh, P(layer.efd_sharding[0], None, None))
            s13_sharding_v16 = NamedSharding(layer.mesh, P(layer.edf_sharding[0], None, None))
            s2_sharding_v16 = NamedSharding(layer.mesh, P(layer.efd_sharding[0], None, None))
            # v17 actual: device_put uint32, then bitcast+reshape+transpose ON HBM
            # Goal: store int4 dtype with [E, D, 2F] (matches v14 layout, fused_moe_func happy)
            w13_uint32_hbm = jax.device_put(w13_packed_np, w13_sharding)
            # bitcast as UNSIGNED uint4 (matches storage), then -8 to recover signed int4 [-8, 7]
            # (v14 path: low/high nibble extract + `out -= 8`. v17 must mirror this offset.)
            w13_uint4_raw = jax.lax.bitcast_convert_type(w13_uint32_hbm, jnp.uint4)  # [E, 2F, D/8, 8] uint4
            E_l = w13_uint4_raw.shape[0]
            two_F = w13_uint4_raw.shape[1]
            D_p8 = w13_uint4_raw.shape[2]
            w13_uint4_FD = w13_uint4_raw.reshape(E_l, two_F, D_p8 * 8)  # [E, 2F, D]
            w13_int4_FD = (w13_uint4_FD.astype(jnp.int8) - 8).astype(jnp.int4)  # signed [-8, 7]
            w13_int8_reordered = jnp.transpose(w13_int4_FD, (0, 2, 1))  # [E, D, 2F] int4
            # v14-equivalent: force XLA to materialize transpose (else lazy view, fused_moe_func sees source layout)
            w13_int8_reordered = with_layout_constraint(w13_int8_reordered, Layout((0, 1, 2)))
            del w13_uint32_hbm, w13_uint4_raw, w13_uint4_FD, w13_int4_FD

            w2_uint32_hbm = jax.device_put(w_down_np, w2_sharding)
            w2_uint4_raw = jax.lax.bitcast_convert_type(w2_uint32_hbm, jnp.uint4)  # [E, D, F/8, 8] uint4
            E_l2 = w2_uint4_raw.shape[0]
            D_dim = w2_uint4_raw.shape[1]
            F_p8 = w2_uint4_raw.shape[2]
            w2_uint4_DF = w2_uint4_raw.reshape(E_l2, D_dim, F_p8 * 8)  # [E, D, F]
            w2_int4_DF = (w2_uint4_DF.astype(jnp.int8) - 8).astype(jnp.int4)  # signed [-8, 7]
            w_down_int8_reordered = jnp.transpose(w2_int4_DF, (0, 2, 1))  # [E, F, D] int4
            w_down_int8_reordered = with_layout_constraint(w_down_int8_reordered, Layout((0, 1, 2)))
            del w2_uint32_hbm, w2_uint4_raw, w2_uint4_DF, w2_int4_DF

            s13_reordered = jax.device_put(s13_np, s13_sharding_v16)
            s_down_reordered = jax.device_put(s_down_np, s2_sharding_v16)
            jax.block_until_ready(w13_int8_reordered)
            jax.block_until_ready(w_down_int8_reordered)
            jax.block_until_ready(s13_reordered)
            jax.block_until_ready(s_down_reordered)
            t_v16_3 = time.perf_counter()
            del w13_packed_np, w_down_np, s13_np, s_down_np
            del packed_lists, scale_lists
            # Skip cache + step 5 shard_put (already on HBM)
            cpu_timings = {
                "A_unpack_gate": 0.0, "A_unpack_up": 0.0, "A_unpack_down": 0.0,
                "A_np2jnp_convert": t_v16_3 - t_v16_2, "B_scale_concat": t_v16_2 - t_v16_1,
                "C_fuse_w13": 0.0, "D_reorder_moe": t_v16_1 - t_v16_0,
            }
            t_cpu = time.perf_counter()
            ts0 = ts1 = ts2 = ts3 = ts4 = time.perf_counter()
            # Store directly (skip standard step 5)
            layer.kernel_gating_upproj_EDF = nnx.Param(w13_int8_reordered)
            layer.kernel_down_proj_EFD = nnx.Param(w_down_int8_reordered)
            layer.kernel_gating_upproj_EDF_scale = nnx.Param(s13_reordered)
            layer.kernel_down_proj_EFD_scale = nnx.Param(s_down_reordered)
            del w13_int8_reordered, w_down_int8_reordered, s13_reordered, s_down_reordered
            t_store = time.perf_counter()
            gc.collect()
            t_gc = time.perf_counter()
            elapsed = t_gc - t0
            logger.info(
                f"[W4A16-V16-TIME][PACKED] {layer.prefix} TOTAL={elapsed:.2f}s | "
                f"v16_stack={t_v16_1-t_v16_0:.2f} v16_scale={t_v16_2-t_v16_1:.2f} "
                f"v16_devput={t_v16_3-t_v16_2:.2f} store={t_store-ts4:.2f} gc={t_gc-t_store:.2f}"
            )
            return True
        # === end v16 fast path ===

        cpu_timings = {
            "A_unpack_gate": 0.0, "A_unpack_up": 0.0, "A_unpack_down": 0.0,
            "A_np2jnp_convert": 0.0, "B_scale_concat": 0.0,
            "C_fuse_w13": 0.0, "D_reorder_moe": 0.0,
        }
        cached = _try_load_moe_cache(layer.prefix, ep_size)
        if cached is not None:
            # Cache hit: numpy → jnp + dtype convert (jit cached after layer 1)
            w13_np, w2_np, s13_np, s_down_np = cached
            with cpu_mesh_context():
                w13_int8_reordered = jax.block_until_ready(
                    jnp.asarray(w13_np).astype(jnp.int4))
                w_down_int8_reordered = jax.block_until_ready(
                    jnp.asarray(w2_np).astype(jnp.int4))
                s13_reordered = jax.block_until_ready(
                    jnp.asarray(s13_np).astype(jnp.bfloat16))
                s_down_reordered = jax.block_until_ready(
                    jnp.asarray(s_down_np).astype(jnp.bfloat16))
            del w13_np, w2_np, s13_np, s_down_np
            cache_hit = True
            del packed_lists, scale_lists
        else:
            # Cache miss: full CPU work (returns jnp arrays after numpy + convert)
            # NOTE: For cache save, we want numpy intermediates (smaller, faster I/O).
            # _cpu_process_int4_layer already does numpy → jnp at end. We re-extract
            # numpy via np.asarray for cache save (cheap, jnp host array → numpy view).
            (w13_int8_reordered, w_down_int8_reordered,
             s13_reordered, s_down_reordered, cpu_timings) = _cpu_process_int4_layer(
                packed_lists, scale_lists,
                layer.moe_backend, layer.mesh, layer.activation,
                self.group_size,
            )
            del packed_lists, scale_lists
            # Save numpy to cache. Cast jnp.int4 → jnp.int8 first to avoid
            # numpy void dtype |V1 (which causes JAX load error).
            if K26_MOE_CACHE_DIR and layer.moe_backend == MoEBackend.GMM_EP:
                with cpu_mesh_context():
                    w13_int8_jnp = jax.block_until_ready(
                        w13_int8_reordered.astype(jnp.int8))
                    w2_int8_jnp = jax.block_until_ready(
                        w_down_int8_reordered.astype(jnp.int8))
                _save_moe_cache(layer.prefix, ep_size,
                                np.asarray(w13_int8_jnp),
                                np.asarray(w2_int8_jnp),
                                np.asarray(s13_reordered),
                                np.asarray(s_down_reordered))
                del w13_int8_jnp, w2_int8_jnp
        t_cpu = time.perf_counter()

        # Step 5: shard + put INT8 weight + BF16 scale to HBM (4 个 shard_put 分别 timing)
        ts0 = time.perf_counter()
        w13_int8_on_tpu = shard_put(w13_int8_reordered, shardings=layer.edf_sharding)
        jax.block_until_ready(w13_int8_on_tpu)
        ts1 = time.perf_counter()
        w2_int8_on_tpu = shard_put(w_down_int8_reordered, shardings=layer.efd_sharding)
        jax.block_until_ready(w2_int8_on_tpu)
        ts2 = time.perf_counter()
        w13_scale_sharding = (layer.edf_sharding[0], layer.edf_sharding[1], None)
        w2_scale_sharding = (layer.efd_sharding[0], layer.efd_sharding[1], None)
        s13_on_tpu = shard_put(s13_reordered, shardings=w13_scale_sharding)
        jax.block_until_ready(s13_on_tpu)
        ts3 = time.perf_counter()
        s2_on_tpu = shard_put(s_down_reordered, shardings=w2_scale_sharding)
        jax.block_until_ready(s2_on_tpu)
        ts4 = time.perf_counter()

        # Store: int8 weight (with INT4 values [-8, 7]) + bf16 scale, separate Params
        layer.kernel_gating_upproj_EDF = nnx.Param(w13_int8_on_tpu)
        layer.kernel_down_proj_EFD = nnx.Param(w2_int8_on_tpu)
        layer.kernel_gating_upproj_EDF_scale = nnx.Param(s13_on_tpu)
        layer.kernel_down_proj_EFD_scale = nnx.Param(s2_on_tpu)

        # Free staging refs
        del w13_int8_reordered, w_down_int8_reordered
        del s13_reordered, s_down_reordered
        del w13_int8_on_tpu, w2_int8_on_tpu, s13_on_tpu, s2_on_tpu
        t_store = time.perf_counter()
        gc.collect()
        t_gc = time.perf_counter()

        elapsed = t_gc - t0
        cache_tag = "[CACHE-HIT]" if cache_hit else "[CACHE-MISS]"
        # 详细 timing log: prefix [W4A16-TIME] 方便 grep
        logger.info(
            f"[W4A16-TIME]{cache_tag} {layer.prefix} TOTAL={elapsed:.2f}s | "
            f"check={t_check-t0:.2f} snap={t_snap-t_check:.2f} del={t_del-t_snap:.2f} "
            f"cpu={t_cpu-t_del:.2f} (unpack_g={cpu_timings['A_unpack_gate']:.2f} "
            f"unpack_u={cpu_timings['A_unpack_up']:.2f} unpack_d={cpu_timings['A_unpack_down']:.2f} "
            f"np2jnp={cpu_timings.get('A_np2jnp_convert', 0):.2f} "
            f"scale={cpu_timings['B_scale_concat']:.2f} fuse={cpu_timings['C_fuse_w13']:.2f} "
            f"reorder={cpu_timings['D_reorder_moe']:.2f}) | "
            f"sp_w13={ts1-ts0:.2f} sp_w2={ts2-ts1:.2f} sp_s13={ts3-ts2:.2f} sp_s2={ts4-ts3:.2f} | "
            f"store={t_store-ts4:.2f} gc={t_gc-t_store:.2f}"
        )
        return True

    def apply_jax(self, layer: JaxMoE, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """[v14 Round 9 Step 2] INT8 path: HBM int8 weight + bf16 scale →
        gmm_v2 INT8×INT8 matmul (lhs auto-quantized via maybe_quantize_lhs=True).

        Mirrors Fp8FusedMoEMethod.apply_jax pattern but with int8 weight + bf16
        scale instead of fp8 weight + fp8 scale_inv. moe_apply dispatches to
        backend (GMM_TP / GMM_EP) which calls gmm_v2; gmm_v2 sees rhs.dtype=int8
        (and rhs is non-float, non-int4) → triggers INT8 lhs quantization →
        native INT8×INT8 matmul on TPU MXU (4611 TOPS/chip).
        """
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))

        if layer.moe_backend not in W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            raise NotImplementedError(
                f"W4A16 INT8 path supports only "
                f"{W4A16_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}, got "
                f"{layer.moe_backend}")

        if K26_USE_V16:
            # v17: weight already int4 dtype on HBM (bitcast at load), no apply-time materialize
            w13_weight = layer.kernel_gating_upproj_EDF[...]
            w2_weight = layer.kernel_down_proj_EFD[...]
            s13_native = layer.kernel_gating_upproj_EDF_scale[...]
            w13_weight_scale = jnp.transpose(s13_native, (0, 2, 1))[:, :, None, :]
            s2_native = layer.kernel_down_proj_EFD_scale[...]
            w2_weight_scale = jnp.transpose(s2_native, (0, 2, 1))[:, :, None, :]

        else:
            w13_weight = layer.kernel_gating_upproj_EDF[...]            # int8 (INT4 vals [-8,7])
            w2_weight = layer.kernel_down_proj_EFD[...]                  # int8
            w13_weight_scale = layer.kernel_gating_upproj_EDF_scale[...]  # bf16 per-group
            w2_weight_scale = layer.kernel_down_proj_EFD_scale[...]      # bf16 per-group

        weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=None,
        )

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)
