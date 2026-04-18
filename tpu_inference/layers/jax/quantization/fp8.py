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

import functools
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import PartitionSpec as P

import tpu_inference.envs as envs
from tpu_inference.layers.common.linear import sharded_quantized_batched_matmul
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.linear_weights import \
    shard_linear_weights
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights)
from tpu_inference.layers.common.quantization import fp8 as common_fp8
from tpu_inference.layers.common.utils import cpu_mesh, cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (QuantizationConfig,
                                                           QuantLinearConfig)
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedFusedMoEMethod, UnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    jax_array_from_reshaped_torch, load_nnx_param_from_reshaped_torch,
    shard_put)
from tpu_inference.utils import t2j

logger = init_logger(__name__)

# TODO (jacobplatin): remove once we support all backends
FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS = [
    MoEBackend.GMM_EP, MoEBackend.GMM_TP
]

# --- Parallel MoE requantization ---
# Shared executor for parallel requantization across MoE layers.
# When MOE_PARALLEL_WORKERS > 1, requant jobs are submitted to this pool
# and processed concurrently while weight loading continues.
_moe_requant_executor: ThreadPoolExecutor | None = None
_moe_requant_pending: dict[str, Future] = {}

# Dedicated single-thread executor for async cache writes (fire-and-forget).
# Cache writes are side effects — they should NOT block shard_put to TPU.
_cache_write_executor: ThreadPoolExecutor | None = None


def _get_moe_executor() -> ThreadPoolExecutor | None:
    """Lazily create the thread pool for parallel MoE requantization."""
    global _moe_requant_executor
    num_workers = envs.MOE_PARALLEL_WORKERS
    if num_workers <= 1:
        return None
    if _moe_requant_executor is None:
        _moe_requant_executor = ThreadPoolExecutor(max_workers=num_workers)
        logger.info(
            f"[MoE parallel] Created thread pool with {num_workers} workers")
    return _moe_requant_executor


def _get_cache_write_executor() -> ThreadPoolExecutor:
    """Lazily create a single-thread executor for async cache writes."""
    global _cache_write_executor
    if _cache_write_executor is None:
        _cache_write_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="moe-cache-writer")
        logger.info("[MoE cache] Created async cache writer pool")
    return _cache_write_executor


def _async_save_moe_cache(cache_path: str, weights: FusedMoEWeights,
                          prefix: str) -> None:
    """Save MoE cache asynchronously — does not block the caller."""

    # Materialize JAX arrays to numpy BEFORE submitting to avoid
    # cross-thread JAX issues. This forces lazy evaluation now.
    save_dict = {}
    for name in ["w13_weight", "w13_weight_scale", "w13_bias",
                  "w2_weight", "w2_weight_scale", "w2_bias"]:
        val = getattr(weights, name)
        if val is not None:
            save_dict[name] = np.asarray(val)
            save_dict[f"{name}_dtype"] = np.array(
                str(val.dtype), dtype=object)

    def _do_write():
        t0 = time.perf_counter()
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, **save_dict)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"[MoE cache saved] {prefix} → {cache_path} "
                f"({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"Failed to save MoE cache for {prefix}: {e}")

    _get_cache_write_executor().submit(_do_write)


def _submit_moe_requant(
    prefix: str,
    input_weights: FusedMoEWeights,
    moe_backend: MoEBackend,
    mesh,
    activation: str,
) -> Future:
    """Submit a MoE requantization job to the thread pool."""

    def _do_requant():
        t0 = time.perf_counter()
        with cpu_mesh_context():
            weights = process_fp8_moe_weights(
                input_weights,
                moe_backend=moe_backend,
                mesh=mesh,
                activation=activation,
                weight_block_size=None,
            )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[MoE parallel] {prefix} requantized in {elapsed:.1f}s")
        return weights

    executor = _get_moe_executor()
    future = executor.submit(_do_requant)
    _moe_requant_pending[prefix] = future
    logger.info(
        f"[MoE parallel] Submitted {prefix} "
        f"({len(_moe_requant_pending)} jobs in flight)")
    return future


# --- Weight cache ---


def _get_moe_cache_path(cache_dir: str, layer_prefix: str) -> str:
    """Get the cache file path for a MoE layer's processed weights."""
    safe_name = layer_prefix.replace(".", "_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe_name}.npz")


def _save_moe_cache(cache_path: str, weights: FusedMoEWeights) -> None:
    """Save processed MoE weights to disk as .npz."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {}
    for name in ["w13_weight", "w13_weight_scale", "w13_bias",
                  "w2_weight", "w2_weight_scale", "w2_bias"]:
        val = getattr(weights, name)
        if val is not None:
            save_dict[name] = np.asarray(val)
            save_dict[f"{name}_dtype"] = np.array(
                str(val.dtype), dtype=object)
    np.savez(cache_path, **save_dict)


def _load_moe_cache(cache_path: str) -> FusedMoEWeights | None:
    """Load processed MoE weights from cache. Returns None if not found."""
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        fields = {}
        for name in ["w13_weight", "w13_weight_scale", "w13_bias",
                      "w2_weight", "w2_weight_scale", "w2_bias"]:
            if name in data:
                dtype_str = str(data[f"{name}_dtype"])
                jax_dtype = jnp.dtype(dtype_str)
                fields[name] = jnp.array(data[name], dtype=jax_dtype)
            else:
                fields[name] = None
        return FusedMoEWeights(**fields)
    except Exception as e:
        logger.warning(f"Failed to load MoE cache from {cache_path}: {e}")
        return None


def load_fp8_weight(jax_param: nnx.Param, torch_weight: torch.Tensor,
                    param_name: str):
    """Loads FP8 weights from a torch tensor into a JAX parameter.

    Args:
        jax_param: The nnx parameter to hold the FP8 weight.
        torch_weight: The source PyTorch tensor.
        param_name: Name of the parameter.
    """
    spec = jax_param.sharding
    if isinstance(jax_param.sharding, jax.sharding.NamedSharding):
        spec = jax_param.sharding.spec
    mesh = getattr(jax_param, 'mesh', None)

    jax_weight = t2j(torch_weight, use_dlpack=False)

    if jax_weight.dtype != jnp.float8_e4m3fn:
        logger.warning(
            f"Loading {param_name}: casting from {jax_weight.dtype} to {jax_param[...].dtype}"
        )
        jax_weight = jax_weight.astype(jax_param[...].dtype)

    jax_param.set_raw_value(shard_put(jax_weight, spec, mesh=mesh))


class Fp8TensorwiseLinearMethod(QuantizeMethodBase,
                                common_fp8.Fp8LinearMethod):
    """Tensor-wise Fp8 method for JAX Linear layer."""

    def __init__(self, layer: JaxEinsum, linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)

        self.einsum_str = layer.einsum_str

        self.output_shape = linear_config.out_features
        self.batch_features = linear_config.batch_features
        self.batch_sharding = linear_config.batch_sharding
        out_features = math.prod(self.output_shape)
        in_features = math.prod(linear_config.in_features)
        self.weight_sharding = linear_config.weight_sharding
        if self.batch_features:
            # Batched case: keep original weight sharding for the full
            # 3D weight (matches kernel_shape).
            self.kernel_shape = layer.kernel_shape
        else:
            self.kernel_shape = (out_features, in_features)

        self.in_features = in_features

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)

        layer.weight = create_param(rngs,
                                    shape=self.kernel_shape,
                                    dtype=jnp.float8_e4m3fn,
                                    sharding=self.weight_sharding)

        # Attach custom loader to avoid default upcasting behavior
        layer.weight.set_metadata(
            'weight_loader',
            functools.partial(load_fp8_weight,
                              param_name=layer.prefix + ".weight"))

        # Scale is always per-output-channel (1D).
        scale_sharding = None
        if self.batch_features:
            # For batched weights, the output dim sharding comes from
            # the weight's non-contracting, non-batch axis.
            if self.batch_sharding:
                scale_sharding = None  # replicated scale for simplicity
        elif isinstance(self.weight_sharding, P) and len(
                self.weight_sharding) > 0:
            scale_sharding = P(self.weight_sharding[0])
        elif isinstance(self.weight_sharding,
                        (tuple, list)) and len(self.weight_sharding) > 0:
            scale_sharding = (self.weight_sharding[0], )

        layer.weight_scale = create_param(rngs,
                                          shape=(out_features, ),
                                          dtype=jnp.float32,
                                          sharding=scale_sharding)

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        bias = layer.bias[...] if layer.bias is not None else None

        if self.batch_features:
            # Batched case: use dot_general with batch dims.
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight[...],
                layer.weight_scale[...],
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            if bias is not None:
                out += bias
            return out

        out = self._apply_fused(x,
                                layer.weight[...],
                                layer.weight_scale[...],
                                bias=bias)
        out = out.reshape(out.shape[:-1] + self.output_shape)
        return out


class Fp8BlockwiseLinearMethod(QuantizeMethodBase, common_fp8.Fp8LinearMethod):
    """Block-wise Fp8 method for JAX Linear layer."""

    def __init__(self, quant_config: "Fp8Config", layer: JaxEinsum,
                 linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config
        self.einsum_str = layer.einsum_str

        self.out_features = linear_config.out_features
        self.in_features = math.prod(linear_config.in_features)
        self.batch_features = linear_config.batch_features
        self.batch_sharding = linear_config.batch_sharding
        self.weight_sharding = linear_config.weight_sharding
        self.bias_sharding = linear_config.bias_sharding
        if self.batch_features:
            # Batched case: keep original weight sharding for the full
            # 3D weight (matches kernel_shape).
            self.kernel_shape = layer.kernel_shape
        else:
            self.kernel_shape = (math.prod(self.out_features),
                                 self.in_features)

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)
        kernel_init = layer.kernel_init

        if self.batch_features:
            # Batched case: create weight with the original 3D kernel shape
            # so the weight loader can populate it directly after transpose.
            # Weight stays in FP8 and is used with sharded_quantized_batched_matmul.
            param_dtype = jnp.float8_e4m3
            layer.weight = nnx.Param(
                nnx.initializers.uniform()(rngs.params(), self.kernel_shape,
                                           param_dtype),
                weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                      permute_dims=None,
                                      param_name=layer.prefix + ".weight"),
                eager_sharding=False)
            layer.weight.set_metadata('sharding', self.weight_sharding)

            # Per-output-channel scale (1D, covers the free weight dim).
            layer.weight_scale_inv = nnx.Param(
                jnp.ones((out_features, ), dtype=layer.dtype),
                weight_loader=partial(
                    load_nnx_param_from_reshaped_torch,
                    permute_dims=None,
                    param_name=layer.prefix + ".weight_scale_inv",
                ),
                eager_sharding=False)
            layer.weight_scale_inv.set_metadata('sharding', ())
            return

        # Follow upstream limitation that only float8_e4m3 is supported.
        # https://github.com/vllm-project/vllm/blob/2a99c5a6c86daef8c766ba2dbf05c385b192c64b/vllm/model_executor/layers/quantization/fp8.py#L283-L284
        param_dtype = jnp.float8_e4m3
        layer.weight = nnx.Param(
            kernel_init(rngs.params(), self.kernel_shape, param_dtype),
            weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                  permute_dims=(0, 1),
                                  param_name=layer.prefix + ".weight"),
            eager_sharding=False)
        layer.weight.set_metadata('sharding', self.weight_sharding)

        # Block-wise quantization scale
        block_n, block_k = self.quant_config.weight_block_size[
            0], self.quant_config.weight_block_size[1]
        layer.weight_scale_inv = nnx.Param(
            kernel_init(
                rngs.params(),
                [(out_features + block_n - 1) // block_n,
                 (self.in_features + block_k - 1) // block_k],
                layer.dtype,
            ),
            weight_loader=partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=(0, 1),
                param_name=layer.prefix + ".weight_scale_inv",
            ),
            eager_sharding=False)
        layer.weight_scale_inv.set_metadata('sharding', self.weight_sharding)

        # Force the parameters to be loaded onto CPU, such that in `process_weights_after_loading`
        # we can process the weights on CPU to avoid OOM on device.
        layer.weight.set_metadata('mesh', cpu_mesh())
        layer.weight_scale_inv.set_metadata('mesh', cpu_mesh())
        if layer.bias is not None:
            layer.bias.set_metadata('mesh', cpu_mesh())

    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        if self.batch_features:
            # Batched case: weight stays in FP8. No blockwise processing
            # needed — the batched matmul uses dot_general with FP8 natively.
            return True

        if not layer.weight.get_metadata(
                "_is_loaded",
                False) or not layer.weight_scale_inv.get_metadata(
                    "_is_loaded", False):
            # Weight and scale could spread across multiple files,
            # so we only process once both of them are loaded.
            return False

        # Do the re-quant process on CPU to avoid OOM on device.
        with cpu_mesh_context():
            weight = layer.weight[...]
            weight_scale_inv = layer.weight_scale_inv[...]
            bias = layer.bias[...] if getattr(layer, 'bias',
                                              None) is not None else None
            if bias is not None:
                bias = bias.reshape(-1)
            weights = common_fp8.process_blockwise_fp8_linear_weights(
                weight,
                weight_scale_inv,
                bias=bias,
                weight_block_size=tuple(self.quant_config.weight_block_size),
                requant_block_size=self.linear_config.requant_block_size,
                output_sizes=tuple(self.linear_config.output_sizes),
                requant_weight_dtype=self.linear_config.requant_weight_dtype,
                fuse_matmuls=self.linear_config.fuse_matmuls,
                n_shards=self.linear_config.n_shards)
            delattr(layer, 'weight')
            delattr(layer, 'weight_scale_inv')
            delattr(layer, 'bias')

            if self.linear_config.enable_quantized_matmul_kernel:
                # The quantized_matmul_kernel expects weight scales shaped (n_out_features, 1, n_blocks) for blockwisze quantization.
                weights.weight_scale = jnp.expand_dims(
                    jnp.transpose(weights.weight_scale),
                    axis=1,
                )

        # Put onto the device.
        weights = shard_linear_weights(
            weights,
            mesh=None,
            weight_p_spec=self.linear_config.weight_sharding,
            bias_p_spec=self.linear_config.bias_sharding,
        )
        if self.linear_config.fuse_matmuls:
            layer.weight = nnx.Param(weights.weight)
            layer.weight_scale_inv = nnx.Param(weights.weight_scale)
            layer.bias = nnx.Param(weights.bias) if bias is not None else None
        else:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        if self.batch_features:
            # Batched case: use dot_general with FP8 and batch dims.
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight[...],
                layer.weight_scale_inv[...],
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            return out

        if not self.linear_config.fuse_matmuls:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")
        weight, scale = layer.weight[...], layer.weight_scale_inv[...]
        bias = layer.bias[...] if layer.bias is not None else None
        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        out = self._apply_fused(x, weight, scale, bias=bias)
        out = out.reshape(out.shape[:-1] + self.out_features)
        return out


class Fp8FusedMoEMethod(QuantizeMethodBase):
    """
    Fp8 method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, weight_block_size: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}
        self.weight_block_size = weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")

    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,
                     weights: Iterable) -> set:
        """Load scale paramters and delegate the weight paramters to `original_load_weights_fn`"""

        # Remaining non-scale parameters will be loaded using original load_weights function.
        remaining_weights = dict()
        cnt = 0
        for torch_name, torch_weight in weights:
            torch_name: str = torch_name.split(
                layer.prefix)[-1]  # ".0.down_proj.weight" for example
            names = torch_name.split(".")
            assert len(
                names
            ) == 3, f"Expected param name to be .<expert_id>.<param_name>.weight, got {torch_name}"
            expert_id, _, _ = names
            expert_id = int(expert_id)
            jax_param_name = ""
            if torch_name.endswith("up_proj." + self.weight_scale_name):
                jax_param_name = "kernel_up_proj_EDF_" + self.weight_scale_name
            elif torch_name.endswith("down_proj." + self.weight_scale_name):
                jax_param_name = "kernel_down_proj_EFD_" + self.weight_scale_name
            elif torch_name.endswith("gate_proj." + self.weight_scale_name):
                jax_param_name = "kernel_gating_EDF_" + self.weight_scale_name
            else:
                remaining_weights[torch_name] = torch_weight
                continue
            cnt += 1
            jax_param = getattr(layer, jax_param_name, None)

            assert isinstance(jax_param, nnx.Param)

            # Here we rely on `jax_array_from_reshaped_torch` to load weights
            # onto CPU and prepend a leading dimension for expert_id, because
            # later in `process_weights_after_loading` the sharded experts
            # will be concatenated altogether then put onto the device.
            jax_weight = jax_array_from_reshaped_torch(torch_weight,
                                                       reshape_dims=(1, ) +
                                                       torch_weight.shape)
            jax_param._weights_to_load[expert_id] = jax_weight

        logger.debug(
            f"Loaded {cnt} weight scales for {layer.prefix} MoE layer.")

        loaded_names = original_load_weights_fn(remaining_weights.items())
        for param_name in {
                "kernel_gating_EDF_" + self.weight_scale_name,
                "kernel_up_proj_EDF_" + self.weight_scale_name,
                "kernel_down_proj_EFD_" + self.weight_scale_name,
        }:
            param = getattr(layer, param_name)
            if all(w is not None for w in param._weights_to_load):
                loaded_names.add(param_name)

        return loaded_names

    def create_weights_jax(self, layer: JaxMoE, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """
        Create the quant method-specific weights.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to create weights for.
        """

        quant_config = layer.quant_config
        assert isinstance(
            quant_config,
            Fp8Config), "Expected fp8 config for Fp8FusedMoEMethod!"

        # TODO (#1681): support other backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            # vLLM reference here:
            # https://github.com/vllm-project/vllm/blob/9bdb06b/vllm/model_executor/layers/quantization/fp8.py#L763
            if not self.block_quant:
                raise NotImplementedError(
                    "Expected blockwise quantization when using Fp8FusedMoEMethod!"
                )
            else:
                assert len(
                    self.weight_block_size
                ) == 2, f"Expected 2D block size, got {self.weight_block_size}"
                block_n, block_k = self.weight_block_size

                # re-create the weights to be in fp8 type
                for param_name in [
                        "kernel_gating_EDF", "kernel_up_proj_EDF",
                        "kernel_down_proj_EFD"
                ]:
                    param = getattr(layer, param_name, None)
                    assert isinstance(
                        param, nnx.Param
                    ), f"Expected nnx.Param for {param_name}, got {type(param)}"
                    init_fn = param.init_fn
                    E, K, N = param[...].shape
                    value = init_fn(rngs.params(), (E, K, N),
                                    jnp.float8_e4m3fn)
                    param.set_raw_value(value)

                    scale_value = jnp.zeros((E, (K + block_k - 1) // block_k,
                                             (N + block_n - 1) // block_n),
                                            device=jax.devices('cpu')[0])
                    setattr(
                        layer, f"{param_name}_{self.weight_scale_name}",
                        nnx.Param(scale_value,
                                  _weights_to_load=[None for _ in range(E)]))
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def process_weights_after_loading(self, layer: JaxMoE) -> bool:
        """
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Supports two optimizations controlled by environment variables:

        1. **Parallel requantization** (MOE_PARALLEL_WORKERS > 1):
           Multiple MoE layers are requantized concurrently via a shared
           ThreadPoolExecutor. This leverages the fact that load_hf_weights
           already uses multi-threaded file loading — each file-loading
           thread submits its layer's requant to the shared pool and
           block-waits for the result, allowing up to MOE_PARALLEL_WORKERS
           layers to be processed simultaneously.

        2. **Weight caching** (MOE_WEIGHT_CACHE_DIR set):
           Processed FP4 weights are saved to disk after the first
           requantization. On subsequent startups, cached weights are
           loaded directly, skipping the expensive CPU requantization.

        Args:
            layer: The layer to process.
        """
        # TODO (#1681): support other backends

        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            gating_scale_name = f"kernel_gating_EDF_{self.weight_scale_name}"
            up_scale_name = f"kernel_up_proj_EDF_{self.weight_scale_name}"
            down_scale_name = f"kernel_down_proj_EFD_{self.weight_scale_name}"

            if any(
                    any(w is None for w in param._weights_to_load) for param in
                [
                    getattr(layer, gating_scale_name),
                    getattr(layer, up_scale_name),
                    getattr(layer, down_scale_name), layer.kernel_gating_EDF,
                    layer.kernel_up_proj_EDF, layer.kernel_down_proj_EFD
                ]):
                # If weights for a module is spread across multiple files,
                # this function may be called more than once. We only want
                # to process the weights once all of them are loaded.
                return False

            # Check if we can load from cache
            cache_dir = envs.MOE_WEIGHT_CACHE_DIR
            cache_path = (_get_moe_cache_path(cache_dir, layer.prefix)
                          if cache_dir else None)
            cached_weights = (_load_moe_cache(cache_path)
                              if cache_path else None)

            t0 = time.perf_counter()

            if cached_weights is not None:
                logger.info(
                    f"[MoE cache hit] Loading pre-processed weights for "
                    f"{layer.prefix} from {cache_path}")
                weights = cached_weights
            else:
                # Prepare fused input weights from per-expert tensors
                with cpu_mesh_context():
                    w_gate = jnp.concatenate(
                        layer.kernel_gating_EDF._weights_to_load, axis=0)
                    w_up = jnp.concatenate(
                        layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                    s_gate = jnp.concatenate(getattr(
                        layer, gating_scale_name)._weights_to_load,
                                             axis=0)
                    s_up = jnp.concatenate(getattr(
                        layer, up_scale_name)._weights_to_load,
                                           axis=0)
                    w2_weight = jnp.concatenate(
                        layer.kernel_down_proj_EFD._weights_to_load, axis=0)
                    w2_weight_scale = jnp.concatenate(getattr(
                        layer, down_scale_name)._weights_to_load,
                                                      axis=0)

                    # Fuse the weights into w13: [Gate, Up]. w2 is expected
                    # to be (num_experts, hidden_size, intermediate_size),
                    # w13 is expected to be
                    # (num_experts, 2 * intermediate_size, hidden_size)
                    w13_weight = jnp.concatenate([w_gate, w_up], axis=1)
                    w13_weight_scale = jnp.concatenate(
                        [s_gate, s_up], axis=1)

                    # TODO (jacobplatin): we should support bias
                    input_weights = FusedMoEWeights(
                        w13_weight=w13_weight,
                        w13_weight_scale=w13_weight_scale,
                        w13_bias=None,
                        w2_weight=w2_weight,
                        w2_weight_scale=w2_weight_scale,
                        w2_bias=None)

                # Requantize: parallel or serial
                executor = _get_moe_executor()
                if executor is not None:
                    # Parallel path: submit to shared pool and block-wait.
                    # The file-loading ThreadPoolExecutor (up to 64 threads)
                    # may call this method for many layers concurrently.
                    # Each thread submits its requant job here; the shared
                    # pool runs up to MOE_PARALLEL_WORKERS at a time.
                    logger.info(
                        f"[MoE parallel] {layer.prefix} — submitting "
                        f"requant job")
                    future = _submit_moe_requant(
                        layer.prefix, input_weights, layer.moe_backend,
                        layer.mesh, layer.activation)
                    weights = future.result()  # Block until done
                    _moe_requant_pending.pop(layer.prefix, None)
                else:
                    # Serial path: process in current thread
                    if cache_dir:
                        logger.info(
                            f"[MoE cache miss] {layer.prefix} — "
                            f"requantizing (serial)")
                    with cpu_mesh_context():
                        weights = process_fp8_moe_weights(
                            input_weights,
                            moe_backend=layer.moe_backend,
                            mesh=layer.mesh,
                            activation=layer.activation,
                            weight_block_size=None,
                        )

                # Save cache asynchronously — fire and forget.
                # np.asarray() materializes JAX lazy arrays now (in
                # the calling thread), then the actual disk write
                # happens in a background thread without blocking
                # shard_put to TPU.
                if cache_path:
                    _async_save_moe_cache(
                        cache_path, weights, layer.prefix)

            # Replace intermediate per-expert params with processed weights
            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF
            delattr(layer, gating_scale_name)
            delattr(layer, up_scale_name)

            # TODO (jacobplatin): we probably want to make the sharding
            # configurable
            layer.kernel_gating_upproj_EDF = nnx.Param(
                shard_put(weights.w13_weight, shardings=layer.edf_sharding))
            layer.kernel_down_proj_EFD = nnx.Param(
                shard_put(weights.w2_weight, shardings=layer.efd_sharding))
            # gmm expects shape [num_groups, num_blocks, 1, n]
            # TODO(gpolovets1): Make sure it works for gmm_v2 as well.
            edf_scale_sharding = (layer.edf_sharding[0], ) + (None, ) * (
                weights.w13_weight_scale.ndim - 2) + (
                    layer.edf_sharding[-1], )
            efd_scale_sharding = (layer.efd_sharding[0], ) + (None, ) * (
                weights.w2_weight_scale.ndim - 2) + (
                    layer.efd_sharding[-1], )
            setattr(
                layer, f"kernel_gating_upproj_EDF_{self.weight_scale_name}",
                nnx.Param(
                    shard_put(weights.w13_weight_scale,
                              shardings=edf_scale_sharding)))
            setattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}",
                nnx.Param(
                    shard_put(weights.w2_weight_scale,
                              shardings=efd_scale_sharding)))

            elapsed = time.perf_counter() - t0
            label = "cached" if cached_weights else "requantized"
            logger.info(
                f"[MoE weights] {layer.prefix} processed in {elapsed:.1f}s"
                f" ({label})")
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! "
                f"Currently supported: "
                f"{FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}")

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """
        Run the forward pass of the MoE layer.

        Args:
            layer: The layer to apply the quantization method to.
            x: The input to the layer.

        Returns:
            The MoE output.
        """
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))

        # Fused weight backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            # router_logits is of shape TE -- we don't return the indices

            if layer.moe_backend == MoEBackend.FUSED_MOE:
                w13_weight = layer.kernel_gating_upproj_E2DF[...]
            else:
                w13_weight = layer.kernel_gating_upproj_EDF[...]
            w2_weight = layer.kernel_down_proj_EFD[...]
            w13_weight_scale = getattr(
                layer,
                f"kernel_gating_upproj_EDF_{self.weight_scale_name}")[...]

            w2_weight_scale = getattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}")[...]

            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=w13_weight_scale,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=w2_weight_scale,
                w2_bias=None,
            )
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


class Fp8Config(QuantizationConfig):

    ACTIVATION_SCHEMES = ["dynamic", "static"]

    def __init__(self, hf_quant_config: dict):
        # Replicating upstream https://github.com/vllm-project/vllm/blob/77c09e1130661197ccac2d968a28cd4a557922d5/vllm/model_executor/layers/quantization/fp8.py#L167-L175

        quant_method = self.get_from_keys(hf_quant_config, ["quant_method"])
        self.is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = self.get_from_keys(hf_quant_config,
                                               ["activation_scheme"])
        ignored_layers = self.get_from_keys(hf_quant_config,
                                            ["ignored_layers"], None)
        weight_block_size = self.get_from_keys(hf_quant_config,
                                               ["weight_block_size"], None)
        if not ignored_layers:
            ignored_layers = self.get_from_keys(hf_quant_config,
                                                ["modules_to_not_convert"],
                                                None)

        if activation_scheme not in self.ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not self.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        self.weight_block_size = weight_block_size

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer, enable_sp=False)
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers):
                return UnquantizedLinearMethod(linear_config)
            if self.weight_block_size is not None:
                return Fp8BlockwiseLinearMethod(self, layer, linear_config)
            else:
                return Fp8TensorwiseLinearMethod(layer, linear_config)
        elif isinstance(layer, JaxMoE):
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers):
                return UnquantizedFusedMoEMethod()
            return Fp8FusedMoEMethod(self.weight_block_size)
        return None
