#!/usr/bin/env python3
"""K2.6 MoE weight offline cache builder.

Reads safetensors, concats 384 experts per layer, fuses w13, saves as
mmap-friendly .npy directory format compatible with envs.MOE_WEIGHT_CACHE_DIR.

Cache stores PACKED uint32 (NOT unpacked int4) to keep V16 fast path
HBM bitcast working at runtime — cache hit -> mmap -> device_put -> bitcast.

Usage:
    # Build single layer (smoke test)
    python build_k26_moe_cache.py --layers 1

    # Build range
    python build_k26_moe_cache.py --layers 1-3

    # Build all 60 MoE layers
    python build_k26_moe_cache.py --layers 1-60
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import ml_dtypes  # bfloat16 numpy support
import numpy as np
import torch
from safetensors import safe_open

# --- defaults ---
# When K2.6 model already staged in SHM (e.g., by vllm), read directly from there
# (saves 130 GB SHM + skips redundant Lustre->SHM staging).
SAFETENSORS_DIR = (
    "/dev/shm/Kimi-K2.6"
    if os.path.exists("/dev/shm/Kimi-K2.6/model.safetensors.index.json")
    else "/lustre/Kimi-K2.6"
)
SHM_DIR = SAFETENSORS_DIR  # No separate staging — read source directly
CACHE_DIR = "/lustre/k26_cache_v2"
KEY_PREFIX = "language_model.model.layers"  # K2.6 multimodal prefix
N_EXPERTS = 384


def _unpack_transpose_repack_w13(w_packed: np.ndarray) -> np.ndarray:
    """[E, 2F, D/8] uint32 → [E, D, 2F/8] uint32, with full transpose.

    Steps (mirror of runtime V16 path, moved offline):
      1. unpack uint32 → uint4 nibbles → int8 in [-8, 7]
      2. transpose (0, 2, 1) so gmm kernel sees [E, D, 2F]
      3. repack int8 → uint32 (8 nibbles per uint32)

    Saves runtime ~22 s/layer (transpose + with_layout_constraint).
    """
    E, two_F, D_p8 = w_packed.shape
    D = D_p8 * 8

    # Unpack uint32 → int8 [E, 2F, D]
    packed_u8 = w_packed.view(np.uint8)  # [E, 2F, D/8 * 4] uint8
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    unpacked_FD = np.empty((E, two_F, D), dtype=np.int8)
    unpacked_FD[..., 0::2] = low
    unpacked_FD[..., 1::2] = high
    unpacked_FD -= 8
    del packed_u8, low, high

    # Transpose [E, 2F, D] → [E, D, 2F], force contiguous for view
    unpacked_DF = np.ascontiguousarray(unpacked_FD.transpose(0, 2, 1))
    del unpacked_FD

    # Repack int8 → uint32 (low/high nibble within each byte, 4 bytes per uint32)
    # Step a: shift to unsigned [0, 15]
    unsigned_DF = (unpacked_DF + 8).astype(np.uint8)
    del unpacked_DF
    # Step b: pack 2 nibbles per byte: byte = (high << 4) | low
    low_n = unsigned_DF[..., 0::2]
    high_n = unsigned_DF[..., 1::2]
    packed_u8_DF = ((high_n << 4) | low_n)  # [E, D, 2F/2] uint8
    del unsigned_DF, low_n, high_n
    # Step c: view 4 bytes as uint32 → [E, D, 2F/8] uint32
    repacked = np.ascontiguousarray(packed_u8_DF).view(np.uint32)
    repacked = repacked.reshape(E, D, two_F // 8)
    return repacked


def _unpack_transpose_repack_w_down(w_packed: np.ndarray) -> np.ndarray:
    """[E, D, F/8] uint32 → [E, F, D/8] uint32, with full transpose.

    Same idea as w13, different starting layout.
    """
    E, D, F_p8 = w_packed.shape
    F = F_p8 * 8

    packed_u8 = w_packed.view(np.uint8)  # [E, D, F/8 * 4] uint8
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    unpacked_DF = np.empty((E, D, F), dtype=np.int8)
    unpacked_DF[..., 0::2] = low
    unpacked_DF[..., 1::2] = high
    unpacked_DF -= 8
    del packed_u8, low, high

    unpacked_FD = np.ascontiguousarray(unpacked_DF.transpose(0, 2, 1))
    del unpacked_DF

    unsigned_FD = (unpacked_FD + 8).astype(np.uint8)
    del unpacked_FD
    low_n = unsigned_FD[..., 0::2]
    high_n = unsigned_FD[..., 1::2]
    packed_u8_FD = ((high_n << 4) | low_n)  # [E, F, D/2] uint8
    del unsigned_FD, low_n, high_n
    repacked = np.ascontiguousarray(packed_u8_FD).view(np.uint32)
    repacked = repacked.reshape(E, F, D // 8)
    return repacked


def parse_layers(spec: str) -> list[int]:
    """Parse '1' or '1-3' or '1,3,5' into [1, 2, 3] etc."""
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def find_shards_for_layers(index_path: str, layers: list[int]) -> set[str]:
    """Return set of shard filenames containing any expert key for the given layers."""
    with open(index_path) as f:
        index = json.load(f)
    needed = set()
    layer_prefixes = [f"{KEY_PREFIX}.{L}.mlp.experts." for L in layers]
    for key, shard in index["weight_map"].items():
        if any(key.startswith(p) for p in layer_prefixes):
            needed.add(shard)
    return needed


def stage_shards_to_shm(shard_names: set[str]):
    """Copy needed shards from Lustre to SHM. Returns SHM dir path."""
    if SHM_DIR == SAFETENSORS_DIR:
        print(f"[stage] SHM_DIR == SAFETENSORS_DIR ({SHM_DIR}), skip staging")
        return
    os.makedirs(SHM_DIR, exist_ok=True)
    print(f"[stage] Staging {len(shard_names)} shards Lustre -> SHM")
    t0 = time.perf_counter()
    total_bytes = 0
    for name in sorted(shard_names):
        src = os.path.join(SAFETENSORS_DIR, name)
        dst = os.path.join(SHM_DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"[stage]   skip {name} (already in SHM)")
            total_bytes += os.path.getsize(src)
            continue
        sz = os.path.getsize(src)
        ts = time.perf_counter()
        shutil.copyfile(src, dst)
        elapsed = time.perf_counter() - ts
        print(f"[stage]   {name}  {sz/1e9:.2f} GB  {elapsed:.1f}s  "
              f"({sz/1e9/elapsed:.2f} GB/s)")
        total_bytes += sz
    t1 = time.perf_counter()
    print(f"[stage] Done. {total_bytes/1e9:.2f} GB in {t1-t0:.1f}s "
          f"({total_bytes/1e9/(t1-t0):.2f} GB/s avg)")


def build_layer(layer_idx: int, ep_size: int, force: bool = False):
    """Build one layer's MoE cache from SHM safetensors.

    Idempotent: skips if meta.json already exists (use --force to rebuild).
    """
    layer_t0 = time.perf_counter()
    layer_cache_dir = os.path.join(
        CACHE_DIR, f"model_layers_{layer_idx}_mlp_experts")
    meta_path = os.path.join(layer_cache_dir, "meta.json")
    if not force and os.path.exists(meta_path):
        print(f"[layer {layer_idx}] cache already exists, skip "
              f"(use --force to rebuild)")
        return
    print(f"\n[layer {layer_idx}] === starting ===")

    # 1. Find which SHM shards have this layer
    index_path = os.path.join(SAFETENSORS_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)["weight_map"]

    layer_prefix = f"{KEY_PREFIX}.{layer_idx}.mlp.experts."

    # Group keys by op (gate/up/down) and attr (packed/scale)
    # Per expert, per op, packed is uint32 [F, D/8] or [D, F/8]
    expert_data = {
        "gate": [None] * N_EXPERTS,
        "up": [None] * N_EXPERTS,
        "down": [None] * N_EXPERTS,
    }
    expert_scale = {
        "gate": [None] * N_EXPERTS,
        "up": [None] * N_EXPERTS,
        "down": [None] * N_EXPERTS,
    }

    # Group keys by shard file for batched safe_open
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in index.items():
        if not key.startswith(layer_prefix):
            continue
        shard_to_keys.setdefault(shard, []).append(key)

    if not shard_to_keys:
        print(f"[layer {layer_idx}] no MoE keys found - dense layer? skip")
        return

    # 2. Read all expert weights from SHM
    # safetensors framework="numpy" can't handle bfloat16, so use "pt" + view+numpy.
    t_read0 = time.perf_counter()
    for shard, keys in shard_to_keys.items():
        shm_path = os.path.join(SHM_DIR, shard)
        with safe_open(shm_path, framework="pt") as f:
            for key in keys:
                # key form: language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed
                parts = key.split(".")
                expert_idx = int(parts[6])  # experts.X
                op = parts[7].split("_")[0]  # gate_proj -> gate
                attr = parts[-1]  # weight_packed / weight_scale / weight_shape
                t = f.get_tensor(key)
                if attr == "weight_packed":
                    # uint32, direct numpy
                    expert_data[op][expert_idx] = t.numpy()
                elif attr == "weight_scale":
                    # bfloat16: torch can't .numpy() directly. View as uint16
                    # to preserve bytes, then reinterpret as ml_dtypes.bfloat16.
                    arr_u16 = t.view(torch.uint16).numpy()
                    expert_scale[op][expert_idx] = arr_u16.view(
                        ml_dtypes.bfloat16)
                # weight_shape is metadata, skip
    t_read1 = time.perf_counter()

    # 3. Concat 384 experts per op (axis=0)
    t_concat0 = time.perf_counter()
    # Each expert weight_packed shape [F, D/8] uint32 (gate/up) or [D, F/8] (down)
    # Stack along new leading dim -> [E=384, F, D/8]
    w_gate = np.stack(expert_data["gate"], axis=0)
    w_up = np.stack(expert_data["up"], axis=0)
    w_down = np.stack(expert_data["down"], axis=0)
    s_gate = np.stack(expert_scale["gate"], axis=0)
    s_up = np.stack(expert_scale["up"], axis=0)
    s_down = np.stack(expert_scale["down"], axis=0)
    t_concat1 = time.perf_counter()

    # 4. Fuse w13 = concat(gate, up) along axis=1 (F dim)
    t_fuse0 = time.perf_counter()
    w13_packed = np.concatenate([w_gate, w_up], axis=1)
    s13 = np.concatenate([s_gate, s_up], axis=1)
    del w_gate, w_up, s_gate, s_up
    t_fuse1 = time.perf_counter()

    # 5. Free expert_data to release SHM
    del expert_data, expert_scale

    # 5b. v18-transposed-cache: pre-transpose so runtime skips transpose+layout_constraint
    t_xpose0 = time.perf_counter()
    w13_packed = _unpack_transpose_repack_w13(w13_packed)
    w_down = _unpack_transpose_repack_w_down(w_down)
    t_xpose1 = time.perf_counter()

    # 6. Save .npy + meta.json (layer_cache_dir computed at function entry)
    os.makedirs(layer_cache_dir, exist_ok=True)
    t_save0 = time.perf_counter()
    np.save(f"{layer_cache_dir}/w13_weight_packed.npy", w13_packed)
    np.save(f"{layer_cache_dir}/w13_weight_scale.npy", s13)
    np.save(f"{layer_cache_dir}/w2_weight_packed.npy", w_down)
    np.save(f"{layer_cache_dir}/w2_weight_scale.npy", s_down)
    meta = {
        "_cache_format": "k26_packed_uint32_v1",
        "ep_size": ep_size,
        "n_experts": N_EXPERTS,
        "layer_idx": layer_idx,
        "w13_weight_packed_shape": list(w13_packed.shape),
        "w13_weight_packed_dtype": str(w13_packed.dtype),
        "w13_weight_scale_shape": list(s13.shape),
        "w13_weight_scale_dtype": str(s13.dtype),
        "w2_weight_packed_shape": list(w_down.shape),
        "w2_weight_packed_dtype": str(w_down.dtype),
        "w2_weight_scale_shape": list(s_down.shape),
        "w2_weight_scale_dtype": str(s_down.dtype),
    }
    with open(f"{layer_cache_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    t_save1 = time.perf_counter()

    # Sizes
    total_gb = sum(os.path.getsize(f"{layer_cache_dir}/{n}")
                   for n in os.listdir(layer_cache_dir)) / 1e9

    print(f"[layer {layer_idx}] read={t_read1-t_read0:.1f}s "
          f"stack={t_concat1-t_concat0:.1f}s "
          f"fuse={t_fuse1-t_fuse0:.1f}s "
          f"xpose={t_xpose1-t_xpose0:.1f}s "
          f"save={t_save1-t_save0:.1f}s "
          f"total={time.perf_counter()-layer_t0:.1f}s "
          f"size={total_gb:.2f} GB "
          f"shapes: w13={w13_packed.shape}, w2={w_down.shape}, s13={s13.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", required=True,
                        help="e.g. '1' or '1-3' or '1,3,5' or '1-60'")
    parser.add_argument("--ep-size", type=int, default=16,
                        help="EP size for cache (just metadata, content same)")
    parser.add_argument("--skip-stage", action="store_true",
                        help="Skip Lustre->SHM staging (for re-runs)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if cache already exists")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    print(f"=== K2.6 MoE cache builder ===")
    print(f"Layers: {layers}")
    print(f"EP size: {args.ep_size}")
    print(f"SHM staging: {SHM_DIR}")
    print(f"Cache out: {CACHE_DIR}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Find shards needed
    index_path = os.path.join(SAFETENSORS_DIR, "model.safetensors.index.json")
    needed_shards = find_shards_for_layers(index_path, layers)
    print(f"Needed shards: {len(needed_shards)}")

    # 2. Stage to SHM
    if not args.skip_stage:
        stage_shards_to_shm(needed_shards)

    # 3. Build each layer
    overall_t0 = time.perf_counter()
    for L in layers:
        build_layer(L, args.ep_size, force=args.force)
    overall_t1 = time.perf_counter()

    print(f"\n=== DONE: {len(layers)} layers in {overall_t1-overall_t0:.1f}s "
          f"({(overall_t1-overall_t0)/len(layers):.1f}s/layer avg) ===")


if __name__ == "__main__":
    main()
