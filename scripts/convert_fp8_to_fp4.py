#!/usr/bin/env python3
"""Convert FP8 MoE cache to native FP4 via proper requantization (CPU-only).

Reads FP8 npy cache, dequantizes to FP32, recomputes per-channel scale,
and quantizes to float4_e2m1fn. Output is stored as native FP4 with
meta.json marking `_storage_format: native_fp4`.

Usage:
  python3 convert_fp8_to_fp4.py --fp8-cache /path/to/fp8_cache --fp4-cache /path/to/fp4_output
  python3 convert_fp8_to_fp4.py --fp8-cache /path/to/fp8_cache --fp4-cache /path/to/fp4_output --workers 8
"""
import argparse
import gc
import json
import os
import time

import ml_dtypes
import numpy as np
from concurrent.futures import ThreadPoolExecutor

fp4_max = float(ml_dtypes.finfo(ml_dtypes.float4_e2m1fn).max)
fp4_min = float(ml_dtypes.finfo(ml_dtypes.float4_e2m1fn).min)


def load_npy(path, dtype_str=""):
    """Load a .npy file, applying dtype view for void types."""
    arr = np.load(path)
    if arr.dtype.kind == "V":
        dt = (getattr(ml_dtypes, dtype_str, None)
              if dtype_str else ml_dtypes.float8_e4m3fn)
        if dt:
            arr = arr.view(dt)
    return arr


def convert_weight(w_fp8, scale):
    """FP8 -> dequant(FP32) -> requant(FP4). Returns (fp4_native, new_scale)."""
    sq = scale.reshape(scale.shape[0], 1, scale.shape[-1])
    fp32 = w_fp8.astype(np.float32) * sq
    del w_fp8
    abs_max = np.max(np.abs(fp32), axis=1, keepdims=True)
    new_scale = np.where(abs_max == 0, 1.0, abs_max / fp4_max)
    fp4 = np.clip(fp32 / new_scale, fp4_min, fp4_max).astype(
        ml_dtypes.float4_e2m1fn)
    del fp32
    new_scale_4d = new_scale.reshape(scale.shape).astype(np.float32)
    return fp4, new_scale_4d


def convert_layer(args):
    """Convert a single MoE layer from FP8 to native FP4."""
    layer_dir, fp4_cache = args
    layer_name = os.path.basename(layer_dir)
    t0 = time.perf_counter()

    with open(os.path.join(layer_dir, "meta.json")) as f:
        meta = json.load(f)

    out_dir = os.path.join(fp4_cache, layer_name)
    os.makedirs(out_dir, exist_ok=True)

    for wn in ["w13_weight", "w2_weight"]:
        sn = f"{wn}_scale"
        dtype_str = meta.get(f"{wn}_dtype", "float8_e4m3fn")

        w = load_npy(os.path.join(layer_dir, f"{wn}.npy"), dtype_str)
        s = np.load(os.path.join(layer_dir, f"{sn}.npy"))

        fp4, new_scale = convert_weight(w, s)
        del w, s

        np.save(os.path.join(out_dir, f"{wn}.npy"), fp4)
        np.save(os.path.join(out_dir, f"{sn}.npy"), new_scale)
        del fp4, new_scale
        gc.collect()

    # Update meta.json with FP4 dtype and storage format marker
    nm = dict(meta)
    nm["w13_weight_dtype"] = "float4_e2m1fn"
    nm["w2_weight_dtype"] = "float4_e2m1fn"
    nm["_storage_format"] = "native_fp4"
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(nm, f, indent=2)

    print(f"  {layer_name}: {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FP8 MoE cache to native FP4")
    parser.add_argument("--fp8-cache", required=True,
                        help="Path to FP8 cache directory (input)")
    parser.add_argument("--fp4-cache", required=True,
                        help="Path to FP4 cache directory (output)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")
    args = parser.parse_args()

    layer_dirs = sorted(
        [os.path.join(args.fp8_cache, d)
         for d in os.listdir(args.fp8_cache)
         if d.startswith("model_layers_")],
        key=lambda x: int(x.split("_")[-3])
    )

    if not layer_dirs:
        print(f"ERROR: No model_layers_* dirs found in {args.fp8_cache}")
        return

    print(f"Converting {len(layer_dirs)} layers (FP8 -> native FP4)")
    print(f"  Input:   {args.fp8_cache}")
    print(f"  Output:  {args.fp4_cache}")
    print(f"  Workers: {args.workers}")
    print(f"  FP4 range: [{fp4_min}, {fp4_max}]")

    os.makedirs(args.fp4_cache, exist_ok=True)
    t_start = time.perf_counter()

    work_items = [(d, args.fp4_cache) for d in layer_dirs]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(convert_layer, work_items))

    t_total = time.perf_counter() - t_start
    n_out = len([d for d in os.listdir(args.fp4_cache)
                 if d.startswith("model_layers_")])
    print(f"\nDone: {len(layer_dirs)} layers in {t_total:.1f}s "
          f"({t_total/60:.1f} min)")
    print(f"Output: {n_out}/{len(layer_dirs)} layer dirs in {args.fp4_cache}")


if __name__ == "__main__":
    main()
