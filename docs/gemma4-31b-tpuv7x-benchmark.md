# Gemma4-31B Inference Benchmark on TPU v7xe

Benchmark results for **Gemma4-31B-IT** running on **TPU v7xe (tpu7x-standard-4t, 4 chips / 8 devices)** via vLLM with the experimental Batched RPA kernel.

## Environment

| Component | Version / Config |
|-----------|-----------------|
| Model | `google/gemma-4-31b-it` (BF16) |
| Hardware | TPU v7xe-8 (4 chips, 768 GB HBM total) |
| Framework | vLLM nightly (dev223) + tpu-inference (main) |
| Kernel | Batched RPA (`USE_BATCHED_RPA_KERNEL=1`) |
| KV Cache | FP8 (`--kv-cache-dtype fp8`) |
| Tensor Parallel | 4 (`--tensor-parallel-size 4`) |
| Max Context | 131072 (`--max-model-len 131072`) |
| Chunked Prefill | Enabled (`--enable-chunked-prefill --max-num-batched-tokens 16384`) |
| GPU Memory Util | 0.95 (`--gpu-memory-utilization 0.95`) |

### Launch Command

```bash
export USE_BATCHED_RPA_KERNEL=1
export VLLM_WORKER_MULTIPROC_METHOD=fork

vllm serve google/gemma-4-31b-it \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8
```

## Benchmark Results (Round 5 — 2026-05-14)

All benchmarks use `vllm bench serve` with `--dataset-name random --ignore-eos --num-warmups 1`.

### Summary Table

| Test | Input | Output | Concurrency | Peak tok/s | Median TTFT | Median TPOT | Status |
|------|-------|--------|-------------|-----------|-------------|-------------|--------|
| B1: Single User | 1K | 1K | 1 | 29 | 86 ms | 35 ms | PASS |
| B3: High Throughput | 1K | 1K | 256 | 6,144 | 7,756 ms | 49 ms | PASS |
| B4: Long Input | 16K | 1K | 16 | 432 | 7,014 ms | 43 ms | PASS |
| B5: Long Output | 1K | 16K | 4 | 116 | 142 ms | 36 ms | PASS |
| B6: 64K Context | 64K | 1K | 1 | 29 | 196 ms | 37 ms | PASS |
| B7: 128K Context | 128K | 1K | 1 | 28 | 421 ms | 37 ms | PASS |

### Detailed Results

#### B1 — Single User Latency (1K/1K)

```
vllm bench serve --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 28.15 |
| Peak tok/s | 29.00 |
| Median TTFT | 86 ms |
| Median TPOT | 35 ms |

#### B3 — High Throughput (1K/1K, P=256)

```
vllm bench serve --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 256 --max-concurrency 256 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 4,495 |
| Peak tok/s | 6,144 |
| Total tok/s | 8,990 |
| Median TTFT | 7,756 ms |
| Median TPOT | 49 ms |

#### B4 — Long Input (16K/1K, P=16)

```
vllm bench serve --dataset-name random \
    --random-input-len 16384 --random-output-len 1024 \
    --num-prompts 16 --max-concurrency 16 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 317 |
| Peak tok/s | 432 |
| Total tok/s | 5,397 |
| Median TTFT | 7,014 ms |
| Median TPOT | 43 ms |

#### B5 — Long Output (1K/16K, P=4)

```
vllm bench serve --dataset-name random \
    --random-input-len 1024 --random-output-len 16384 \
    --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 110 |
| Peak tok/s | 116 |
| Median TTFT | 142 ms |
| Median TPOT | 36 ms |

#### B6 — 64K Context (64K/1K, Single User)

Previously crashed with E0200 (before the kernel fix). Now stable.

```
vllm bench serve --dataset-name random \
    --random-input-len 63488 --random-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 27 |
| Peak tok/s | 29 |
| Median TTFT | 196 ms |
| Median TPOT | 37 ms |

#### B7 — 128K Context (128K/1K, Single User)

Full 128K context window. Previously impossible (crash at >32K). Enabled by the kernel fix below.

```
vllm bench serve --dataset-name random \
    --random-input-len 130048 --random-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 27 |
| Peak tok/s | 28 |
| Median TTFT | 421 ms |
| Median TPOT | 37 ms |

## Long Text Benchmark (Sonnet Dataset — 2026-05-14)

Real English text benchmark using Shakespeare sonnets (`--dataset-name sonnet`). Unlike random tokens, sonnet tests realistic tokenization patterns and attention behavior with natural language at extreme context lengths.

### Summary Table

| Test | Input | Concurrency | Output tok/s | Peak tok/s | Median TTFT | Median TPOT | Status |
|------|-------|-------------|-------------|-----------|-------------|-------------|--------|
| L1: 64K Single | 64K | 1 | 27.49 | 29 | 231 ms | 36 ms | PASS |
| L2: 96K Single | 96K | 1 | 26.74 | 28 | 302 ms | 37 ms | PASS |
| L3: 120K Single | 120K | 1 | 27.39 | 29 | 373 ms | 36 ms | PASS |
| L4: 128K Single | 128K | 1 | 27.98 | 29 | 378 ms | 35 ms | PASS |
| L5: 128K Dual | 128K | 2 | 44.98 | 58 | 4,714 ms | 40 ms | PASS |
| L6: 64K Quad | 64K | 4 | 82.99 | 112 | 6,052 ms | 42 ms | PASS |

### Key Findings

- **TPOT stability**: 35-42 ms across all input lengths (64K-128K), confirming decode performance is independent of context length
- **TTFT scales linearly**: 231 ms (64K) → 378 ms (128K), consistent with chunked prefill processing (~8 chunks for 128K)
- **Concurrent 128K works**: Two simultaneous 128K requests (L5) complete successfully with near-linear throughput scaling (28→45 tok/s)
- **Real text ≈ random tokens**: No significant performance difference vs random token benchmarks (Round 5 B6/B7), indicating stable attention kernel behavior with natural language

### Detailed Results

#### L1 — 64K Single User (Sonnet)

```
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 63488 --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 27.49 |
| Peak tok/s | 29 |
| Median TTFT | 231 ms |
| Median TPOT | 36 ms |

#### L2 — 96K Single User (Sonnet)

```
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 98304 --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 26.74 |
| Peak tok/s | 28 |
| Median TTFT | 302 ms |
| Median TPOT | 37 ms |

#### L3 — 120K Single User (Sonnet)

```
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 122880 --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 27.39 |
| Peak tok/s | 29 |
| Median TTFT | 373 ms |
| Median TPOT | 36 ms |

#### L4 — 128K Single User (Sonnet)

Full context window with real text.

```
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 130048 --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 27.98 |
| Peak tok/s | 29 |
| Median TTFT | 378 ms |
| Median TPOT | 35 ms |

#### L5 — 128K Dual Concurrent (Sonnet)

Two simultaneous 128K requests — tests memory pressure under concurrent full-context workloads.

```
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 130048 --sonnet-output-len 1024 \
    --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 44.98 |
| Peak tok/s | 58 |
| Median TTFT | 4,714 ms |
| Median TPOT | 40 ms |

#### L6 — 64K Quad Concurrent (Sonnet)

Four simultaneous 64K requests — tests throughput scaling at moderate context length.

```
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 63488 --sonnet-output-len 1024 \
    --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos
```

| Metric | Value |
|--------|-------|
| Output tok/s | 82.99 |
| Peak tok/s | 112 |
| Median TTFT | 6,052 ms |
| Median TPOT | 42 ms |

## Kernel Fix: Full 128K Context Support

### Problem

With the default Batched RPA kernel, Gemma4-31B crashes with `E0200 RuntimeUnexpectedCoreHalt` when context exceeds ~32K tokens (or ~80K with chunked prefill enabled). The crash occurs in the MIXED mode attention kernel (`RPAm-p256-b2-q256-k256`).

### Root Cause

`calculate_vmem_usage()` in `wrapper.py` only accounts for pipeline buffers (Q/KV/O arrays) but omits scratch arrays (`m`, `l`, `acc` from `lm_scratch_shape` and `acc_scratch_shape`). With `prefill_batch_size=2`, the untracked scratch memory (~24 MB) pushes total VMEM usage to ~93% of the 64 MB v7x VMEM capacity, causing non-deterministic overflow crashes.

### Fix

One-line change in `tpu_inference/kernels/experimental/batched_rpa/wrapper.py`, line 327:

```diff
-    prefill_batch_size = 2
+    prefill_batch_size = 1
```

This halves MIXED mode scratch memory usage (~24 MB → ~12 MB), keeping total VMEM at ~75% and eliminating overflow.

### Validation

| Context Length | Before Fix (batch=2) | After Fix (batch=1) |
|---------------|---------------------|---------------------|
| ≤ 32K | PASS | PASS |
| 64K | CRASH (E0200) | PASS |
| 95K | CRASH | PASS |
| 119K | CRASH | PASS |
| 125K | CRASH | PASS |
| 128K (full) | CRASH | PASS |
| 131K (max_model_len) | CRASH | PASS |

### Performance Impact

No throughput regression observed. TPOT remains stable at 35-37 ms (single user) across all context lengths. The fix only affects MIXED mode (chunked prefill + decode), not pure DECODE mode which handles the majority of token generation.

## Methodology

- **Warmup**: Each benchmark runs 1 warmup request before the main run to trigger XLA compilation
- **Metrics**: `vllm bench serve` reports both average and peak throughput; summary table uses median TTFT/TPOT for consistency
- **Dataset (Round 5)**: Random tokens (`--dataset-name random`) with fixed input/output lengths
- **Dataset (Long Text)**: Shakespeare sonnets (`--dataset-name sonnet`, `/workspace/vllm/benchmarks/sonnet.txt`, 517 lines) repeated to fill desired input length via `--sonnet-input-len`
- **Endpoint**: `/v1/completions` (raw prompt, no chat template overhead)
- **EOS handling**: `--ignore-eos` forces full output length generation for consistent measurements
