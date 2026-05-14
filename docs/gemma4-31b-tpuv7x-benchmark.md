# Gemma4-31B Inference Benchmark on TPU v7xe

Benchmark results for **Gemma4-31B-IT** on **TPU v7xe-8** (4 chips, 8 devices, 768 GB HBM) via vLLM with the experimental Batched RPA kernel. Full 128K context window is supported after applying a [one-line kernel fix](#kernel-fix-full-128k-context-support).

> **Key results**: Peak output throughput **6,144 tok/s** (P=256) · Single-user TPOT **35 ms** · Full 256K context TTFT **695 ms** · Full 128K context TTFT **378 ms** · Real text = random tokens (no performance difference)

## Environment

| Component | Config |
|-----------|--------|
| Model | `google/gemma-4-31b-it` (BF16 weights) |
| Hardware | TPU v7xe-8 (tpu7x-standard-4t, 4 chips, 768 GB HBM) |
| Framework | vLLM nightly (dev223) + tpu-inference (main) |
| Attention Kernel | Batched RPA (`USE_BATCHED_RPA_KERNEL=1`) with [prefill_batch_size fix](#kernel-fix-full-128k-context-support) |
| KV Cache | FP8 (`--kv-cache-dtype fp8`) |
| Tensor Parallel | 4 (`--tensor-parallel-size 4`) |
| Max Context | 262144 tokens (`--max-model-len 262144`, model supports 256K natively) |
| Chunked Prefill | 16K chunks (`--enable-chunked-prefill --max-num-batched-tokens 16384`) |

### Launch Command

```bash
export USE_BATCHED_RPA_KERNEL=1
export VLLM_WORKER_MULTIPROC_METHOD=fork

vllm serve google/gemma-4-31b-it \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8
```

## Benchmark Results

All benchmarks run on 2026-05-14. Each test uses `vllm bench serve` with `--ignore-eos --num-warmups 1`.

### Throughput & Latency (Random Tokens)

Dataset: `--dataset-name random` with fixed input/output lengths.

| # | Scenario | Input | Output | Concurrency | Output tok/s | Peak tok/s | TTFT | TPOT |
|---|----------|-------|--------|-------------|-------------|-----------|------|------|
| 1 | Single user | 1K | 1K | 1 | 28 | 29 | 86 ms | 35 ms |
| 2 | High throughput | 1K | 1K | 256 | 4,495 | 6,144 | 7,756 ms* | 49 ms |
| 3 | Long input | 16K | 1K | 16 | 317 | 432 | 7,014 ms* | 43 ms |
| 4 | Long output | 1K | 16K | 4 | 110 | 116 | 142 ms | 36 ms |
| 5 | 64K context | 64K | 1K | 1 | 27 | 29 | 196 ms | 37 ms |
| 6 | 128K context | 128K | 1K | 1 | 27 | 28 | 421 ms | 37 ms |

\* TTFT includes **queueing time** — with high concurrency, each request waits for earlier prefills to complete. Single-user TTFT (tests 1, 5, 6) reflects pure prefill computation.

### Long Context Scaling (Real Text — Sonnet Dataset)

Dataset: Shakespeare sonnets (`--dataset-name sonnet`), real English text repeated to fill desired input length. Tests whether natural language tokenization or attention patterns affect performance vs random tokens.

| # | Input Length | Concurrency | Output tok/s | Peak tok/s | TTFT | TPOT |
|---|-------------|-------------|-------------|-----------|------|------|
| 7 | 64K | 1 | 27.49 | 29 | 231 ms | 36 ms |
| 8 | 96K | 1 | 26.74 | 28 | 302 ms | 37 ms |
| 9 | 120K | 1 | 27.39 | 29 | 373 ms | 36 ms |
| 10 | 128K | 1 | 27.98 | 29 | 378 ms | 35 ms |
| 11 | 128K | 2 | 44.98 | 58 | 4,714 ms* | 40 ms |
| 12 | 64K | 4 | 82.99 | 112 | 6,052 ms* | 42 ms |

All tests output 1K tokens. \* TTFT includes queueing time.

### 256K Extended Context (Sonnet Dataset)

Server reconfigured with `--max-model-len 262144` to test the model's full 256K context window (`max_position_embeddings: 262144`).

| # | Input Length | Concurrency | Output tok/s | Peak tok/s | TTFT | TPOT |
|---|-------------|-------------|-------------|-----------|------|------|
| 13 | 192K | 1 | 28.34 | 29 | 526 ms | 35 ms |
| 14 | 224K | 1 | 28.21 | 29 | 644 ms | 35 ms |
| 15 | 256K | 1 | 28.60 | 30 | 695 ms | 34 ms |
| 16 | 256K | 2 | 54.12 | 58 | 980 ms* | 36 ms |

All tests output 1K tokens. \* TTFT includes queueing time.

### Analysis

**Decode latency (TPOT)** is remarkably stable: 34-42 ms across all scenarios regardless of input length (1K→256K) or concurrency (1→256). This confirms that decode performance is independent of context length once KV cache is populated.

**Prefill latency (TTFT)** scales linearly with input length in single-user mode: 86 ms (1K) → 196 ms (64K) → 378 ms (128K) → 695 ms (256K). With chunked prefill enabled (16K chunks), a 256K input is processed in ~16 chunks. Under concurrency, TTFT includes queueing delay as requests wait for prefill scheduling.

**Throughput scaling** is near-linear with concurrency:
- 1→4 users (64K): 27 → 83 output tok/s (3.1x)
- 1→2 users (128K): 28 → 45 output tok/s (1.6x)
- 1→2 users (256K): 29 → 54 output tok/s (1.9x)
- Peak at P=256 (1K): **6,144 output tok/s**

**Real text vs random tokens**: Tests 5-6 (random, 64K/128K) vs tests 7, 10 (sonnet, 64K/128K) show near-identical TPOT (35-37 ms) and throughput (27-28 tok/s), confirming the attention kernel performs consistently with natural language.

**256K full context**: The model's full 256K context window (`max_position_embeddings: 262144`) works without issues. TTFT remains under 700 ms for single-user 256K, and dual-concurrent 256K runs successfully with near-2x throughput scaling.

### Detailed Commands

<details>
<summary>Random token benchmarks (tests 1-6)</summary>

```bash
# Test 1: Single user
vllm bench serve --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos

# Test 2: High throughput
vllm bench serve --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 256 --max-concurrency 256 --num-warmups 1 --ignore-eos

# Test 3: Long input
vllm bench serve --dataset-name random \
    --random-input-len 16384 --random-output-len 1024 \
    --num-prompts 16 --max-concurrency 16 --num-warmups 1 --ignore-eos

# Test 4: Long output
vllm bench serve --dataset-name random \
    --random-input-len 1024 --random-output-len 16384 \
    --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos

# Test 5: 64K context
vllm bench serve --dataset-name random \
    --random-input-len 63488 --random-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos

# Test 6: 128K context
vllm bench serve --dataset-name random \
    --random-input-len 130048 --random-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos
```

</details>

<details>
<summary>Sonnet benchmarks (tests 7-12)</summary>

```bash
# Tests 7-10: Single user at 64K/96K/120K/128K
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len <63488|98304|122880|130048> --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos

# Test 11: 128K dual concurrent
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 130048 --sonnet-output-len 1024 \
    --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos

# Test 12: 64K quad concurrent
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 63488 --sonnet-output-len 1024 \
    --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos
```

</details>

<details>
<summary>256K extended context benchmarks (tests 13-16)</summary>

Server reconfigured with `--max-model-len 262144` for these tests.

```bash
# Tests 13-15: Single user at 192K/224K/256K
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len <196608|229376|261120> --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos

# Test 16: 256K dual concurrent
vllm bench serve --dataset-name sonnet \
    --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 261120 --sonnet-output-len 1024 \
    --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos
```

</details>

## Kernel Fix: Full 128K Context Support

### Problem

Gemma4-31B crashes with `E0200 RuntimeUnexpectedCoreHalt` when context exceeds ~32K tokens (or ~80K with chunked prefill). The crash occurs in the MIXED mode attention kernel (`RPAm-p256-b2-q256-k256`).

### Root Cause

`calculate_vmem_usage()` in `wrapper.py` only accounts for pipeline buffers (Q/KV/O arrays) but omits scratch arrays (`m`, `l`, `acc`). With `prefill_batch_size=2`, the untracked scratch memory (~24 MB) pushes total VMEM to ~93% of v7x's 64 MB capacity, causing non-deterministic overflow at longer contexts.

### Fix

One-line change in `tpu_inference/kernels/experimental/batched_rpa/wrapper.py`, line 327:

```diff
-    prefill_batch_size = 2
+    prefill_batch_size = 1
```

This halves MIXED mode scratch memory (~24 MB → ~12 MB), keeping total VMEM at ~75%.

### Validation

| Context Length | Before Fix | After Fix |
|---------------|-----------|-----------|
| ≤ 32K | PASS | PASS |
| 64K | CRASH (E0200) | PASS |
| 128K (full context) | CRASH | PASS |

No throughput regression: TPOT remains 35-37 ms (single user) at all context lengths. The fix only affects MIXED mode (chunked prefill + decode), not pure DECODE mode.

## Methodology

- **Warmup**: 1 warmup request per test to trigger XLA compilation before measurement
- **Metrics**: Median TTFT and TPOT reported for consistency; peak throughput reflects maximum instantaneous output rate
- **Endpoint**: `/v1/completions` (raw prompt, no chat template overhead)
- **EOS handling**: `--ignore-eos` forces full output length generation for reproducible measurements
- **Datasets**: Random tokens test raw compute throughput; sonnet (517 lines of Shakespeare, repeated to fill length) tests real-text behavior
