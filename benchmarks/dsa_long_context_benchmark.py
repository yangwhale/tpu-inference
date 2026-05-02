#!/usr/bin/env python3
"""DSA Long-Context Benchmark: Dense vs DSA decode performance at different context lengths.

Usage:
    python3 dsa_long_context_benchmark.py --server http://localhost:8000 --context-lengths 4096,16384,32768,65536

Measures TTFT (prefill time) and decode tok/s at each context length.
Outputs results as JSON for easy comparison.
"""

import argparse
import json
import time
import requests
import sys
from typing import Optional


def generate_padding_text(target_tokens: int) -> str:
    """Generate padding text of approximately target_tokens length.

    English text: ~1.3 tokens per word, ~6.5 chars per word.
    So ~5 chars per token. We use 4.5 to be conservative (slightly over-generate).
    """
    base = (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "Scientists discovered a new species of butterfly in the Amazon rainforest last summer. "
        "The ancient library contained manuscripts dating back to the Roman Empire. "
        "Modern architecture blends sustainable materials with innovative design principles. "
        "The orchestra performed a breathtaking rendition of Beethoven's Ninth Symphony. "
    )
    chars_needed = int(target_tokens * 4.5)
    repetitions = (chars_needed // len(base)) + 1
    text = (base * repetitions)[:chars_needed]
    return text


def run_single_test(
    server_url: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    model: str = "default",
) -> dict:
    """Send a single completion request and measure timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    first_token_time = None
    start_time = time.time()
    token_count = 0
    full_text = ""

    try:
        resp = requests.post(
            f"{server_url}/v1/completions",
            json=payload,
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        if text:
                            if first_token_time is None:
                                first_token_time = time.time()
                            token_count += 1
                            full_text += text
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

    end_time = time.time()

    if first_token_time is None:
        return {"error": "No tokens received"}

    ttft = first_token_time - start_time
    decode_time = end_time - first_token_time
    total_time = end_time - start_time
    decode_tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0

    return {
        "ttft_s": round(ttft, 3),
        "decode_time_s": round(decode_time, 3),
        "total_time_s": round(total_time, 3),
        "output_tokens": token_count,
        "decode_tok_per_s": round(decode_tps, 2),
        "output_preview": full_text[:100],
    }


def count_tokens_approx(text: str, server_url: str, model: str = "default") -> Optional[int]:
    """Use the server's tokenize endpoint if available, else estimate."""
    try:
        resp = requests.post(
            f"{server_url}/tokenize",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            return data.get("count") or len(data.get("tokens", []))
    except Exception:
        pass
    return None


def warmup(server_url: str, model: str = "default"):
    """Send a short request to warm up XLA compilation."""
    print("  Warming up (short request to trigger XLA compilation)...")
    payload = {
        "model": model,
        "prompt": "Hello, how are you?",
        "max_tokens": 10,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            f"{server_url}/v1/completions", json=payload, timeout=300
        )
        resp.raise_for_status()
        print("  Warmup done.")
    except Exception as e:
        print(f"  Warmup failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="DSA Long-Context Benchmark")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument(
        "--context-lengths",
        default="4096,16384,32768",
        help="Comma-separated list of context lengths to test",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=2, help="Runs per context length (first is warmup)")
    parser.add_argument("--mode", default="unknown", help="Label: 'dense' or 'dsa'")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if not set)")
    args = parser.parse_args()

    if args.model is None:
        try:
            resp = requests.get(f"{args.server}/v1/models", timeout=10)
            if resp.ok:
                models = resp.json().get("data", [])
                if models:
                    args.model = models[0]["id"]
        except Exception:
            pass
        if args.model is None:
            args.model = "default"
        print(f"Auto-detected model: {args.model}")

    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]

    print(f"=== DSA Long-Context Benchmark ===")
    print(f"Mode: {args.mode}")
    print(f"Server: {args.server}")
    print(f"Context lengths: {context_lengths}")
    print(f"Max output tokens: {args.max_tokens}")
    print(f"Runs per length: {args.runs}")
    print()

    if not args.skip_warmup:
        warmup(args.server, model=args.model)
        print()

    question = (
        "\n\nBased on the text above, write a brief summary of the key themes "
        "discussed. Be concise and focus on the main ideas."
    )

    all_results = []

    for ctx_len in context_lengths:
        print(f"--- Context length: {ctx_len} ---")

        # Reserve tokens for question + output
        padding_tokens = ctx_len - 50  # ~50 tokens for question
        padding = generate_padding_text(padding_tokens)
        prompt = padding + question

        # Try to get exact token count
        actual_tokens = count_tokens_approx(prompt, args.server, model=args.model)
        if actual_tokens:
            print(f"  Actual prompt tokens: {actual_tokens}")
        else:
            print(f"  Estimated prompt tokens: ~{padding_tokens}")

        for run_idx in range(args.runs):
            label = "warmup" if run_idx == 0 else f"run_{run_idx}"
            print(f"  [{label}] Running...", end=" ", flush=True)
            result = run_single_test(
                args.server, prompt, max_tokens=args.max_tokens,
                model=args.model,
            )

            if "error" in result:
                print(f"ERROR: {result['error']}")
                result_entry = {
                    "context_length": ctx_len,
                    "actual_tokens": actual_tokens,
                    "mode": args.mode,
                    "run": label,
                    "error": result["error"],
                }
            else:
                print(
                    f"TTFT={result['ttft_s']:.1f}s  "
                    f"decode={result['decode_tok_per_s']:.1f} tok/s  "
                    f"({result['output_tokens']} tokens in {result['decode_time_s']:.1f}s)"
                )
                result_entry = {
                    "context_length": ctx_len,
                    "actual_tokens": actual_tokens,
                    "mode": args.mode,
                    "run": label,
                    **result,
                }

            all_results.append(result_entry)

        print()

    # Print summary table
    print("=== Summary ===")
    print(f"{'Context':>10} {'Run':>8} {'TTFT(s)':>8} {'Decode tok/s':>14} {'Tokens':>8}")
    print("-" * 55)
    for r in all_results:
        if "error" in r:
            print(f"{r['context_length']:>10} {r['run']:>8} {'ERROR':>8}")
        else:
            print(
                f"{r['context_length']:>10} {r['run']:>8} "
                f"{r['ttft_s']:>8.1f} {r['decode_tok_per_s']:>14.1f} "
                f"{r['output_tokens']:>8}"
            )

    # Save results
    output_path = args.output or f"/tmp/dsa_benchmark_{args.mode}_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "server": args.server,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    main()
