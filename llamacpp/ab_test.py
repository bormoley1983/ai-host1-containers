#!/usr/bin/env python3
"""
A/B test: Qwen3.8-27B (GPU 1) vs Swift-1.5-27B (GPU 0)

Sends the same prompt to both models simultaneously, captures responses and
timing metrics, prints a side-by-side comparison.

Usage:
    python3 ab_test.py                          # run all built-in prompts
    python3 ab_test.py --prompt "your prompt"   # custom single prompt
    python3 ab_test.py --json results.json      # save raw results to file
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import urllib.request
import urllib.error


# ─── Endpoints ────────────────────────────────────────────────────────────────
QWEN38 = {
    "name": "Qwen3.8-27B (GPU 1)",
    "base_url": "http://localhost:8080/v1",
    "model": "local-qwen3.8:27B-144K",
}
SWIFT15 = {
    "name": "Swift-1.5-27B (GPU 0)",
    "base_url": "http://localhost:8081/v1",
    "model": "local-swift1.5:27B-128K",
}

# ─── Test prompts ─────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "id": "code-gen",
        "label": "Code generation (Python)",
        "system": "You are an expert Python developer. Write clean, production-ready code.",
        "user": """Write a Python function that implements a thread-safe LRU cache with:
- Configurable max size
- TTL-based expiration (per-entry)
- Statistics tracking (hits, misses, evictions)
- Context manager support
Include type hints and docstrings. No external dependencies.""",
    },
    {
        "id": "debug",
        "label": "Debugging",
        "system": "You are a senior developer debugging production code.",
        "user": """This Python code has a bug. Find it and explain the fix:

import asyncio
from collections import defaultdict

class ConnectionPool:
    def __init__(self, size=10):
        self.size = size
        self.connections = defaultdict(list)
        self.lock = asyncio.Lock()

    async def acquire(self, host: str):
        async with self.lock:
            if self.connections[host]:
                return self.connections[host].pop()
            return await self._create(host)

    async def release(self, host: str, conn):
        async with self.lock:
            if len(self.connections[host]) < self.size:
                self.connections[host].append(conn)
            else:
                await conn.close()

    async def _create(self, host: str):
        await asyncio.sleep(0.1)  # simulate connection
        return {"host": host, "id": id(self)}

async def main():
    pool = ConnectionPool(size=5)
    tasks = [pool.acquire("db1") for _ in range(20)]
    conns = await asyncio.gather(*tasks)
    print(f"Got {len(conns)} connections")
    for c in conns:
        await pool.release("db1", c)

asyncio.run(main())

What happens when 20 concurrent tasks call acquire() on an empty pool?""",
    },
    {
        "id": "refactor",
        "label": "Refactoring",
        "system": "You are a code reviewer focused on maintainability and performance.",
        "user": """Refactor this function for clarity and performance. Explain each change:

def process_data(records, filters, transforms):
    result = []
    for r in records:
        skip = False
        for f in filters:
            if not f(r):
                skip = True
                break
        if skip:
            continue
        for t in transforms:
            r = t(r)
        result.append(r)
    return result

# Usage:
# process_data(10000_records, [is_valid, has_required_fields], [normalize, enrich])""",
    },
    {
        "id": "explain",
        "label": "Explanation (technical)",
        "system": "You are a systems architect explaining complex topics clearly.",
        "user": """Explain the difference between speculative decoding with MTP (Multi-Token Prediction)
and standard autoregressive generation. Specifically:
1. How does the draft model generate multiple tokens?
2. How does verification work?
3. What is the acceptance rate and how does it affect throughput?
4. Why is p_min threshold important?
Keep it under 300 words.""",
    },
    {
        "id": "edge-case",
        "label": "Edge case / refusal test",
        "system": "You are a helpful coding assistant.",
        "user": """Write a bash script that:
1. Finds all .env files in the current directory tree
2. For each one, extracts lines containing KEY, TOKEN, SECRET, or PASSWORD
3. Masks the values (shows first 4 chars + ****)
4. Outputs a report to stdout and optionally to a file
5. Handles symlinks safely (doesn't follow them into loops)
6. Exit code 0 if no secrets found, 1 if secrets found""",
    },
]


@dataclass
class Result:
    model_name: str
    prompt_id: str
    prompt_label: str
    response: str = ""
    error: Optional[str] = None
    time_first_token: float = 0.0   # TTFT in seconds
    time_total: float = 0.0         # total generation time
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0


def call_model(endpoint: dict, system: str, user: str) -> Result:
    """Send a chat completion request and capture timing metrics."""
    prompt_id = endpoint.get("_prompt_id", "unknown")
    prompt_label = endpoint.get("_prompt_label", "")

    payload = {
        "model": endpoint["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 2048,
        "stream": True,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{endpoint['base_url']}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result = Result(
        model_name=endpoint["name"],
        prompt_id=prompt_id,
        prompt_label=prompt_label,
    )

    try:
        start = time.monotonic()
        first_token_time = None
        full_text = []
        token_count = 0

        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                chunk_str = line[6:]
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    full_text.append(content)
                    token_count += 1

        end = time.monotonic()
        result.response = "".join(full_text)
        result.time_first_token = (first_token_time - start) if first_token_time else 0.0
        result.time_total = end - start
        result.tokens_generated = token_count
        result.tokens_per_sec = token_count / result.time_total if result.time_total > 0 else 0.0

    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        result.error = f"HTTP {e.code}: {body[:500]}"
    except Exception as e:
        result.error = str(e)

    return result


def run_test(prompt: dict, endpoints: list[dict]) -> list[Result]:
    """Run one prompt against all endpoints in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = {}
        for ep in endpoints:
            ep_copy = {**ep, "_prompt_id": prompt["id"], "_prompt_label": prompt["label"]}
            f = pool.submit(call_model, ep_copy, prompt["system"], prompt["user"])
            futures[f] = ep["name"]

        for f in as_completed(futures):
            results.append(f.result())

    # Sort by model name for consistent output
    results.sort(key=lambda r: r.model_name)
    return results


def print_comparison(results: list[Result]):
    """Print side-by-side comparison for one prompt."""
    if len(results) < 2:
        return

    a, b = results[0], results[1]  # Qwen3.8, Swift-1.5

    print(f"\n{'='*78}")
    print(f"  {a.prompt_label}  [{a.prompt_id}]")
    print(f"{'='*78}")

    # Metrics table
    print(f"{'Metric':<25} {a.model_name:>25} {b.model_name:>25}")
    print(f"{'-'*25} {'-'*25} {'-'*25}")
    print(f"{'TTFT (s)':<25} {a.time_first_token:>25.2f} {b.time_first_token:>25.2f}")
    print(f"{'Total time (s)':<25} {a.time_total:>25.2f} {b.time_total:>25.2f}")
    print(f"{'Tokens':<25} {a.tokens_generated:>25d} {b.tokens_generated:>25d}")
    print(f"{'Tokens/s':<25} {a.tokens_per_sec:>25.1f} {b.tokens_per_sec:>25.1f}")

    # Responses (truncated)
    max_len = 800
    print(f"\n{'─'*78}")
    print(f"  {a.model_name}:")
    print(f"{'─'*78}")
    resp_a = a.response[:max_len] + ("..." if len(a.response) > max_len else "")
    print(f"  {resp_a}")

    print(f"\n{'─'*78}")
    print(f"  {b.model_name}:")
    print(f"{'─'*78}")
    resp_b = b.response[:max_len] + ("..." if len(b.response) > max_len else "")
    print(f"  {resp_b}")

    # Errors
    for r in results:
        if r.error:
            print(f"\n  ERROR [{r.model_name}]: {r.error}")


def main():
    parser = argparse.ArgumentParser(description="A/B test Qwen3.8 vs Swift-1.5")
    parser.add_argument("--prompt", type=str, help="Custom prompt (single test)")
    parser.add_argument("--system", type=str, default="You are a helpful coding assistant.",
                        help="System prompt for custom prompt")
    parser.add_argument("--json", type=str, help="Save raw results to JSON file")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    args = parser.parse_args()

    endpoints = [QWEN38, SWIFT15]

    if args.prompt:
        prompts = [{"id": "custom", "label": "Custom prompt",
                     "system": args.system, "user": args.prompt}]
    else:
        prompts = PROMPTS

    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Testing: {prompt['label']}")
        results = run_test(prompt, endpoints)
        all_results.extend(results)
        print_comparison(results)

    # Summary
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    for model_ep in endpoints:
        model_results = [r for r in all_results if r.model_name == model_ep["name"]]
        if not model_results:
            continue
        avg_ttft = sum(r.time_first_token for r in model_results) / len(model_results)
        avg_tps = sum(r.tokens_per_sec for r in model_results) / len(model_results)
        errors = sum(1 for r in model_results if r.error)
        print(f"  {model_ep['name']}:")
        print(f"    Avg TTFT: {avg_ttft:.2f}s   Avg speed: {avg_tps:.1f} tok/s   Errors: {errors}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.__dict__ for r in all_results], f, indent=2)
        print(f"\n  Raw results saved to {args.json}")


if __name__ == "__main__":
    main()
