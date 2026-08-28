#!/usr/bin/env python3
"""Prefix-cache hit-rate probe for hybrid GDN models.

Sends a multi-turn thinking-off conversation and reports per-turn prefix-cache
hits from the server's /metrics endpoint. On hybrid GDN + mamba-cache-mode
align, hits stay at 0% until the align-mode checkpoint bug (vllm-project/vllm
#45238) is fixed upstream. A non-zero hit rate on this probe is the signal the
fix landed.

Usage:
    python3 benchmarks/prefix_cache_probe.py [model] [turns]

Requires a reachable vLLM server (default http://localhost:8000, override with
VLLM_BASE_URL). Runs from the host or inside the container.
"""
import os
import sys
import time

import requests

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b"
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SECRET = "the answer to everything is the number 42"


def _metric(name: str) -> float | None:
    text = requests.get(BASE + "/metrics", timeout=30).text
    for line in text.splitlines():
        if line.startswith(name + "{"):
            return float(line.rsplit("} ", 1)[-1])
    return None


def _cache_geometry() -> None:
    text = requests.get(BASE + "/metrics", timeout=30).text
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            for part in line.split("}")[0].split(","):
                if part.startswith(("block_size=", "mamba_cache_mode=")):
                    print(f"  {part}", flush=True)
            break


def turn(history: list[dict], tail: str, max_tokens: int = 64):
    body = {
        "model": MODEL,
        "messages": history + [{"role": "user", "content": tail}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = requests.post(BASE + "/v1/chat/completions", json=body, timeout=300)
    dt = time.time() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return (data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"],
            dt, data["choices"][0]["message"]["content"])


def main() -> None:
    q0, h0 = _metric("vllm:prefix_cache_queries_total"), _metric(
        "vllm:prefix_cache_hits_total")
    if q0 is None or h0 is None:
        raise SystemExit(f"metrics not found on {BASE}; is the server up?")
    print(f"model={MODEL} baseline: queries={q0:.0f} hits={h0:.0f}", flush=True)
    print("cache geometry:", flush=True)
    _cache_geometry()

    history = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": f"Remember this secret: {SECRET}."},
    ]
    for i in range(1, TURNS + 1):
        tail = (f"Turn {i}: still remember the secret? Tell me what it is, "
                f"plus one extra fact about the number.")
        _, _, dt, out = turn(history, tail)
        history += [
            {"role": "assistant", "content": out},
            {"role": "user", "content": tail},
        ]
        q1, h1 = _metric("vllm:prefix_cache_queries_total"), _metric(
            "vllm:prefix_cache_hits_total")
        dq, dh = (q1 or q0) - q0, (h1 or h0) - h0
        pct = (100.0 * dh / dq) if dq > 0 else 0.0
        print(f"  turn {i}: {dt:6.1f}s  queries={dq:7.0f}  hits={dh:7.0f}  "
              f"hit%={pct:5.1f}", flush=True)
        q0, h0 = q1, h1

    pt, ct, dt, out = turn(history[:-1], "What is the secret? Answer in one word.")
    print(f"  coherence: prompt={pt} gen={ct} time={dt:.1f}s -> {out[:60]!r}",
          flush=True)
    if SECRET.split()[-1] not in out:
        raise SystemExit(f"FAIL: model lost the secret: {out[:120]!r}")
    print("PASS")


if __name__ == "__main__":
    main()