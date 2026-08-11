#!/usr/bin/env python3
"""Run depth sweep benchmarks via llama-benchy and generate tables."""

import json, subprocess, time

PROFILE = "qwen3.6-35b-a3b"
DEPTHS = [4096, 8192, 16384, 32768, 65536, 128000]

def run_bench(depth):
    cmd = [
        "just", "exec", "uvx", "llama-benchy",
        "--base-url", "http://localhost:8180/v1",
        "--served-model-name", PROFILE,
        "--tg", "32",
        "--depth", str(depth),
        "--runs", "2",
        "--save-result", "/tmp/bench.json",
        "--format", "json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # extract from stdout
    print(f"stdout: {result.stdout[-200:] if result.stdout else ''}")
    print(f"stderr: {result.stderr[-200:] if result.stderr else ''}")
    return result.returncode == 0

if __name__ == "__main__":
    # start server
    print("Starting server...")
    subprocess.run(["just", "down"], capture_output=True)
    time.sleep(2)
    result = subprocess.run(
        ["MODEL_PROFILE="+PROFILE, "just", "up"],
        capture_output=True, text=True
    )
    print(f"Server started: {'OK' if result.returncode == 0 else 'FAIL'}")
    
    results = {}
    for depth in DEPTHS:
        print(f"\n=== depth={depth} ===")
        ok = run_bench(depth)
        if ok:
            with open('/tmp/bench.json') as f:
                data = json.load(f)
            for b in data.get('benchmarks', []):
                tg = b.get('tg_throughput', {}).get('mean', 0)
                pp = b.get('pp_throughput', {}).get('mean', 0)
                ttft = b.get('ttfr', {}).get('mean', 0) / 1000 if 'ttfr' in b else 0
                results[depth] = {'tg32': tg, 'pp': pp, 'ttft': ttft}
                print(f"  ✓ tg32={tg:.2f} t/s, pp={pp:.0f} t/s, ttft={ttft:.3f}s")
        else:
            print(f"  ✗ benchmark failed")
    
    print("\n=== Results ===")
    for d in DEPTHS:
        r = results.get(d, {})
        print(f"d{d}: tg32={r.get('tg32', 'N/A'):.2f}, pp={r.get('pp', 'N/A'):.0f}")
