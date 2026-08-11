#!/usr/bin/env python3
"""Quick depth sweep benchmark for 35B-A3B and 27B."""

import json, time, requests

URL = "http://localhost:8180/v1/completions"
MODELS = ["qwen3.6-35b-a3b", "qwen3.6-27b"]
A_PROMPT_1K = "the quick brown fox jumps over the lazy dog " * 200  # ~1000 tokens
A_PROMPTS = {2048: "the quick brown fox jumps over the lazy dog " * 300}

def make_prompt(depth):
    base_word = "the quick brown fox jumps over the lazy dog this is a long context benchmark to measure token throughput of the vllm serving stack on amd radeon ai pro r9700 GPUs"
    needed = depth // len(base_word.split()) + 1
    return (base_word + " ") * needed

def run_tg(model_name, depth, n_tokens, reps=2):
    prompt = make_prompt(depth)
    results = []
    for r in range(reps):
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": n_tokens,
                "temperature": 0.1,
                "stream": True
            }
            start = time.time()
            tokens = 0
            with requests.post(URL, json=payload, stream=True, timeout=600) as resp:
                for line in resp.iter_lines():
                    if line and line.startswith(b"data: "):
                        data = line[6:].decode()
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            tok = chunk["choices"][0].get("text", "")
                            if tok:
                                tokens += 1
                        except:
                            pass
            elapsed = time.time() - start
            tps = tokens / elapsed if elapsed > 0 else 0
            results.append(tps)
            print(f"    run {r}: {tokens} toks, {elapsed:.1f}s, {tps:.2f} t/s")
        except Exception as e:
            print(f"    run {r} failed: {e}")
    if results:
        avg = sum(results)/len(results)
        print(f"  ➜ avg: {avg:.2f} t/s")
        return avg
    return None

def main():
    depths = [4096, 8192, 16384, 32768, 65536, 128000]
    
    for model in MODELS:
        # check model available
        try:
            m = requests.get(f"http://localhost:8180/v1/models", timeout=10).json()
            ids = [x["id"] for x in m.get("data",[])]
        except:
            ids = []
        
        if model not in ids:
            print(f"SKIP {model}: not available (got {ids})\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {model} (BF16 KV + tuned MOE, MTP {'OFF' if '35b' in model else 'ON'})")
        print(f"{'='*60}")
        
        for depth in depths:
            print(f"\nDepth ~{depth // 1000}k tokens:")
            tg32 = run_tg(model, depth, 32, reps=2)
            tg128 = run_tg(model, depth, 128, reps=2)
            print(f"  tg32: {tg32:.2f} t/s | tg128: {tg128:.2f} t/s")

if __name__ == "__main__":
    main()
