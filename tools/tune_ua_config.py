"""Sweep aiter unified-attention config knobs on this GPU (gfx1201).

Aiter's select_2d_config/select_3d_config hard-code per-arch launch knobs
(num_warps, num_stages, waves_per_eu, ...). This script monkey-patches those
selectors with knob overrides, calls the public unified_attention() API with
the Qwen3.8-27B full-attention geometry (GQA 24:4, head 256, KV page 832,
bf16 KV), and measures wall-clock per scenario so the best knobs can be baked
into a patches/aiter tuning patch.

Correctness: each candidate's output is compared against the baseline
config's output (same kernel, different launch config; mathematically
identical modulo reduction order). A config that exceeds the 64 KiB LDS
limit fails loudly at Triton's pre-launch check (ROCm/aiter#4329) and is
reported as invalid.

Run inside the vllm container with the server stopped (clean timings):
  just down
  docker compose --env-file .env --env-file env/2xr9700.vllm.common \
    --env-file env/aiter-unified-attention.env --env-file env/qwen3.6.env.common \
    --env-file env/qwen3.8-27b.env \
    run -T --rm --no-deps --entrypoint python vllm \
    /home/philip/r9700-serving/tools/tune_ua_config.py \
    --out /workspace/ua_sweep.json

`--qheads/--kvheads` override the head count, e.g. `--qheads 12 --kvheads 2`
times the per-GPU (TP=2) attention slice for a decode-step fraction estimate
(see benchmarks/2026-08-25_gfx1201_ua_tuning.md).
"""

import argparse
import json
import math
import sys
import time

import torch
import triton

import aiter.ops.triton.attention.unified_attention as ua

# Qwen3.8-27B full-attention geometry (4 of 64 layers are full attention;
# the GDN linear layers do not use this kernel). KV page size is the vLLM
# block size for this hybrid: 832 tokens on the bf16-KV default.
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_SIZE = 256
BLOCK_SIZE = 832
KV_DTYPE = torch.bfloat16
Q_DTYPE = torch.bfloat16
DEVICE = "cuda"

SOFTMAX_SCALE = HEAD_SIZE ** -0.5
NUM_QUERY_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS

# scenario name -> (num_seqs, q_len, kv_len)
SCENARIOS = {
    "d512": (1, 1, 512),      # 2D decode path (kv <= 512)
    "d1k": (1, 1, 1024),      # 3D decode path
    "d16k": (1, 1, 16384),
    "d64k": (1, 1, 65536),
    "d128k": (1, 1, 131072),
    "d64k_q4": (1, 4, 65536),     # MTP-style batch decode (4 query tokens)
    "d128k_q4": (1, 4, 131072),
    "d256k_q4": (1, 4, 262144),
    "p2k": (1, 2048, 2048),        # 2D prefill (num_2d_prgms > target)
    "p2k_c64k": (1, 2048, 65536),  # chunked-extend with long context
}


def make_scenario(num_seqs, q_len, kv_len):
    q_lens = [q_len] * num_seqs
    kv_lens = [kv_len] * num_seqs
    total_q = sum(q_lens)
    blocks_per_seq = math.ceil(kv_len / BLOCK_SIZE)
    num_blocks = num_seqs * blocks_per_seq

    g = torch.Generator(device="cpu").manual_seed(1234)
    k = (
        torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                    generator=g, dtype=torch.float32)
        .mul_(0.1)
        .to(KV_DTYPE)
        .to(DEVICE)
    )
    v = (
        torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                    generator=g, dtype=torch.float32)
        .mul_(0.1)
        .to(KV_DTYPE)
        .to(DEVICE)
    )
    q = (
        torch.randn(total_q, NUM_Q_HEADS, HEAD_SIZE, generator=g,
                    dtype=torch.float32)
        .mul_(0.1)
        .to(Q_DTYPE)
        .to(DEVICE)
    )
    out = torch.empty_like(q)

    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)),
                        dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    block_table = torch.stack([
        torch.arange(i * blocks_per_seq, (i + 1) * blocks_per_seq,
                     dtype=torch.int32)
        for i in range(num_seqs)
    ]).to(DEVICE)

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_q": cu_q,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "max_seqlen_q": max(q_lens),
        "max_seqlen_k": max(kv_lens),
        "num_q_tokens": total_q,
        "num_seqs": num_seqs,
    }


def call_ua(sc):
    return ua.unified_attention(
        q=sc["q"],
        k=sc["k"],
        v=sc["v"],
        out=sc["out"],
        cu_seqlens_q=sc["cu_q"],
        seqused_k=sc["seqused_k"],
        max_seqlen_q=sc["max_seqlen_q"],
        max_seqlen_k=sc["max_seqlen_k"],
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        window_size=(-1, -1),
        block_table=sc["block_table"],
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=None,
        output_scale=None,
        shuffled_kv_cache=False,
    )


def patch_selectors(ovr_2d, ovr_3d_attn, ovr_3d_reduce):
    orig_2d = ua.select_2d_config
    orig_3d = ua.select_3d_config

    def wrapped_2d(*a, **k):
        cfg = orig_2d(*a, **k)
        cfg.update(ovr_2d)
        if "BLOCK_M" in ovr_2d:
            cfg["BLOCK_Q"] = cfg["BLOCK_M"] // NUM_QUERY_PER_KV
        return cfg

    def wrapped_3d(*a, **k):
        attn, red = orig_3d(*a, **k)
        attn.update(ovr_3d_attn)
        red.update(ovr_3d_reduce)
        return attn, red

    ua.select_2d_config = wrapped_2d
    ua.select_3d_config = wrapped_3d


def unpatch_selectors(orig_2d, orig_3d):
    ua.select_2d_config = orig_2d
    ua.select_3d_config = orig_3d


def run_once(sc):
    sc["out"].zero_()
    call_ua(sc)
    torch.cuda.synchronize()
    return sc["out"].clone()


def time_it(sc, iters, warmup):
    for _ in range(warmup):
        call_ua(sc)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call_ua(sc)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def is_lds_error(ex):
    s = str(ex)
    t = type(ex).__name__
    return (
        "out of resource" in s
        or "OutOfResources" in t
        or "shared memory" in s
    )


def main():
    global NUM_Q_HEADS, NUM_KV_HEADS, NUM_QUERY_PER_KV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/ua_sweep.json")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-3d", action="store_true")
    ap.add_argument("--skip-2d", action="store_true")
    ap.add_argument("--3d-warps", default="1,2,4",
                    help="comma list of attn num_warps to sweep (3D)")
    ap.add_argument("--3d-waves", default="6,8",
                    help="comma list of waves_per_eu to sweep (3D)")
    ap.add_argument("--qheads", type=int, default=None,
                    help="override num query heads (e.g. 12 for per-GPU TP=2)")
    ap.add_argument("--kvheads", type=int, default=None,
                    help="override num kv heads")
    args = ap.parse_args()

    if args.qheads is not None:
        NUM_Q_HEADS = args.qheads
    if args.kvheads is not None:
        NUM_KV_HEADS = args.kvheads
    NUM_QUERY_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS

    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    if free_gb < 3.0:
        print(f"warning: only {free_gb:.1f} GiB free VRAM; stop the server "
              f"for clean runs", file=sys.stderr)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        sys.exit(f"unknown scenarios: {unknown}; have {list(SCENARIOS)}")

    orig_2d = ua.select_2d_config
    orig_3d = ua.select_3d_config

    # Candidate overrides.
    ovr_2d_candidates = []
    if not args.skip_2d:
        for warps in (2, 4, 8):
            for stages in (1, 2):
                ovr_2d_candidates.append(
                    (f"2d:w{warps}:s{stages}",
                     {"num_warps": warps, "num_stages": stages}, {}, {})
                )
    ovr_3d_candidates = []
    if not args.skip_3d:
        three_d_warps = getattr(args, "3d_warps")
        three_d_waves = getattr(args, "3d_waves")
        for warps in (int(x) for x in three_d_warps.split(",")):
            for stages in (1, 2):
                for waves in (int(x) for x in three_d_waves.split(",")):
                    ovr_3d_candidates.append(
                        (f"3d:w{warps}:s{stages}:eu{waves}",
                         {}, {"num_warps": warps, "num_stages": stages,
                              "waves_per_eu": waves}, {})
                    )

    results = {"geometry": {
        "num_q_heads": NUM_Q_HEADS, "num_kv_heads": NUM_KV_HEADS,
        "head_size": HEAD_SIZE, "block_size": BLOCK_SIZE,
        "kv_dtype": str(KV_DTYPE), "arch": ua.DEVICE_ARCH,
    }, "scenarios": {}}

    for name in scenarios:
        num_seqs, q_len, kv_len = SCENARIOS[name]
        print(f"\n=== scenario {name}: seqs={num_seqs} q={q_len} kv={kv_len}",
              flush=True)
        sc = make_scenario(num_seqs, q_len, kv_len)

        # Baseline: stock selectors, unpatched.
        base_out = None
        base_ms = None
        base_cfg = None
        try:
            base_out = run_once(sc)
            base_ms = time_it(sc, args.iters, args.warmup)
        except Exception as ex:  # noqa: BLE001
            print(f"  baseline FAILED: {ex}", flush=True)
            del sc
            torch.cuda.empty_cache()
            results["scenarios"][name] = {"baseline_failed": str(ex)}
            continue
        print(f"  baseline: {base_ms:.3f} ms", flush=True)

        # Capture the stock config(s) for this scenario's dispatch path
        # (mirrors the dispatch math in unified_attention()).
        block_m = (16 if NUM_QUERY_PER_KV <= 16
                   else triton.next_power_of_2(NUM_QUERY_PER_KV))
        block_q = block_m // NUM_QUERY_PER_KV
        if sc["max_seqlen_q"] == 1:
            num_2d_prgms = num_seqs * NUM_KV_HEADS
        else:
            num_2d_prgms = (sc["num_q_tokens"] // block_q
                            + num_seqs) * NUM_KV_HEADS
        target_num_prgms = ua.get_num_sms() * 4
        use2d = ua.use_2d_kernel(
            HEAD_SIZE, 0, sc["max_seqlen_q"] == 1, sc["max_seqlen_q"],
            sc["max_seqlen_k"], target_num_prgms, num_2d_prgms,
        )
        if use2d:
            base_cfg = dict(orig_2d(
                BLOCK_SIZE, HEAD_SIZE, 0, sc["max_seqlen_q"] == 1,
                sc["max_seqlen_q"], sc["max_seqlen_k"],
                NUM_Q_HEADS // NUM_KV_HEADS, num_2d_prgms,
                Q_DTYPE, KV_DTYPE, False,
            ))
        else:
            attn, red = orig_3d(
                HEAD_SIZE, BLOCK_SIZE, sc["max_seqlen_k"], target_num_prgms,
                num_2d_prgms, Q_DTYPE, KV_DTYPE, False, 1, 0,
            )
            base_cfg = {"attn": dict(attn), "reduce": dict(red)}
        print(f"  dispatch: {'2D' if use2d else '3D'}, base config: {base_cfg}",
              flush=True)

        rows = [{"label": "baseline", "ms": base_ms, "valid": True,
                 "max_diff": 0.0,
                 "tokens_per_s": (sc["num_q_tokens"] / base_ms * 1000)}]

        candidates = []
        if use2d:
            if not args.skip_2d:
                candidates = ovr_2d_candidates
        else:
            if not args.skip_3d:
                candidates = ovr_3d_candidates

        for label, o2, o3a, o3r in candidates:
            patch_selectors(o2, o3a, o3r)
            try:
                out = run_once(sc)
                diff = (out.float() - base_out.float()).abs().max().item()
                ms = time_it(sc, args.iters, args.warmup)
                rows.append({"label": label, "ms": ms, "valid": True,
                             "max_diff": diff,
                             "tokens_per_s": sc["num_q_tokens"] / ms * 1000})
                print(f"  {label}: {ms:.3f} ms (diff {diff:.4f})", flush=True)
            except Exception as ex:  # noqa: BLE001
                if is_lds_error(ex):
                    print(f"  {label}: INVALID (LDS overflow)", flush=True)
                    rows.append({"label": label, "ms": None, "valid": False,
                                 "error": "lds-overflow"})
                else:
                    print(f"  {label}: ERROR {ex}", flush=True)
                    rows.append({"label": label, "ms": None, "valid": False,
                                 "error": str(ex)[:200]})
            finally:
                unpatch_selectors(orig_2d, orig_3d)

        results["scenarios"][name] = {
            "shape": {"num_seqs": num_seqs, "q_len": q_len, "kv_len": kv_len},
            "dispatch": "2d" if use2d else "3d",
            "base_config": base_cfg,
            "rows": rows,
        }
        del sc, base_out
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown summary.
    print("\n\n# unified-attention config sweep\n")
    for name, res in results["scenarios"].items():
        if "rows" not in res:
            continue
        base_ms = res["rows"][0]["ms"]
        print(f"## {name} ({res['dispatch']}), base "
              f"{base_ms:.3f} ms\n")
        print("| label | ms | t/s | vs base | max_diff |")
        print("|:------|---:|----:|--------:|---------:|")
        for r in res["rows"]:
            if r["ms"] is None:
                print(f"| {r['label']} | — | — | invalid ({r.get('error')}) | — |")
            else:
                speedup = base_ms / r["ms"]
                print(f"| {r['label']} | {r['ms']:.3f} | "
                      f"{r['tokens_per_s']:.1f} | {speedup:.3f}x | "
                      f"{r['max_diff']:.4f} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
