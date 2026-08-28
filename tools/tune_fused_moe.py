import argparse
import itertools
import json
import time

import torch
import triton

import vllm.model_executor.layers.fused_moe.fused_moe as fm
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

# vLLM keys the fused MoE config file on the per-GPU geometry:
# E = experts (vLLM replicates experts across TP ranks, so E stays global),
# N = per-partition intermediate (global moe_intermediate_size / TP),
# K = hidden_size. Defaults below are Qwen3.6-35B-A3B at TP=2 (global 512 / 2);
# `just tune` / run_tune.py pass the active profile's TP-derived geometry.
E = 256
N = 256
K = 2048
TOP_K = 8
BLOCK_SHAPE = [128, 128]

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 0,
}


def make_tensors(M: int, device="cuda"):
    torch.manual_seed(0)
    hidden = (torch.randn(M, K, device=device) * 0.01).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * N, K, device=device) * 0.01).to(torch.float8_e4m3fn)
    w2 = (torch.randn(E, K, N, device=device) * 0.01).to(torch.float8_e4m3fn)
    w1_scale = torch.rand(E, 2 * N // BLOCK_SHAPE[0], K // BLOCK_SHAPE[1], device=device)
    w2_scale = torch.rand(E, K // BLOCK_SHAPE[0], N // BLOCK_SHAPE[1], device=device)
    topk_weights = torch.rand(M, TOP_K, device=device)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device).to(torch.int32)
    q = fp8_w8a8_moe_quant_config(w1_scale, w2_scale, block_shape=BLOCK_SHAPE)
    return hidden, w1, w2, topk_weights, topk_ids, q


def bench_one(M, cfg, tensors, reps=50, warmup=5):
    hidden, w1, w2, topk_weights, topk_ids, q = tensors
    try:
        with override_config(cfg):
            for _ in range(warmup):
                fm.fused_experts(hidden, w1, w2, topk_weights, topk_ids,
                                 global_num_experts=E, quant_config=q)
        torch.cuda.synchronize()
        times = []
        with override_config(cfg):
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fm.fused_experts(hidden, w1, w2, topk_weights, topk_ids,
                                 global_num_experts=E, quant_config=q)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        med = sorted(times)[len(times) // 2]
        return med
    except Exception:
        return None


def variants(base: dict):
    """Generate a focused candidate set around a base config."""
    bm = {16, 32, 64}
    bn = {64, 128, 256}
    bk = {64, 128, 256}
    gm = {1, 4, 8, 16}
    nw = {4, 8}
    ns = {2, 3}
    wpe = {0, 1, 2, 4}
    # local moves from base
    cands = [base]
    for k, pool in [("BLOCK_SIZE_M", bm), ("BLOCK_SIZE_N", bn),
                    ("BLOCK_SIZE_K", bk), ("GROUP_SIZE_M", gm),
                    ("num_warps", nw), ("num_stages", ns),
                    ("waves_per_eu", wpe)]:
        for v in pool:
            c = dict(base)
            c[k] = v
            cands.append(c)
    # cross of the three block dims at base warps/stages, plus a global sweep
    for m, n, k in itertools.product(bm, bn, bk):
        cands.append({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k,
                      "GROUP_SIZE_M": base["GROUP_SIZE_M"],
                      "num_warps": base["num_warps"],
                      "num_stages": base["num_stages"],
                      "waves_per_eu": base["waves_per_eu"]})
    # dedupe preserving order
    seen, out = set(), []
    for c in cands:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main():
    global E, N, K, TOP_K
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=str,
                    default="1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--E", type=int, default=256,
                    help="experts (global; replicated across TP ranks)")
    ap.add_argument("--N", type=int, default=256,
                    help="per-partition intermediate (global / TP)")
    ap.add_argument("--K", type=int, default=2048, help="hidden size")
    ap.add_argument("--topk", type=int, default=8,
                    help="experts routed per token")
    ap.add_argument("--out", type=str, default=None,
                    help="config file to write (default: derived from E/N)")
    ap.add_argument("--seed", type=str, default=None,
                    help="existing config to seed from (default: --out)")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    E, N, K, TOP_K = args.E, args.N, args.K, args.topk
    if args.out is None:
        args.out = (f"/app/fused_moe_configs/E={E},N={N},"
                    f"device_name=AMD_Radeon_R9700,dtype=fp8_w8a8,"
                    f"block_shape={[128,128]}.json")
    if args.seed is None:
        args.seed = args.out

    print(f"triton {triton.__version__}", flush=True)
    Ms = [int(x) for x in args.M.split(",")]

    seed = {}
    try:
        with open(args.seed) as f:
            seed = json.load(f)
    except FileNotFoundError:
        pass
    seed.pop("triton_version", None)
    seed_cfgs = {int(k): v for k, v in seed.items()}

    results = {}
    for M in Ms:
        base = seed_cfgs.get(M)
        if base is None:
            if seed_cfgs:
                base = seed_cfgs[min(seed_cfgs, key=lambda k: abs(k - M))]
            else:
                base = dict(DEFAULT_CONFIG)
        configs = [base] if args.no_sweep else variants(base)
        # Allocate the benchmark tensors once per M and reuse across configs;
        # regenerating them per candidate wasted ~0.5 GB allocs and added noise.
        tensors = make_tensors(M)
        print(f"M={M}: sweeping {len(configs)} configs from seed "
              f"{base.get('BLOCK_SIZE_M')}/{base.get('BLOCK_SIZE_N')}/"
              f"{base.get('BLOCK_SIZE_K')}", flush=True)
        scored = []
        for cfg in configs:
            t = bench_one(M, cfg, tensors, reps=args.reps)
            if t is not None:
                scored.append((t, cfg))
            print(f"  {cfg['BLOCK_SIZE_M']}/{cfg['BLOCK_SIZE_N']}/"
                  f"{cfg['BLOCK_SIZE_K']} g{cfg['GROUP_SIZE_M']} "
                  f"w{cfg['num_warps']} s{cfg['num_stages']} "
                  f"wpe{cfg['waves_per_eu']}: "
                  f"{(t*1e6 if t else float('nan')):>8.0f} us", flush=True)
        scored.sort()
        best = scored[0][1]
        bt = scored[0][0]
        results[M] = best
        print(f"  -> best {best['BLOCK_SIZE_M']}/{best['BLOCK_SIZE_N']}/"
              f"{best['BLOCK_SIZE_K']} g{best['GROUP_SIZE_M']} "
              f"w{best['num_warps']} s{best['num_stages']} "
              f"wpe{best['waves_per_eu']}: {bt*1e6:.0f} us", flush=True)

    out = {"triton_version": triton.__version__}
    out.update({str(M): cfg for M, cfg in results.items()})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
