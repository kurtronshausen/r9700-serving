"""Tune the Triton WNA16 (int4_w4a16) fused-MoE kernel config for flashnext.

Companion to tune_fused_moe.py, which only covers the fp8_w8a8 block-scaled
MoE kernel used by the 27b/35b-a3b FP8 profiles. Flash-Next is served
int4_w4a16 (our RTN group-128 quantizer) via a structurally different Triton
kernel (fused_moe_kernel_gptq_awq, dispatched through
invoke_fused_moe_wna16_triton_kernel) with its own config schema (adds
SPLIT_K; drops the block_shape-keyed fp8 BLOCK_SIZE_K/scale-group coupling).

Geometry is EXPERT-PARALLEL, not TP-sliced: flashnext runs
enable_expert_parallel=True, so E = global num_experts / EP-ranks (whole
experts per rank) and N = moe_intermediate_size UNSLICED (not divided by TP,
unlike the FP8 profiles' TP-sliced N). Defaults below are flashnext at TP=4:
E=512/4=128, N=640, K=2560 (hidden_size), group_size=128, topk=10 -- exactly
the shape vLLM logs as missing at boot ("Config file not found at
.../E=128,N=640,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json").

Dummy-weight strategy: rather than hand-deriving the int4 interleave/repack
bit layout (ROCm-specific: on_rdna() triggers an extra repack_int4_to_int32
pass beyond the generic uint8 path -- see
vllm/model_executor/layers/fused_moe/oracle/int_wna16.py), this script builds
random weights in the pre-repack "Marlin"-shaped layout that
CompressedTensorsWNA16MoEMethod.create_weights() allocates (verified against
that method's get_weight_shape() and against the real checkpoint's per-expert
safetensors shapes), then calls vLLM's OWN
convert_to_wna16_moe_kernel_format(backend=TRITON, ...) to do the real
repack/transpose -- guaranteeing the exact runtime tensor layout without us
re-implementing ROCm's bit-packing. Since this is a wall-clock-only kernel
timing benchmark (not a correctness test), the weight *values* are
numerically meaningless random data; only shape/dtype matter.
"""
import argparse
import itertools
import json
import time

import torch
import triton

from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
)
import vllm.model_executor.layers.fused_moe.fused_moe as fm

E = 128
N = 640
K = 2560
GROUP_SIZE = 128
TOP_K = 10

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 0,
    "SPLIT_K": 1,
}


def build_weights(device="cuda"):
    """Build random int4-packed MoE weights via vLLM's own repack pipeline.

    Returns (w13_qweight, w2_qweight, quant_config) ready for fm.fused_experts.
    """
    torch.manual_seed(0)
    # Pre-repack "Marlin"-shaped layout (CompressedTensorsWNA16MoEMethod
    # .get_weight_shape(), is_transposed=True): w13 combines gate+up (2*N),
    # K-first packed 8-int4-per-int32.
    w13 = torch.randint(
        -(2**31), 2**31 - 1, (E, K // 8, 2 * N), dtype=torch.int32, device=device
    )
    w2 = torch.randint(
        -(2**31), 2**31 - 1, (E, N // 8, K), dtype=torch.int32, device=device
    )
    w13_scale = (torch.rand(E, K // GROUP_SIZE, 2 * N, device=device) * 0.02).to(
        torch.bfloat16
    )
    w2_scale = (torch.rand(E, N // GROUP_SIZE, K, device=device) * 0.02).to(
        torch.bfloat16
    )

    qa = QuantizationArgs(
        num_bits=4, type="int", symmetric=True, group_size=GROUP_SIZE,
        strategy="group",
    )
    w13_q, w2_q, w13_s, w2_s, *_ = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=None,
        quant_config=qa,
        input_dtype=torch.bfloat16,
        w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale,
    )
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w13_s, w2_scale=w2_s, block_shape=[0, GROUP_SIZE]
    )
    return w13_q, w2_q, quant_config


def make_tensors(M, device="cuda"):
    torch.manual_seed(0)
    hidden = (torch.randn(M, K, device=device) * 0.01).to(torch.bfloat16)
    topk_weights = torch.rand(M, TOP_K, device=device)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device).to(torch.int32)
    return hidden, topk_weights, topk_ids


def bench_one(M, cfg, weights, tensors, reps=50, warmup=5):
    w13_q, w2_q, quant_config = weights
    hidden, topk_weights, topk_ids = tensors
    try:
        with override_config(cfg):
            for _ in range(warmup):
                fm.fused_experts(hidden, w13_q, w2_q, topk_weights, topk_ids,
                                 global_num_experts=E, quant_config=quant_config)
        torch.cuda.synchronize()
        times = []
        with override_config(cfg):
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fm.fused_experts(hidden, w13_q, w2_q, topk_weights, topk_ids,
                                 global_num_experts=E, quant_config=quant_config)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        med = sorted(times)[len(times) // 2]
        return med
    except Exception:
        return None


def variants(base: dict):
    """Candidate configs around a base config.

    BLOCK_SIZE_N must be divisible by 8 (interleave path assert in
    invoke_fused_moe_wna16_triton_kernel). SPLIT_K left at 1 -- untested
    beyond that; the fp8 kernel has no SPLIT_K equivalent so there's no established
    starting point to sweep from, and >1 requires atomic-add output
    accumulation this script doesn't verify.
    """
    bm = {16, 32, 64}
    bn = {32, 64, 128, 256}
    bk = {32, 64, 128}
    gm = {1, 4, 8, 16}
    nw = {4, 8}
    ns = {2, 3}
    wpe = {0, 1, 2, 4}
    cands = [base]
    for k, pool in [("BLOCK_SIZE_M", bm), ("BLOCK_SIZE_N", bn),
                    ("BLOCK_SIZE_K", bk), ("GROUP_SIZE_M", gm),
                    ("num_warps", nw), ("num_stages", ns),
                    ("waves_per_eu", wpe)]:
        for v in pool:
            c = dict(base)
            c[k] = v
            cands.append(c)
    for m, n, k in itertools.product(bm, bn, bk):
        cands.append({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k,
                      "GROUP_SIZE_M": base["GROUP_SIZE_M"],
                      "num_warps": base["num_warps"],
                      "num_stages": base["num_stages"],
                      "waves_per_eu": base["waves_per_eu"],
                      "SPLIT_K": base["SPLIT_K"]})
    seen, out = set(), []
    for c in cands:
        if c["BLOCK_SIZE_N"] % 8 != 0:
            continue
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main():
    global E, N, K, GROUP_SIZE, TOP_K
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=str,
                    default="1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--E", type=int, default=128,
                    help="experts (local per EP rank for expert-parallel profiles)")
    ap.add_argument("--N", type=int, default=640,
                    help="moe_intermediate_size (unsliced for EP profiles)")
    ap.add_argument("--K", type=int, default=2560, help="hidden size")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=str, default=None)
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    E, N, K, GROUP_SIZE, TOP_K = args.E, args.N, args.K, args.group_size, args.topk
    if args.out is None:
        # NOTE: despite this kernel using int4_w4a16 weights, the runtime
        # config lookup (CompressedTensorsWNA16MoEMethod ->
        # int4_w4a16_moe_quant_config -> FusedMoEQuantConfig.use_int4_w4a16)
        # resolves to a *None* dtype string for this code path, so the
        # actual filename vLLM looks for at runtime has no ",dtype=..."
        # suffix. Confirmed via the live "Config file not found at ..."
        # boot warning after restarting the server with a dtype-suffixed
        # file in place.
        args.out = "/app/fused_moe_configs/" + get_config_file_name(
            E, N, None, [0, GROUP_SIZE]
        )
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

    weights = build_weights()

    results = {}
    for M in Ms:
        base = seed_cfgs.get(M)
        if base is None:
            if seed_cfgs:
                base = seed_cfgs[min(seed_cfgs, key=lambda k: abs(k - M))]
            else:
                base = dict(DEFAULT_CONFIG)
        configs = [base] if args.no_sweep else variants(base)
        tensors = make_tensors(M)
        print(f"M={M}: sweeping {len(configs)} configs from seed "
              f"{base.get('BLOCK_SIZE_M')}/{base.get('BLOCK_SIZE_N')}/"
              f"{base.get('BLOCK_SIZE_K')}", flush=True)
        scored = []
        for cfg in configs:
            t = bench_one(M, cfg, weights, tensors, reps=args.reps)
            if t is not None:
                scored.append((t, cfg))
            print(f"  {cfg['BLOCK_SIZE_M']}/{cfg['BLOCK_SIZE_N']}/"
                  f"{cfg['BLOCK_SIZE_K']} g{cfg['GROUP_SIZE_M']} "
                  f"w{cfg['num_warps']} s{cfg['num_stages']} "
                  f"wpe{cfg['waves_per_eu']} sk{cfg.get('SPLIT_K', 1)}: "
                  f"{(t*1e6 if t else float('nan')):>8.0f} us", flush=True)
        if not scored:
            print(f"  -> no valid config found for M={M}, skipping", flush=True)
            continue
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
