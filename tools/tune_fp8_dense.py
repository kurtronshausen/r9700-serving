import argparse
import json
import time

import torch
import triton

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.utils.platform_utils import get_device_name_as_file_name

# The repo serves Qwen3.6-35B-A3B-FP8 at tensor-parallel size 2. vLLM keys the
# dense w8a8 block-fp8 config file on the per-GPU weight shape
# (N = local output channels, K = local input channels). These five shapes are
# the ones the 35B-A3B model instantiates (per-GPU, TP=2).
DEFAULT_SHAPES = [(2048, 2048), (2048, 256), (4608, 2048), (512, 2048),
                  (6144, 2048)]
BLOCK_N = 128
BLOCK_K = 128
DEVICE_NAME = get_device_name_as_file_name()

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def dense_configs():
    """Candidate configs for block_shape=[128,128].

    BLOCK_SIZE_K must divide block_k: the kernel loads one scale per k-step
    (offs_ks = k_start // group_k), so with 128-wide scale tiles only BK in
    {64,128} is numerically valid (BK=256 mixes two different scale groups).
    BLOCK_SIZE_N in {128,256} keeps the per-128-row scale indexing valid.
    BLOCK_SIZE_M <= 128 avoids the register-pressure/compile regimes that make
    Triton spin for minutes on gfx1201 for this kernel."""
    out = []
    for bm in [16, 32, 64, 128]:
        for bn in [128, 256]:
            for bk in [64, 128]:
                for gm in [1, 4, 8, 16, 32, 64]:
                    for w in [4, 8]:
                        for s in [2, 3, 4]:
                            out.append({
                                "BLOCK_SIZE_M": bm,
                                "BLOCK_SIZE_N": bn,
                                "BLOCK_SIZE_K": bk,
                                "GROUP_SIZE_M": gm,
                                "num_warps": w,
                                "num_stages": s,
                            })
    return out


def make_tensors(M, N, K, out_dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    A = (torch.rand(M, K, device=device) - 0.5) * 2 * fp8_max
    A = A.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B = (torch.rand(N, K, device=device) - 0.5) * 2 * fp8_max
    B = B.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    As = torch.rand(M, k_tiles, device=device) * 1e-2
    Bs = torch.rand(n_tiles, k_tiles, device=device) * 1e-2
    C = torch.empty(M, N, dtype=out_dtype, device=device)
    return A, B, As, Bs, C


def run_kernel(A, B, As, Bs, C, M, N, K, cfg):
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) *
            triton.cdiv(N, cfg["BLOCK_SIZE_N"]), )
    _w8a8_triton_block_scaled_mm[grid](
        A, B, C, As, Bs, M, N, K, BLOCK_N, BLOCK_K,
        A.stride(-2), A.stride(-1), B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1), As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0), **cfg)


def is_correct(A, B, As, Bs, C, N, K):
    """Compare the kernel output in C (already containing the last run) against
    an fp32 reference. B/As/Bs must come from make_tensors with the same seed."""
    Af = A.float() * As.float().repeat_interleave(BLOCK_K, dim=1)
    Bf = (B.float() *
          Bs.float().repeat_interleave(BLOCK_N, dim=0).repeat_interleave(
              BLOCK_K, dim=1))
    ref = Af @ Bf.t()
    scale = ref.abs().max()
    return ((C.float() - ref).abs().max() / scale) < 1e-2


def bench_one(A, B, As, Bs, C, M, N, K, cfg, iters=100, warmup=10):
    def run():
        run_kernel(A, B, As, Bs, C, M, N, K, cfg)

    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters * 1000
    except (triton.OutOfResources, triton.runtime.autotuner.OutOfResources,
            RuntimeError):
        return None


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def tune_shape(N, K, Ms, configs, iters, failed):
    print(f"[{time.strftime('%H:%M:%S')}] tune N={N} K={K}", flush=True)
    best = {}
    for M in Ms:
        t0 = time.time()
        A, B, As, Bs, C = make_tensors(M, N, K)
        scored = []
        for cfg in configs:
            if key(cfg) in failed:
                continue
            t = bench_one(A, B, As, Bs, C, M, N, K, cfg, iters=iters)
            if t is None:
                failed.add(key(cfg))
                continue
            scored.append((t, cfg))
        scored.sort(key=lambda x: x[0])
        if not scored:
            print(f"  M={M}: NO valid configs, using default", flush=True)
            best[M] = dict(DEFAULT_CONFIG)
            continue
        bt, bc = scored[0]
        # Gate the winner numerically; a "fast" config that is also wrong would
        # silently corrupt generations (e.g. BLOCK_SIZE_K=256 mixes scale
        # groups). Fall back to the stock default if it fails.
        run_kernel(A, B, As, Bs, C, M, N, K, bc)
        torch.cuda.synchronize()
        if not is_correct(A, B, As, Bs, C, N, K):
            print(f"  M={M}: best config failed numeric check, using default "
                  f"(was BM{bc['BLOCK_SIZE_M']}/BN{bc['BLOCK_SIZE_N']}/"
                  f"BK{bc['BLOCK_SIZE_K']} g{bc['GROUP_SIZE_M']} "
                  f"w{bc['num_warps']} s{bc['num_stages']})", flush=True)
            best[M] = dict(DEFAULT_CONFIG)
            continue
        best[M] = bc
        print(f"  M={M}: best "
              f"BM{bc['BLOCK_SIZE_M']}/BN{bc['BLOCK_SIZE_N']}/"
              f"BK{bc['BLOCK_SIZE_K']} g{bc['GROUP_SIZE_M']} "
              f"w{bc['num_warps']} s{bc['num_stages']}: {bt:.0f} us "
              f"({time.time()-t0:.0f}s, {len(scored)} cands)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=str,
                    default=",".join(f"{n}:{k}" for n, k in DEFAULT_SHAPES))
    ap.add_argument("--Ms", type=str,
                    default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=str, default="/app/fp8_configs")
    args = ap.parse_args()

    print(f"triton {triton.__version__}", flush=True)
    torch.cuda.set_device(0)
    shapes = [tuple(map(int, s.split(":"))) for s in args.shapes.split(",")]
    Ms = [int(x) for x in args.Ms.split(",")]
    configs = dense_configs()
    failed = set()
    print(f"{len(configs)} configs, {len(shapes)} shapes, {len(Ms)} Ms",
          flush=True)

    for N, K in shapes:
        t0 = time.time()
        best = tune_shape(N, K, Ms, configs, args.iters, failed)
        name = (f"N={N},K={K},device_name={DEVICE_NAME},dtype=fp8_w8a8,"
                f"block_shape=[{BLOCK_N},{BLOCK_K}].json")
        with open(f"{args.out}/{name}", "w") as f:
            json.dump({str(m): c for m, c in best.items()}, f, indent=4)
            f.write("\n")
        print(f"wrote {args.out}/{name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()