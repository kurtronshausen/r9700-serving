# Benchmarks

All benchmarks use `llama-benchy` (0.4.0, via `uvx`) against
`http://localhost:8180/v1`. Per-run tables live in [`benchmarks/`](benchmarks/).

## Setup

`--max-num-batched-tokens 4096`, `--max-num-seqs 1`, `--gpu-memory-utilization
0.9`, `-tp 2`, MTP4, `--kv-cache-dtype fp8`, `GPU_MAX_HW_QUEUES=1`.
Single-request numbers are invariant to `--max-num-seqs`; the server runs at
`--max-num-seqs 1` because concurrency loses to serial on this stack (see
Concurrency).

## Current (2026-08-10, vLLM 0.27.0, MTP4, fp8 KV, NCCL 4-ch)

Single-run data; averages across 3 benchmark sets are in the comparison table.

> Note: these runs were measured with the tuned-MoE `fused_moe_configs`
> deployed (`VLLM_TUNED_CONFIG_FOLDER`). Those configs were removed after the
> torch 2.13 / triton 3.8 bump (see README "Dead ends"), so current containers
> run stock MoE autotuning.

| model                     |   test |       t/s |
|:--------------------------|-------:|----------:|
| Qwen/Qwen3.6-27B-FP8      | pp2048 | 2924.03 ± 19.96 |
| Qwen/Qwen3.6-27B-FP8      |   tg32 |    87.42 ± 0.09 |
| Qwen/Qwen3.6-27B-FP8      |  tg128 |    76.34 ± 6.50 |
| Qwen/Qwen3.6-35B-A3B-FP8  | pp2048 | 11287.33 ± 367.56 |
| Qwen/Qwen3.6-35B-A3B-FP8  |   tg32 |   188.80 ± 13.15 |
| Qwen/Qwen3.6-35B-A3B-FP8  |  tg128 |   150.74 ± 11.28 |

### v0.26 → v0.27 upgrade

| model          | metric  | v0.26.2.dev0 | v0.27.0 | delta |
|:---------------|:--------|-------------:|-----------:|------:|
| 27B            | pp2048  |      ~2927   |    ~2916   | flat  |
| 27B            | tg32    |        ~75   |      ~87   | **+16%** |
| 27B            | tg128   |        ~66   |      ~76   | **+15%** |
| 35B-A3B        | pp2048  |     ~10864   |   ~11143   | +2.6%  |
| 35B-A3B        | tg32    |       ~182   |     ~189   | +3.8%  |
| 35B-A3B        | tg128   |       ~144   |     ~151   | +4.9%  |

The dense 27B benefits most from v0.27's spec-decode improvements and
multi-layer MTP refactor. Both models show consistent gains with no regressions.

## Baseline comparisons

| description                      | pp2048 t/s | tg32 t/s | vLLM  |
|:---------------------------------|-----------:|---------:|------:|
| 27B andy upstream (MTP3, fp8 KV) |     2750   |   81.9   | v0.25 |
| 35B-A3B no MTP, stock config     |   ~10075   |    ~83   | v0.26 |
| 27B (MTP4, fp8 KV, all opts)    |    ~2916   |    ~87   | v0.27 |
| 35B (MTP4, fp8 KV, all opts)    |   ~11143   |   ~189   | v0.27 |

The 35B-A3B MoE model is 3.8× faster on prefill and 2.2× faster on decode than
the dense 27B. MTP4 + NCCL tuning + v0.27 deliver 2.3×
decode throughput over the no-MTP baseline.

## MTP impact (35B-A3B)

Enabling MTP roughly doubles decode speed. Draft-token count was tuned on the
35B-A3B; both models now run MTP4.

| MTP | pp2048 (t/s) | tg32 (t/s) | acceptance |
|:----|-------------:|-----------:|-----------:|
| off |        10075 |       82.9 |         -  |
| 2   |   8000 ± 527 | 145.54 ± 6.51 | 59.8% |
| 3   | 9354 ± 171.64 | 143.80 ± 6.85 | **72.3%** |

## NCCL channels

Two R9700s on separate PCIe 5.0 x8 root ports, P2P disabled
(`NCCL_P2P_DISABLE=1`). `all_reduce_perf` (rccl-tests, out-of-place busbw GB/s):

| channels |     1M |    4M |     8M |    32M |    64M |
|:---------|-------:|------:|-------:|-------:|-------:|
| 1        |   8.04 |  9.51 |  11.09 |  11.81 |  11.94 |
| 2        |   8.88 | 11.19 |  12.15 |  12.54 |  12.61 |
| **4**    | **9.21** | 11.50 | 11.86 | **12.80** | **12.91** |
| 8        |   8.88 | 11.50 |  11.79 |  12.72 |  12.87 |
| 16       |   8.20 | 11.67 |  12.10 |  12.76 |  12.89 |
| 32       |   7.52 | 10.75 |  11.70 |  12.51 |  12.74 |
| 112      |   9.13 | 11.07 |  12.16 |  12.52 |  12.60 |

4 channels is fastest or near-fastest at every size. Serving A/B confirms
+12-19% tg128 decode on 4-ch vs 112-ch. Recommendation: pin both
`NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` to 4.

## Concurrency

35B-A3B. The table below is the original MTP3-era data; current MTP4 findings
follow. `total` = aggregate across all concurrent requests, `req` = per
request.

| test                | c1 (t/s) | c4 total (t/s) | c4 req (t/s) | scaling |
|:--------------------|---------:|---------------:|-------------:|--------:|
| pp2048 / tg128      |  157.73 ± 9.18 |  279.00 ± 7.90 |   91.20 ± 13.65 | 1.77x |
| pp2048 / tg512      |  140.69 ± 6.98 |  326.72 ± 8.35 |   89.91 ± 5.00 | **2.32x** |
| pp8192 / tg256      |  136.03 ± 1.74 |  187.88 ± 5.21 |   67.76 ± 15.30 | 1.38x |

Single-request decode is flat ~140-158 t/s regardless of generation length.
Longer generations scale best at concurrency 4 (2.32× for tg512). Longer prompts
cut scaling (prefill competes with decode for the `max_num_batched_tokens`
budget).

The same limit hits long *contexts* even with short generations: a c4 depth
sweep (tg32) keeps full-context prefill scaling well (~11-12k t/s aggregate at
d1024-d8192) but decode aggregate collapses from 347 t/s @ d1024 to 3.6 t/s @
d64000 — the 64k-token contexts can no longer share a batch within the 4096
token budget, and e2e TTFT reaches ~23.9 s. See
[`08_10_qwen3.6-35b-a3b_mtp4_depth_c4.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_depth_c4.md).

A c2 depth sweep finds the sweet spot for long-context serving: aggregate decode
matches c4 (geomean 86.1 vs 86.0 t/s) but e2e TTFT @ d64000 is ~15× better
(1.6 s vs 23.5 s), since the 64k prefill no longer serializes against a second
deep prefill in the batch budget. Per-request throughput is still 23-68% below
c1 at depth. A/Bs (fp8 KV, MTP2, 8192 batch) all lost to the baseline — the
deep-context decode cost is inherent to attending over a huge cached KV, not
fixable by those knobs. See
[`08_10_qwen3.6-35b-a3b_mtp4_depth_c2.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_depth_c2.md).

### c1 vs c2 head-to-head (2026-08-10)

A same-boot A/B (`--max-num-seqs 1` vs 2, identical depth sweep, see
[`08_10_qwen3.6-35b-a3b_mtp4_c1_vs_c2_depth.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_c1_vs_c2_depth.md))
settles it: **serial (c1) is strictly more efficient**. The earlier c2 sweep
was not a bad-process-state artifact — it reproduces closely (c2 total geomean
95 vs 86 before). Concurrency never reaches single-request decode:

| depth | c1 | c2 total | c2/req | c2 total / c1 |
|:------|---:|---------:|-------:|:--------------|
| d1024 | 180.0 |    127.2 |  113.1 | 71% |
| d16384| 156.1 |     97.4 |  104.1 | 62% |
| d32000| 175.7 |     76.7 |   97.4 | 44% |
| d64000| 171.2 |     46.9 |   77.0 | 27% |
| geomean | 169.5 |   94.8 |  100.4 | - |

Two concurrent requests (c2 total) move *fewer* tokens/s than one request alone
at every depth, so two deep requests finish faster back-to-back, and c2's
latency is worse (incremental TTFT @ d64000 1070 vs 1562 ms; full-context load
9292 vs 14178 ms). The server runs at `--max-num-seqs 1`. Use c2 only when
multiple users must progress simultaneously at ~45-55% lower per-request
decode.

## Depth sweep (35B-A3B, MTP4)

Single-request speeds at increasing prompt depth (d = prefix tokens before the
2048-token prompt), current stack. `pp2048` = **incremental** prefill of only the
2048 fresh tokens (depth prefix cached via `--enable-prefix-caching`); each new
token attends over the full cached KV, so this falls with depth. Full-context
rows (`ctx_pp`, e2e TTFT) are the comparable-to-old metric. Full table in
[`08_10_qwen3.6-35b-a3b_mtp4_depth.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_depth.md).

> Note: the old MTP3 sweep's `pp2048 @ dXXX` column measured full-context
> prefill (depth + 2048, uncached). Compare against the new `ctx_pp` rows, not
> `pp2048`: old 10043 @ d4096 → ctx_pp 11247, old 6774 @ d64000 → ctx_pp 6878,
> e2e TTFT @ d64000 9750 → 9308 ms. No regression.

| test            |               t/s |       ttfr (ms) |
|:----------------|------------------:|----------------:|
| pp2048 @ d0     | 8386.44 ± 3406.48 |   318.82 ± 181.06 |
| tg32 @ d0       |    187.97 ± 11.48 |                 |
| pp2048 @ d1024  |  6998.75 ± 111.74 |    293.66 ± 4.62 |
| tg32 @ d1024    |     196.10 ± 0.22 |                 |
| pp2048 @ d4096  |   5397.80 ± 11.07 |    380.37 ± 0.78 |
| tg32 @ d4096    |    175.41 ± 17.97 |                 |
| pp2048 @ d8192  |   5080.98 ± 8.20  |    404.03 ± 0.65 |
| tg32 @ d8192    |    174.81 ± 18.35 |                 |
| pp2048 @ d16384 |   4402.15 ± 4.97  |    466.19 ± 0.53 |
| tg32 @ d16384   |     175.45 ± 0.81 |                 |
| pp2048 @ d32000 |   3105.65 ± 8.75  |    660.41 ± 1.86 |
| tg32 @ d32000   |     149.69 ± 8.02 |                 |
| pp2048 @ d64000 |   1907.59 ± 20.29 |   1074.68 ± 11.37 |
| tg32 @ d64000   |     180.88 ± 23.24 |                 |

Decode stays flat ~150-196 t/s across all depths. Incremental prefill drops
with depth (attention span over the cached prefix grows): ~7000 t/s at d1024 →
~1900 t/s at d64000. TTFT of the incremental prompt grows sub-linearly with
depth thanks to prefix caching (294 → 1075 ms); full-context e2e TTFT scales
linearly (258 ms @ d1024 → 9308 ms @ d64000).
