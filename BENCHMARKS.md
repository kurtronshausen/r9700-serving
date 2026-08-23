# Benchmarks

Current-state benchmark setup and index for this stack. All benchmarks use
`llama-benchy` (0.4.0, via `uvx`) against `http://localhost:8000/v1`.
Historical data (vLLM 0.25/0.26/0.27.0 era, MTP4 35B sweeps, concurrency
A/Bs) is archived in [`archive/BENCHMARKS.md`](archive/BENCHMARKS.md) with
per-run files in [`archive/benchmarks/`](archive/benchmarks/).

## Setup (current)

vLLM 0.27.1 + local patches (see README "Source-build patches"), torch 2.13,
triton 3.8.0, ROCm 7.14.0, AITER v0.1.20 unified attention, froggeric chat
template v22.3, thinking off. `-tp 2`, `--gpu-memory-utilization 0.85`,
`GPU_MAX_HW_QUEUES=1`. fp8 KV is the default on the dense profiles; 35B-A3B
runs bf16 KV. `--max-num-seqs 2` on the MTP profiles (the #35288 cap);
`--max-num-batched-tokens 4096` on 35B-A3B.

Single-request numbers are invariant to `--max-num-seqs`; long-context
concurrency degrades sharply (see the c1-vs-c2 head-to-head in
[`archive/BENCHMARKS.md`](archive/BENCHMARKS.md)).

## Current per-run files

| file | contents |
|:-----|:---------|
| [`benchmarks/08_21_qwen3.8-27b_dflash2_bench.md`](benchmarks/08_21_qwen3.8-27b_dflash2_bench.md) | Qwen3.8-27B d0 with DFlash2 drafter (V2 runner; tg32 ~88 t/s vs MTP3 ~62) |
| [`benchmarks/08_21_qwen3.8-27b_v0.28.0rc2_bench.md`](benchmarks/08_21_qwen3.8-27b_v0.28.0rc2_bench.md) | Qwen3.8-27B d0 on vLLM 0.28.0rc2 bump (MTP3, no regression; prefix probe 0%) |
| [`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_patches_bench.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_patches_bench.md) | Qwen3.8-27B d0 + depth, vLLM 0.27.1 + 3 local patches |
| [`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_d0.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_d0.md) | Qwen3.8-27B d0 re-bench (patched) |
| [`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md) | Qwen3.8-27B depth re-sweep d4K–d128K (patched) |
| [`benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_bench.md`](benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_bench.md) | Qwen3.8-27B d0, pre-patch fp8-KV baseline |
| [`benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_depth.md) | Qwen3.8-27B depth sweep, pre-patch |
| [`benchmarks/08_20_qwen3.8-27b_v223_template_stability.md`](benchmarks/08_20_qwen3.8-27b_v223_template_stability.md) | v22.3 chat-template stability run (multi-turn, tools, load 40/40) |
| [`benchmarks/08_12_qwen3.6-27b_bf16_mtp4_bench.md`](benchmarks/08_12_qwen3.6-27b_bf16_mtp4_bench.md) | Qwen3.6-27B d0 (latest for that profile: bf16 KV, MTP4) |
| [`benchmarks/08_12_qwen3.6-35b-a3b_bf16_mtpoff_bench.md`](benchmarks/08_12_qwen3.6-35b-a3b_bf16_mtpoff_bench.md) | 35B-A3B d0 (latest for that profile: bf16 KV, MTP off, tuned MoE) |
| [`benchmarks/08_11_qwen3.6-35b-a3b_BF16+MoeTuned+MtPOff_128k_depth.md`](benchmarks/08_11_qwen3.6-35b-a3b_BF16+MoeTuned+MtPOff_128k_depth.md) | 35B-A3B depth sweep to 128K (MTP off, tuned MoE) |
| [`benchmarks/STABILITY_TESTS.md`](benchmarks/STABILITY_TESTS.md) | smoke/stress/long-context coherence test scripts + baselines |
| [`benchmarks/prefix_cache_probe.py`](benchmarks/prefix_cache_probe.py) | prefix-cache hit-rate probe (used by the AGENTS.md update workflow) |

## NCCL channels (current hardware)

Two R9700s on separate PCIe 5.0 x8 root ports, P2P disabled
(`NCCL_P2P_DISABLE=1`). `all_reduce_perf` (rccl-tests, out-of-place busbw GB/s):

| channels | 1M | 4M | 8M | 32M | 64M |
|:---------|-------:|------:|-------:|-------:|-------:|
| 1 | 8.04 | 9.51 | 11.09 | 11.81 | 11.94 |
| 2 | 8.88 | 11.19 | 12.15 | 12.54 | 12.61 |
| **4** | **9.21** | 11.50 | 11.86 | **12.80** | **12.91** |
| 8 | 8.88 | 11.50 | 11.79 | 12.72 | 12.87 |
| 16 | 8.20 | 11.67 | 12.10 | 12.76 | 12.89 |
| 32 | 7.52 | 10.75 | 11.70 | 12.51 | 12.74 |
| 112 | 9.13 | 11.07 | 12.16 | 12.52 | 12.60 |

4 channels is fastest or near-fastest at every size; serving A/B confirms
+12-19% tg128 decode on 4-ch vs 112-ch. This is why
`NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` are pinned to 4.
