# vLLM on Radeon AI PRO R9700

Build and run vLLM from source for AMD Radeon AI PRO R9700 GPUs. The default
configuration targets two R9700s (`gfx1201`) and serves a model through vLLM's
OpenAI-compatible API.

## Requirements

- Docker with the Compose plugin (`docker compose`), or Podman (`podman
  compose`); `just` recipes default to Docker
- [`just`](https://just.systems/)
- SELinux hosts need no special relabeling: bind mounts mount unlabeled because
  the container runs with `label=disable`
- One or more R9700 GPUs; the included configuration assumes two

## Quick start

```sh
cp .env.example .env  # Build version pins + default model profile (untracked)
just build       # Build localhost/vllm-fullbuild:latest
just check       # Validate the compose config for the selected profile
just up          # Start vLLM in the background (default: 35B-A3B MoE)
just --set model qwen3.6-27b up  # Switch to dense 27B model
just logs        # Follow service logs
just down        # Stop and remove containers
```

To use Podman: `just --set runtime podman build` or `RUNTIME=podman just up`.
Run `just --list` to see all recipes including `rebuild` (force-rebuild) and
`clear-vllm-caches` (wipe host-side Triton/Inductor/AITER caches; preserves
HuggingFace model cache).

Always go through `just`: `compose.yaml` interpolates the model arguments from
`env/<profile>.env`, which the recipes pass to compose via `--env-file`. A bare
`docker compose up` fails with a required-variable error rather than starting a
server with no model.

The vLLM OpenAI-compatible API is available at `http://localhost:8180/v1`.
Other containers on the same compose network can reach it via the `llm-backend`
network alias instead of the host port.

## Configuration

Build versions are pinned in `.env` (untracked; copy `.env.example` to create it).

| component    | version |
|:-------------|:--------|
| ROCm         | 7.14.0 (`rocm/dev-ubuntu-24.04:7.14.0-full`) |
| PyTorch      | 2.13.0+rocm7.14.0 |
| vLLM         | 0.27.0 |
| AITER        | v0.1.19.post2 |
| Flash Attention | @ 1cc7ff67 |

ROCm 7.14 is on AMDs "TheRock" technology-preview stream (7.9/7.13/7.14); the
production 7.2.x line lacks RDNA4/`gfx1201` support. AITER `v0.1.19.post2` is
the latest tagged release; vLLM is the 0.27.0 release since `gfx1201`
requires source builds.

The active model is `Qwen/Qwen3.6-35B-A3B-FP8` (MoE, 35B total / 3B active);
switch to the dense `Qwen/Qwen3.6-27B-FP8` with `just --set model qwen3.6-27b`.
Model selection is controlled by `MODEL_PROFILE` in `.env` — override inline
with `MODEL_PROFILE=qwen3.6-27b just up`.

Runtime environment is split across files:
- `env/2xr9700.vllm.common` — two-GPU ROCm config (arch, NCCL, HSA, compile caches)
- `env/aiter-unified-attention.env` — enables AITER unified attention only
- `env/qwen3.6-35b-a3b.env` — MoE model config (path, tokenizer, MTP disabled, tool use)
- `env/qwen3.6-27b.env` — dense 27B model config

### Chat template

The froggeric v21.3 chat template (`chat-templates/qwen36.jinja`) is wired in via
`--chat-template` — it partitions thinking into the `reasoning` field and the
final answer into `content`.

### Non-standard vLLM flags

- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`** (`VLLM_TOOL_CHOICE`, both profiles): OpenAI
  tool-calling with Qwen's `qwen3_coder` parser. `--reasoning-parser qwen3` is
  also required for the froggeric chat template to correctly split thinking
  into the `reasoning` field.
- **`--limit-mm-per-prompt '{"image": 99, "audio": 0, "video": 0}'`**: multimodal
  images allowed, audio/video disabled.
- **`--override-generation-config`**: server-side sampling defaults
  (`temperature` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0, no penalties).
- **`--enable-prefix-caching`**: reuse KV for shared prompt prefixes.
- **`--max-model-len 131072`**, **`--max-num-seqs 1`**, **`-tp 2`**,
  **`--gpu-memory-utilization 0.8`**.
- **`--kv-cache-dtype bfloat16`** (`VLLM_KV_CACHE_DTYPE`). The AITER BF16 LDS-fit
  patch (`patches/aiter/unified-attention-bf16-kv.patch`) caps `TILE_SIZE` and
  `attn_stages` to fit 64 KiB LDS. Prior "garbage" output was caused by MTP token
  loops (see "MTP bug" below), not the patch — BF16 is now the default.
- **`--attention-backend ROCM_AITER_UNIFIED_ATTN`** + `--speculative-config`
  (MTP4 on 27B only; disabled on 35B, see "MTP workaround").

### Runtime env knobs

Non-standard environment set across `compose.yaml`, `Dockerfile.fullbuild`,
and `env/2xr9700.vllm.common` (loaded via `env_file`):

| var | value | why |
|:----|:------|:----|
| `GPU_MAX_HW_QUEUES` | `1` | avoids RDNA4 decode regression (see tuning) |
| `NCCL_P2P_DISABLE` | `1` | two GPUs on separate PCIe root ports |
| `NCCL_MIN/MAX_NCHANNELS` | `4` | bandwidth sweet spot (see tuning) |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `1` | needed for the ROCm stack |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | avoid scratch reallocation stalls |
| `HIP_FORCE_DEV_KERNARG` | `1` | force device-side kernel args |
| `LD_PRELOAD` | `libamd_smi.so` | expose GPU metrics via amd_smi |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` | prefer hipBLASLt GEMMs |
| `SAFETENSORS_FAST_GPU` | `1` | fast safetensors load on GPU |
| `PYTORCH_NVML_BASED_CUDA_CHECK` | `1` | NVML-based CUDA check on ROCm |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | enable Triton FA on AMD |
| `TOKENIZERS_PARALLELISM` | `false` | avoid HF tokenizer thread churn |
| `TORCHINDUCTOR_CACHE_DIR` | `/root/.cache/torchinductor` | persist compile cache (host-mounted) |
| `TRITON_CACHE_DIR` | `/root/.cache/triton` | persist Triton compile cache (host-mounted) |
| `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` | `0,1` | select the two R9700s |
| `HIP_ARCHITECTURES`/`AMDGPU_TARGETS`/etc. | `gfx1201` | target the R9700 ISA |

The `VLLM_ROCM_USE_AITER_*` flags in `env/aiter-unified-attention.env` enable
only AITER's unified attention; MoE/linear/RMSNorm stay on stock vLLM kernels
(AITER's MoE/FP8 backends don't support `gfx1201` yet).

Key tuning decisions:
- **MTP4 speculative decoding** (27B only): 4 draft tokens per step (~72% acceptance
  on 27B), roughly doubles decode throughput vs no MTP. MTP is **disabled on
  35B-A3B** — see "MTP bug" below.
- **`GPU_MAX_HW_QUEUES=1`** is required. Multiple queues cause a 55-63% decode
  throughput regression on RDNA4 — one queue per process avoids kernel launch
  scheduling overhead.
- **NCCL channels pinned to 4** (`NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`):
  the bandwidth sweet spot for two GPUs on separate PCIe 5.0 x8 root ports with
  P2P disabled.
- **BF16 KV cache**: restored via an AITER LDS-fit patch. Prior "garbage" output
  was caused by MTP token loops, not the patch. BF16 at ~88 t/s outperforms fp8.
- **Tuned dense w8a8 block-FP8 configs** (`fp8_configs/N=*,K=*,device_name=AMD_Radeon_R9700,...json`):
  the 5 per-GPU weight shapes for both 35B-A3B and 27B (TP=2) are now tuned for the
  R9700 via `tools/tune_fp8_dense.py`. Sweeps 576 Triton tile configurations per shape
  with fp32-reference numeric gating (eliminating structurally invalid configs — BK=256
  mixes 128-wide scale groups). Same-boot A/B vs stock defaults: **35B tg32 +4%, tg128 +5%;
  27B tg32 +19%, pp2048 +3%** (tg128 flat).
- **Tuned fused MOE configs** (`fused_moe_configs/E=256,N=256,...json`): tuned via
  `tools/tune_fused_moe.py`. vLLM keys the config file on the per-GPU geometry at
  TP size 2 (`E=256,N=256` = local experts × local intermediate); an earlier
  `E=256,N=512` file never matched, so the server silently ran the stock MoE
  config. Enabled via `VLLM_TUNED_CONFIG_FOLDER=/app/fused_moe_configs`.
- **`--max-num-batched-tokens 4096`** is required for the MoE model (its
  gated-delta layers force an attention block size of 2112 tokens).

### MTP bug (35B only)

vLLM's native MTP speculative decoding has a confirmed bug with Qwen3-MoE models
(`Qwen3.6-35B-A3B`, `Qwen3.6-27B-A3B`) where deep agentic conversations degenerate
into garbled token loops with no usable output. This affects multiple upstream issues:

| Issue | Summary |
|:------|:--------|
| [vllm-project/vllm#47087](https://github.com/vllm-project/vllm/issues/47087) | MTP token loops on Qwen3-MoE; output quality collapse mid-conversation |
| [vllm-project/vllm#35288](https://github.com/vllm-project/vllm/issues/35288) | Native MTP speculative decoding instability on MoE architectures |

**Impact**: 35B-A3B throughput drops from ~185 tg32 (MTP4) to ~83 tg32 (no MTP).
The 27B (dense) model is unaffected and MTP works correctly.

**Workaround**: `compose.yaml` and `env/qwen3.6-35b-a3b.env` disable MTP for
the 35B model only. The 35B profile sets `VLLM_SPEC_DECODE=` (empty), skipping
`--speculative-config`. The 27B profile inherits `VLLM_SPEC_DECODE` from
`qwen3.6.env.common` and retains MTP4.

## Dead ends

- **AITER MoE/FP8 backend on gfx1201**: vLLM aborts at startup. Enable once
  upstream AITER adds RDNA4 support.
- **`--enable-expert-parallel` on top of `-tp 2`**: regresses decode ~7-12% on
  the 35B-A3B (tg32 160-175 vs ~181-191, tg128 135-137 vs ~146) with flat
  prefill. EP's AllToAll doesn't pay off for a 3B-active MoE at tp=2. Skip at
  this scale; revisit only for much larger active-parameter MoEs.

## Performance

Measured on 2× R9700, BF16 KV + tuned MOE config, single request, vLLM 0.27.0,
torch 2.13, triton 3.8.0+git (ROCm 7.14.0). MTP4 is enabled for the 27B model;
disabled for 35B-A3B (see "MTP bug" above).

| model                           | MTP (draft #)      | pp2048 t/s | tg32 t/s | tg128 t/s |
|:--------------------------------|:-------------------|-----------:|---------:|----------:|
| Qwen3.6-27B (Andy & upstream baseline) | MTP3, fp8 KV |     2750 |    81.9 |    —     |
| Qwen3.6-27B-FP8 (v0.26)         | MTP4, fp8 KV       |    ~2927 |     ~75 |    ~66   |
| Qwen3.6-27B-FP8 (v0.27)         | MTP4, fp8 KV       |    ~2916 |     ~87 |    ~76   |
| Qwen3.6-35B-A3B-FP8 (v0.26)     | MTP4, fp8 KV       | ~10864 |    ~182 |   ~144   |
| Qwen3.6-35B-A3B-FP8 (v0.27)     | MTP4, fp8 KV       | ~11143 |    ~189 |   ~151   |
| Qwen3.6-35B-A3B-BF16+MoETuned+MtPOff | MTP off       | ~8788 |   ~87.8 |   ~87.1  |
| Qwen3.6-35B-A3B-BF16+MoETuned+DenseTuned+MtPOff | MTP off, tuned dense | ~8510 |  **91.0** |  **91.3** |
| Qwen3.6-27B-BF16+MTP4           | MTP4, bf16 KV       |    ~2471 |   ~80.6 |   ~63.7  |
| Qwen3.6-27B-BF16+MTP4+DenseTuned | MTP4, bf16 KV, tuned dense | ~2500 |  **90.8** |   ~69  |

Full methodology, depth sweeps, and tuning history in
[`BENCHMARKS.md`](BENCHMARKS.md).

### Depth sweep (35B-A3B, MTP off, tuned dense vs stock)

Deep-context decode is dominated by attention over the cached KV, so the GEMM
tuning benefit narrows with depth. Same-boot A/B (bf16 KV, tuned MoE, thinking
off):

| depth | stock tg32 | +tuned dense tg32 | uplift |
|------:|-----------:|------------------:|:------|
| 0     | 86.8 | **90.3** | +4% |
| 4096  | 86.8 | **91.4** | +5% |
| 65536 | 78.2 | **79.1** | +1% |
| 128000| 71.4 | **72.3** | +1% |

### Long-context concurrency

Decode cost is dominated by attending over the cached KV, so concurrent
deep-context requests degrade sharply. The table below was measured on 35B-A3B
(BF16 KV + tuned MOE, tg32, MTP disabled).
`c2 total` = aggregate across 2 concurrent requests, `c2/req` = per request.

| depth | c1 | c2 total | c2/req |
|------:|---:|---------:|-------:|
| d1024 | 180 |      127 |    113 |
| d16384| 156 |       97 |    104 |
| d32000| 176 |       77 |     97 |
| d64000| 171 |       47 |     77 |

The server runs at **c1 (`--max-num-seqs 1`)**, which a head-to-head confirms
is the most efficient long-context setup: two concurrent requests (c2 total,
geomean ~95 t/s) never reach one request's decode (c1, geomean ~170 t/s) at any
depth, so even two deep requests finish faster served back-to-back, and c2's
latency is worse too (incremental TTFT @ d64000 1070 vs 1562 ms; full-context
load 9292 vs 14178 ms). c2 ≈ c4 aggregate (geomean 86-95) — neither reaches c1.
Use concurrency only when multiple users must progress simultaneously and you
can accept ~45-55% lower per-request decode; for raw throughput or a single
active session, serial wins. A/Bs of fp8 KV, MTP2, and an 8192-token batch
budget all lost to the tuned baseline; the cost is inherent to the stack.
