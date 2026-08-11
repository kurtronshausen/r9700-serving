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
- **`--kv-cache-dtype fp8`** (`VLLM_KV_CACHE_DTYPE`). bf16 KV required an AITER
  LDS-fit patch under triton 3.7.1 (torch 2.12) that produced garbage output under
  triton 3.8.0 (torch 2.13), so fp8 is the safe choice.
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
  on 27B), roughly doubles decode throughput vs no MTP. MTP is **disabled on 35B-A3B**
  — see "MTP workaround" below.
- **`GPU_MAX_HW_QUEUES=1`** is required. Multiple queues cause a 55-63% decode
  throughput regression on RDNA4 — one queue per process avoids kernel launch
  scheduling overhead.
- **NCCL channels pinned to 4** (`NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`):
  the bandwidth sweet spot for two GPUs on separate PCIe 5.0 x8 root ports with
  P2P disabled.
- **fp8 KV cache**: the default. bf16 KV was tested and showed zero perf
  regression, but the AITER patch needed to fit the 64 KiB LDS limit produced
  garbage output under triton 3.8.0 (torch 2.13), so fp8 KV is the safe choice.
- **`--max-num-batched-tokens 4096`** is required for the MoE model (its
  gated-delta layers force an attention block size of 2112 tokens).

### MTP workaround (35B only)

vLLM's native MTP speculative decoding has a known bug with Qwen3-MoE models
(`Qwen3.6-35B-A3B`, `Qwen3.6-27B-A3B`) where deep agentic conversations can
degenerate into garbled token loops producing no usable output
([vllm-project/vllm#47087](https://github.com/vllm-project/vllm/issues/47087)).
The bug causes output quality collapse — not just slight degradation — and can
be triggered mid-conversation even after many clean turns.

**Impact on 35B-A3B**: throughput drops from ~185 tg32 (with MTP4) to
~83 tg32 (no MTP). The 27B (dense) model is unaffected and MTP works correctly.

**Workaround**: `compose.yaml` and `env/qwen3.6-35b-a3b.env` have been patched
to disable MTP for the 35B model only. The 27B model profile still uses MTP4.
This resolves to a simple variable override — the 35B profile sets
`VLLM_SPEC_DECODE=` (empty), which skips the `--speculative-config` flag.
The 27B profile inherits the shared `VLLM_SPEC_DECODE` from `qwen3.6.env.common`.

## Dead ends

- **Tuned MoE kernel configs**: per-token-count optimal Triton tile configs for
  the stock vLLM `fused_experts` kernel. Re-tuned for triton 3.8.0 and A/B tested
  vs stock defaults (depth 0–128K). Gains at 0–32K depth (ctx_tg +16% at d32K,
  tg32 +13% at d16K) but **losses at deep context** (ctx_tg −14% at d64K,
  −13% at d128K; tg32 −9% at d128K). The sweep picks configs that favor
  shallow-batch MoE at the expense of deep-context codegen — not safe to deploy
  for long-context serving. Dropped; MoE uses stock autotuned defaults.
- **AITER MoE/FP8 backend on gfx1201**: vLLM aborts at startup. Enable once
  upstream AITER adds RDNA4 support.
- **`--enable-expert-parallel` on top of `-tp 2`**: regresses decode ~7-12% on
  the 35B-A3B (tg32 160-175 vs ~181-191, tg128 135-137 vs ~146) with flat
  prefill. EP's AllToAll doesn't pay off for a 3B-active MoE at tp=2. Skip at
  this scale; revisit only for much larger active-parameter MoEs.

## Performance

Measured on 2× R9700, fp8 KV, single request, vLLM 0.27.0, torch 2.13,
triton 3.8.0. No tuned MoE configs — stock triton autotuned defaults.
MTP4 is enabled for the 27B model; disabled for 35B-A3B (see "MTP workaround").

| model                     | pp2048 t/s | tg32 t/s | tg128 t/s |
|:--------------------------|-----------:|---------:|----------:|
| Qwen3.6-27B (Andy & upstream baseline) |     2750 |    81.9 |    —     |
| Qwen3.6-27B-FP8 (v0.26)     |    ~2927 |     ~75 |    ~66   |
| Qwen3.6-27B-FP8 (v0.27)     |    ~2916 |     ~87 |    ~76   |
| Qwen3.6-35B-A3B-FP8 (v0.26)    | ~10864 |    ~182 |   ~144   |
| Qwen3.6-35B-A3B-FP8 (v0.27)    | ~11143 |    ~189 |   ~151   |
| Qwen3.6-35B-A3B-FP8 (MTP disabled) | ~10075 |     ~83 |    —     |
| Qwen3.6-27B-FP8 (current)      |  ~2924 |     ~87 |     ~76   |

Full methodology, depth sweeps, and tuning history in
[`BENCHMARKS.md`](BENCHMARKS.md).

### Long-context concurrency

Decode cost is dominated by attending over the cached KV, so concurrent
deep-context requests degrade sharply. The table below was measured on 35B-A3B
(tg32, MTP4). The current production config has MTP disabled for 35B per the
workaround above — actual throughput at every depth is ~2x lower.
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
