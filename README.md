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
just build       # Build localhost/vllm-fullbuild:latest
just up          # Start vLLM in the background (default: 35B-A3B MoE)
just --set model qwen3.6-27b up  # Switch to dense 27B model
just logs        # Follow service logs
just down        # Stop and remove containers
```

To use Podman: `just --set runtime podman build` or `RUNTIME=podman just up`.
Run `just --list` to see all recipes including `rebuild` (force-rebuild) and
`clear-vllm-caches` (wipe host-side Triton/Inductor/AITER caches; preserves
HuggingFace model cache).

The vLLM OpenAI-compatible API is available at `http://localhost:8180/v1`.
Other containers on the same compose network can reach it via the `llm-backend`
network alias instead of the host port.

## Configuration

Build versions are pinned in `.env`.

| component    | version |
|:-------------|:--------|
| ROCm         | 7.14.0 (`rocm/dev-ubuntu-24.04:7.14.0-full`) |
| PyTorch      | 2.12.0+rocm7.14.0 |
| vLLM         | 0.27.0rc2 |
| AITER        | v0.1.19.post2 |
| Flash Attention | @ 1cc7ff67 |

ROCm 7.14 is on AMDs "TheRock" technology-preview stream (7.9/7.13/7.14); the
production 7.2.x line lacks RDNA4/`gfx1201` support. AITER `v0.1.19.post2` is
the latest tagged release; vLLM is the 0.27 RC2 preview since `gfx1201`
requires source builds.

The active model is `Qwen/Qwen3.6-35B-A3B-FP8` (MoE, 35B total / 3B active);
switch to the dense `Qwen/Qwen3.6-27B-FP8` with `just --set model qwen3.6-27b`.
Model selection is controlled by `MODEL_PROFILE` in `.env` — override inline
with `MODEL_PROFILE=qwen3.6-27b just up`.

Runtime environment is split across files:
- `env/2xr9700.vllm.common` — two-GPU ROCm config (arch, NCCL, HSA)
- `env/aiter-unified-attention.env` — enables AITER unified attention only
- `env/aiter-moe-unified-attention.env` — also enables AITER MoE/FP8 kernels
  (not active: AITER's FP8 MoE backend does not yet support `gfx1201`)
- `env/qwen3.6-35b-a3b.env` — MoE model config (path, tokenizer, MTP, tool use)
- `env/qwen3.6-27b.env` — dense 27B model config

### Chat template

The official Qwen3.6 template is replaced with the community "froggeric" fixed
template (v21.3) at `chat-templates/qwen36.jinja`, wired in via
`VLLM_CHAT_TEMPLATE`. It fixes render errors, KV-cache invalidation, and
agentic-loop stalls in the stock Qwen template, and adds `think_on`/`think_off`
tokens, tool-error detection, and per-tool arg truncation. Passed
`--default-chat-template-kwargs '{"preserve_thinking": true}'` (`VLLM_CHAT_KWARGS`)
keeps past turns' reasoning in the prompt; set it to `false` to drop it.

### Non-standard vLLM flags

- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`** (`VLLM_TOOL_CHOICE`, 35B profile): OpenAI
  tool-calling with Qwen's `qwen3_coder` parser.
- **`--limit-mm-per-prompt '{"image": 99, "audio": 0, "video": 0}'`**: multimodal
  images allowed, audio/video disabled.
- **`--override-generation-config`**: server-side sampling defaults
  (`temperature` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0, no penalties).
- **`--enable-prefix-caching`**: reuse KV for shared prompt prefixes.
- **`--max-model-len 131072`**, **`--max-num-seqs 1`**, **`-tp 2`**,
  **`--gpu-memory-utilization 0.9`**.
- **`--kv-cache-dtype auto`** (`VLLM_KV_CACHE_DTYPE`) — bf16 for these models.
- **`--attention-backend ROCM_AITER_UNIFIED_ATTN`** + `--speculative-config`
  (MTP4, see above).

### Runtime env knobs

Non-standard environment set in `compose.yaml` and `Dockerfile.fullbuild`:

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
| `VLLM_TUNED_CONFIG_FOLDER` | `/app/fused_moe_configs` | deploy tuned MoE tile configs |
| `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` | `0,1` | select the two R9700s |
| `HIP_ARCHITECTURES`/`AMDGPU_TARGETS`/etc. | `gfx1201` | target the R9700 ISA |

The `VLLM_ROCM_USE_AITER_*` flags in `env/aiter-unified-attention.env` enable
only AITER's unified attention; MoE/linear/RMSNorm stay on stock vLLM kernels
(AITER's MoE/FP8 backends don't support `gfx1201` yet).

### Tuning tools

- `tools/tune_fused_moe.py` — sweeps Triton tile configs for the stock vLLM
  `fused_experts` kernel and writes the per-token-count optimum to
  `fused_moe_configs/`. Flags: `--M 1,2,...`, `--E`, `--N`, `--K`, `--topk`,
  `--reps`, `--out`, `--no-sweep`. The checked-in JSONs cover E=256/N=256 and
  E=256/N=512 at `block_shape=[128,128]`, `fp8_w8a8`.
- `tools/tune_w8a8_fp8.py` — sweeps dense W8A8 block-scaled MM tile configs.
  Experimental; see the dead end below (defaults are already optimal).

Key tuning decisions:
- **MTP4 speculative decoding**: 4 draft tokens per step (~72% acceptance on
  35B-A3B), roughly doubles decode throughput vs no MTP.
- **`GPU_MAX_HW_QUEUES=1`** is required. Multiple queues cause a 55-63% decode
  throughput regression on RDNA4 — one queue per process avoids kernel launch
  scheduling overhead.
- **NCCL channels pinned to 4** (`NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`):
  the bandwidth sweet spot for two GPUs on separate PCIe 5.0 x8 root ports with
  P2P disabled.
- **bf16 KV cache**: requires patching AITER's `TILE_SIZE` from 64→32 to stay
  within the R9700's 64 KiB LDS limit. ~2× KV memory cost vs fp8, zero perf
  regression.
- **`--max-num-batched-tokens 4096`** is required for the MoE model (its
  gated-delta layers force an attention block size of 2112 tokens).
- **Tuned MoE kernel configs** (`fused_moe_configs/`): per-token-count optimal
  Triton tile sizes for the stock vLLM `fused_experts` kernel (not AITER MoE,
  which doesn't support gfx1201). Deployed via
  `VLLM_TUNED_CONFIG_FOLDER`. Provides +5-11% throughput.

## Dead ends

- **W8A8 dense-linear tuning**: only 4 tile shapes fit the 64 KiB LDS limit,
  defaults already optimal — no gain. The MoE kernel succeeded because its
  per-expert N=256/512 allows 24 viable shapes vs 4 for dense.
- **AITER MoE/FP8 backend on gfx1201**: vLLM aborts at startup. Enable once
  upstream AITER adds RDNA4 support.
- **`--enable-expert-parallel` on top of `-tp 2`**: regresses decode ~7-12% on
  the 35B-A3B (tg32 160-175 vs ~181-191, tg128 135-137 vs ~146) with flat
  prefill. EP's AllToAll doesn't pay off for a 3B-active MoE at tp=2, and the
  tuned `fused_moe_configs` (TP layout) no longer apply. Skip at this scale;
  revisit only for much larger active-parameter MoEs.

## Performance

Measured on 2× R9700, MTP4, bf16 KV, single request, vLLM 0.27.0rc2.

| model                     | pp2048 t/s | tg32 t/s | tg128 t/s |
|:--------------------------|-----------:|---------:|----------:|
| Qwen3.6-27B (andy baseline) |     2750 |    81.9 |    —     |
| Qwen3.6-35B-A3B (no MTP)   |   ~10075 |     ~83 |    —     |
| Qwen3.6-27B-FP8 (v0.26)     |    ~2927 |     ~75 |    ~66   |
| Qwen3.6-27B-FP8 (v0.27)     |    ~2916 |     ~87 |    ~76   |
| Qwen3.6-35B-A3B-FP8 (v0.26) | ~10864 |    ~182 |   ~144   |
| Qwen3.6-35B-A3B-FP8 (v0.27) | ~11143 |    ~189 |   ~151   |

Full methodology, depth sweeps, and tuning history in
[`BENCHMARKS.md`](BENCHMARKS.md).

### Long-context concurrency

Decode cost is dominated by attending over the cached KV, so concurrent
deep-context requests degrade sharply. Measured on 35B-A3B (tg32, MTP4);
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
