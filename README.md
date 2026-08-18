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
just up          # Start vLLM in the background (default: Qwen3.8-27B-FP8)
just --set model qwen3.6-27b up     # Switch to Qwen3.6-27B-FP8 (dense)
just --set model qwen3.6-35b-a3b up  # Switch to MoE 35B-A3B model
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
Host-specific settings also live there: `USER_UID`/`USER_GID` (the user the
container runs as, keeping host cache dirs user-owned) and `RENDER_GID` (the
host render group gid for `/dev/dri` access — check with `getent group render`).

| component    | version |
|:-------------|:--------|
| ROCm         | 7.14.0 (`rocm/dev-ubuntu-24.04:7.14.0-full`) |
| PyTorch      | 2.13.0+rocm7.14.0 |
| vLLM         | 0.27.1 |
| AITER        | v0.1.20 |
| Flash Attention | @ 1cc7ff67 |

ROCm 7.14 is on AMDs "TheRock" technology-preview stream (7.9/7.13/7.14); the
production 7.2.x line lacks RDNA4/`gfx1201` support. AITER `v0.1.20` is the
latest tagged release; vLLM is the 0.27.1 release since `gfx1201`
requires source builds.

The default (active) model is `Qwen/Qwen3.8-27B-FP8` (`qwen3.8-27b`, the
newest dense 27B hybrid linear/full-attention architecture, MTP trained,
vision). Alternatives: `Qwen/Qwen3.6-27B-FP8` (`qwen3.6-27b`, dense) and
`Qwen/Qwen3.6-35B-A3B-FP8` (`qwen3.6-35b-a3b`, 35B total / 3B active MoE).
Model selection is controlled by `MODEL_PROFILE` in `.env` — override inline
with `MODEL_PROFILE=qwen3.6-27b just up`.

Runtime environment is split across files:
- `env/2xr9700.vllm.common` — two-GPU ROCm config (arch, NCCL, HSA, compile caches)
- `env/aiter-unified-attention.env` — enables AITER unified attention only
- `env/qwen3.6.env.common` — shared qwen3.6/3.8 config (KV cache dtype, MTP spec-decode, tool choice)
- `env/qwen3.6-35b-a3b.env` — MoE model config (path, tokenizer, MTP disabled)
- `env/qwen3.6-27b.env` — dense 27B model config
- `env/qwen3.8-27b.env` — Qwen3.8-27B-FP8 dense model config (same
  architecture as 3.6-27B, so it shares the 3.6 common settings and tuned
  per-shape fp8 GEMM configs)

### Chat template

All profiles mount and use [froggeric's Qwen-Fixed-Chat-Templates]
(`chat-templates/qwen.jinja`, pinned to **v22.1** — `qwen3.8-froggeric-v22.1`,
fetched from the repo's `main`). It is applied to every model via
`--chat-template` in `compose.yaml`, overriding each model's bundled template.
The template fixes rendering bugs, KV-cache invalidation, token waste, and
agentic stalling in the official Qwen templates, and adds tool-error retry
warnings plus optional `tool_call_format="json"` / reasoning-effort steering
(`reasoning_effort`) kwargs. Thinking is partitioned into the `reasoning` field
and the final answer into `content`; `--reasoning-parser qwen3` (see below) is
required for the split to work.

Refresh the overlay from upstream when a newer version ships:

```sh
curl -L -o chat-templates/qwen.jinja \
  https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja
```

[froggeric's Qwen-Fixed-Chat-Templates]: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates

### Non-standard vLLM flags

- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`** (`VLLM_TOOL_CHOICE`, all profiles): OpenAI
  tool-calling with Qwen's `qwen3_coder` parser. `--reasoning-parser qwen3` is
  required for the Qwen chat template to correctly split thinking
  into the `reasoning` field.
- **`--limit-mm-per-prompt '{"image": 99, "audio": 0, "video": 0}'`**: multimodal
  images allowed, audio/video disabled.
- **`--override-generation-config`**: server-side sampling defaults
  (`temperature` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0, no penalties).
- **`--enable-prefix-caching`**: reuse KV for shared prompt prefixes.
- **`--max-model-len`** (default `131072`; the Qwen3.8-27B profile overrides to
  `262144` for 256K contexts), **`-tp 2`**,
  **`--gpu-memory-utilization 0.85`**. **`--max-num-seqs`** defaults to `4`
  (`compose.yaml`), but `qwen3.6.env.common` caps it to `2` on the MTP profiles
  to stay below the #35288 corruption threshold (see "MTP bug").
- **`--kv-cache-dtype fp8`** (`VLLM_KV_CACHE_DTYPE`, default `fp8`). fp8 KV is
  the default across all Qwen profiles — it halves KV-cache memory (enabling
  the 256K context on Qwen3.8-27B) and its smaller K/V bytes keep
  deep-context decode up (see depth sweep below). For BF16 KV, set
  `VLLM_KV_CACHE_DTYPE=bfloat16` plus the AITER BF16 LDS-fit patch
  (`patches/aiter/unified-attention-bf16-kv.patch`), which caps `TILE_SIZE`
  and `attn_stages` to fit 64 KiB LDS. Prior "garbage" output was caused by
  MTP token loops (see "MTP bug" below), not the patch.
- **`--attention-backend ROCM_AITER_UNIFIED_ATTN`** + `--speculative-config`
  (MTP4 on the dense profiles; disabled on 35B, see "MTP workaround").

### Runtime overlays (bind-mounted source fixes)

Version-locked patches applied at runtime by read-only bind-mounts in
`compose.yaml` (no image rebuild needed). Refresh the overlay files when bumping
the pinned dependency.

- **Tolerate empty `tools` arrays** (`patches/vllm/protocol.py`, pinned to
  `VLLM_REF` v0.27.1): some clients send `{"tools": [], "tool_choice": "none"}`
  on chat completions, which upstream vLLM's `check_tool_usage` rejects with a
  400 (it treats any empty tools array as malformed, even when `tool_choice` is
  `"none"`). The overlay of
  `vllm/entrypoints/openai/chat_completion/protocol.py` treats `tools: []` as a
  no-tools request when `tool_choice` is `"none"`/omitted, while still rejecting
  genuinely invalid combos (`auto`/`required`/named tool_choice with empty
  tools).

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
| `HOME` | `$HOME` (compose `user:`) | container runs as host user; whole home mounted, caches redirected under `~/.cache` (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `AITER_JIT_DIR`, `TILELANG_CACHE_DIR`) |
| `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` | `0,1` | select the two R9700s |
| `HIP_ARCHITECTURES`/`AMDGPU_TARGETS`/etc. | `gfx1201` | target the R9700 ISA |

The `VLLM_ROCM_USE_AITER_*` flags in `env/aiter-unified-attention.env` enable
only AITER's unified attention; MoE/linear/RMSNorm stay on stock vLLM kernels
(AITER's MoE/FP8 backends don't support `gfx1201` yet).

Key tuning decisions:
- **MTP speculative decoding** (dense profiles): MTP4 on Qwen3.6-27B (~72%
  acceptance, ~doubles decode). Qwen3.8-27B peaks at **MTP3** (tg32 57.6; MTP2
  56.0, MTP1 45.6, MTP4 49.2, no-MTP 32.0) because its MTP head accepts drafts
  poorly past position 3, so more drafts waste compute and fewer lose
  throughput. MTP is **disabled on 35B-A3B** — see "MTP bug" below.
- **`GPU_MAX_HW_QUEUES=1`** is required. Multiple queues cause a 55-63% decode
  throughput regression on RDNA4 — one queue per process avoids kernel launch
  scheduling overhead.
- **NCCL channels pinned to 4** (`NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`):
  the bandwidth sweet spot for two GPUs on separate PCIe 5.0 x8 root ports with
  P2P disabled.
- **fp8 KV cache** (current default, `VLLM_KV_CACHE_DTYPE=fp8`): halves
  KV-cache memory (enables 256K contexts on Qwen3.8-27B) and keeps
  deep-context decode up, since decode at depth is bound by the K/V bytes
  moved per attention step (see depth sweep below). BF16 KV (via the AITER
  LDS-fit patch) outperformed fp8 on 35B-A3B at shallow context (~88 t/s) and
  remains the option when context length is not the constraint; the prior
  "garbage" output was MTP token loops, not the KV dtype.
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

### MTP bug (35B) and MTP concurrency bug (dense)

vLLM's native MTP speculative decoding has two confirmed bugs that affect this
stack:

| Issue | Summary | Scope |
|:------|:--------|:------|
| [vllm-project/vllm#47087](https://github.com/vllm-project/vllm/issues/47087) | MTP token loops on Qwen3-MoE; output quality collapse mid-conversation | 35B-A3B |
| [vllm-project/vllm#35288](https://github.com/vllm-project/vllm/issues/35288) | MTP produces corrupted output when 4+ decode sequences share a batch (garbage header → repetition loop → `max_tokens`) | dense 27B MTP profiles |

**35B**: MTP disabled. 35B-A3B throughput drops from ~185 tg32 (MTP4) to ~83
tg32 (no MTP).

**Dense 27B**: MTP works correctly at c1–3, but vLLM never forms a ≥4-sequence
decode batch: `env/qwen3.6.env.common` sets `VLLM_MAX_NUM_SEQS=2` (see "MTP
workaround" below), so concurrent decode batches stay below the corruption
threshold regardless of incoming concurrency. Verified with the #35288 repro
(4/6/8 concurrent requests → all coherent) and the 400-request stress test.

**Workaround**: the 35B profile sets `VLLM_SPEC_DECODE=` (empty), skipping
`--speculative-config`. The dense profiles override `VLLM_SPEC_DECODE` from
`qwen3.6.env.common`: `qwen3.6-27b` inherits MTP4, `qwen3.8-27b` overrides to
MTP3 (its MTP head accepts drafts poorly past position 3 — see "Key tuning
decisions"). Because #35288 corrupts output when 4+ decode sequences share a
batch, `qwen3.6.env.common` also sets `VLLM_MAX_NUM_SEQS=2`, capping concurrent
sequences below the trigger for the MTP profiles.

### aiter op-namespace teardown crash

Upstream aiter's `torch_compile_guard` registers custom ops by passing a
namespace-prefixed name to `torch.library.Library("aiter")`:

```python
op_schema = f"aiter::{loadName}" + schema
aiter_lib.define(op_schema, tags=tags)
aiter_lib.impl(f"aiter::{loadName}", ...)
```

torch's `Library` wrapper already prepends its namespace, so its bookkeeping
records the qualname as `aiter::aiter::{opname}` (double-namespaced). At
interpreter exit torch's `_clear_torch_ops_cache` does
`qualname.split("::")` expecting exactly two parts and crashes with
`ValueError: too many values to unpack`. This is an aiter bug (incorrect torch
`Library.define` usage — vLLM itself registers ops with bare names, e.g.
`vllm/utils/torch_utils.py:936`); the runtime symptom is benign noise during
interpreter shutdown in the throwaway prewarm container.

**Upstream status**: fixed in [ROCm/aiter PR #4593](https://github.com/ROCm/aiter/pull/4593)
("fix(torch_guard): drop redundant aiter:: prefix from define schema for torch
2.13"), merged into `main` as commit `86cc388` on 2026-08-06. The upstream
commit independently confirms this diagnosis: torch ≥2.13 stores the doubled
`aiter::aiter::<op>` in `_op_defs` (2.12 stripped it), and the shutdown
`qualname.split("::")` raises `too many values to unpack`. The upstream patch
matches our former overlay byte-for-byte (bare op name to `define`/`impl`).
**Shipped in `AITER_REF` v0.1.20 (2026-08-18)**; the overlay was removed on the
bump to v0.1.20. If reverting to v0.1.19.x, restore the overlay from git
history and clear caches per the cache-clearing notes.

**Impact**: cosmetic only — a traceback at process exit in the prewarm and any
short-lived aiter process. No effect on the running server.

**Workaround (removed)**: the overlay `patches/aiter/torch_guard.py` passed the
bare op name to `define`/`impl`, which `Library("aiter")` namespaces itself;
`torch.ops.aiter.*` still resolves. With `AITER_REF` ≥ v0.1.20 the fix ships
upstream, so no overlay is needed.

## Dead ends

- **AITER MoE/FP8 backend on gfx1201**: vLLM aborts at startup. Enable once
  upstream AITER adds RDNA4 support.
- **`--enable-expert-parallel` on top of `-tp 2`**: regresses decode ~7-12% on
  the 35B-A3B (tg32 160-175 vs ~181-191, tg128 135-137 vs ~146) with flat
  prefill. EP's AllToAll doesn't pay off for a 3B-active MoE at tp=2. Skip at
  this scale; revisit only for much larger active-parameter MoEs.

## Performance

Measured on 2× R9700, tuned MOE/dense configs, single request, vLLM 0.27.0,
torch 2.13, triton 3.8.0+git (ROCm 7.14.0). The 0.27.0 → 0.27.1 bump is a
packaging/pin update; the source-build pin is now 0.27.1 (see Configuration).
KV dtype per row is as labeled; the **current default stack is fp8 KV**
(`VLLM_KV_CACHE_DTYPE=fp8`). MTP4 is enabled for Qwen3.6-27B, MTP3 for
Qwen3.8-27B; disabled for 35B-A3B (see "MTP bug" above and "Key tuning
decisions"). The Qwen3.8-27B row below was measured with BF16 KV + tuned
dense, MTP3, froggeric chat template, thinking off.

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
| Qwen3.8-27B-FP8 (bf16 KV, MTP3, tuned dense) | MTP3, bf16 KV | ~2498–2547 | **57.6** | 52.1 |

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

### Depth sweep (Qwen3.8-27B-FP8, fp8 KV, MTP3, 2026-08-18)

Current default stack (fp8 KV + MTP3 + 256K max-model-len), full-context
prefill at depth:

| depth | ctx_pp t/s | tg32 t/s | e2e TTFT (s) |
|------:|-----------:|---------:|-------------:|
| 4096  |     2698 |     48.7 |           2.3 |
| 8192  |     2754 |     51.3 |           3.7 |
| 16384 |     2699 |     51.3 |           6.8 |
| 32768 |     2588 |     46.5 |          13.5 |
| 65536 |     2364 |     59.8 |          28.6 |
| 128000|     2032 |     40.3 |          64.0 |

Decode is flat ~47–51 t/s to d16K and holds up against the 57.6 t/s d0
baseline, with fp8 KV's halved K/V bytes offsetting the growing attention
span; d128K decodes at 40.3 t/s (−20% vs d8K). Full-context prefill degrades
−25% (2698 → 2032 t/s) and e2e TTFT scales linearly to 64 s at d128K. See
[`benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_depth.md).

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

## Stability tests

Smoke tests, sustained-load stress, and long-context generation checks for
catching crashes, memory errors, and token-loop degeneration.

See [`benchmarks/STABILITY_TESTS.md`](benchmarks/STABILITY_TESTS.md) for scripts
and baseline results. Quick health check:

```sh
just check && curl -sf http://localhost:8180/health && echo "OK"
```
