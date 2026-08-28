# vLLM on Radeon AI PRO R9700

Build and run vLLM from source for AMD Radeon AI PRO R9700 GPUs. The default
configuration targets four R9700s (`gfx1201`) and serves a model through
vLLM's OpenAI-compatible API, with an optional second, concurrent container
for a vision-language profile (see [Multiple containers](#multiple-containers)).

## Requirements

- Docker with the Compose plugin (`docker compose`), or Podman (`podman
  compose`); `just` recipes default to Docker
- [`just`](https://just.systems/)
- [`git`](https://git-scm.com/) (to fetch the source)
- SELinux hosts need no special relabeling: bind mounts mount unlabeled because
  the container runs with `label=disable`
- One or more R9700 GPUs; the included configuration assumes four

## Quick start

Get the source (skip if you already have the repo checked out):

```sh
# Install git if you don't have it (Debian/Ubuntu: apt install git,
# Fedora: dnf install git, Arch: pacman -S git)
git clone https://github.com/prcoe1/r9700-serving.git
cd r9700-serving
```

(`git clone` downloads a full copy of the repository, including its version
history, into a new `r9700-serving/` directory; `cd` then moves into it. All
commands below run from inside that directory.)

```sh
cp .env.example .env  # Build version pins + default model profile (untracked)
just build       # Build localhost/vllm-fullbuild:latest
just check       # Validate the compose config for the selected profiles
just up          # Start both containers in the background (default: Qwen3.8-27B-FP8 + Qwen2.5-VL-72B vision)
just --set model qwen3.6-27b up     # Switch the main container to Qwen3.6-27B-FP8 (dense)
just --set model qwen3.6-35b-a3b up  # Switch the main container to the MoE 35B-A3B model
just --set vision_model qwen2.5-vl-72b-instruct up  # Switch the vision container's profile
just logs        # Follow service logs (`just logs vllm-vision` for one)
just down        # Stop and remove containers
```

To use Podman: `just --set runtime podman build` or `RUNTIME=podman just up`.
Run `just --list` to see all recipes including `rebuild` (force-rebuild) and
`clear-vllm-caches` (wipe host-side Triton/Inductor/AITER caches; preserves
HuggingFace model cache).

Always go through `just`: each service's `env/<profile>.env` (passed to
compose via `--env-file`) feeds the container's environment, and
`entrypoint.sh` assembles the `vllm serve` arguments from it at startup. A
bare `docker compose up` fails with a required-variable error
(`MODEL_PROFILE`/`VISION_MODEL_PROFILE` unset) rather than starting servers
with no model; a container whose profile vars are missing aborts with a
message instead of serving.

The vLLM OpenAI-compatible API is available at `http://localhost:8000/v1`
(main container; `PORT` in `.env` to move it). Other containers on the same
compose network can reach it via the `llm-backend` network alias instead of
the host port. The vision container serves at `http://localhost:8001/v1`
(`VISION_PORT`) / `llm-vision-backend` alias.

## Configuration

Build versions are pinned in `.env` (untracked; copy `.env.example` to create it).
Host-specific settings also live there: `USER_UID`/`USER_GID` (the user the
container runs as, keeping host cache dirs user-owned) and `RENDER_GID` (the
host render group gid for `/dev/dri` access — check with `getent group render`).

| component    | version |
|:-------------|:--------|
| ROCm         | 7.14.0 (`rocm/dev-ubuntu-24.04:7.14.0-full`) |
| PyTorch      | 2.13.0+rocm7.14.0 |
| vLLM         | 0.28.0rc2 |
| AITER        | v0.1.20 |
| Flash Attention | @ 1cc7ff67 |

ROCm 7.14 is on AMDs "TheRock" technology-preview stream (7.9/7.13/7.14); the
production 7.2.x line lacks RDNA4/`gfx1201` support. AITER `v0.1.20` is the
latest tagged release; vLLM is the 0.28.0rc2 prerelease (latest stable is
0.27.1) since `gfx1201` requires source builds.

The default (active) model is `Qwen/Qwen3.8-27B-FP8` (`qwen3.8-27b`, the
newest dense 27B hybrid linear/full-attention architecture, MTP trained,
vision). Alternatives: `Qwen/Qwen3.6-27B-FP8` (`qwen3.6-27b`, dense) and
`Qwen/Qwen3.6-35B-A3B-FP8` (`qwen3.6-35b-a3b`, 35B total / 3B active MoE).
Model selection is controlled by `MODEL_PROFILE` in `.env` — override inline
with `MODEL_PROFILE=qwen3.6-27b just up`.

Runtime environment is split across files:
- `env/2xr9700.vllm.common` — ROCm config for all four visible GPUs (arch, NCCL, HSA)
- `env/aiter-unified-attention.env` — enables AITER unified attention only
- `env/qwen3.6.env.common` — shared qwen3.6/3.8 config (KV cache dtype, MTP spec-decode, tool choice)
- `env/qwen3.6-35b-a3b.env` — MoE model config (path, tokenizer, MTP disabled)
- `env/qwen3.6-27b.env` — dense 27B model config
- `env/qwen3.8-27b.env` — Qwen3.8-27B-FP8 dense model config (same
  architecture as 3.6-27B, so it shares the 3.6 common settings and tuned
  per-shape fp8 GEMM configs)
- `env/qwen2.5-vl-72b-instruct.env` — Qwen2.5-VL-72B-Instruct-AWQ vision
  profile for the `vllm-vision` container

### Multiple containers

`compose.yaml` defines a second service, `vllm-vision`: the same image and
`entrypoint.sh`, but fed by `env/${VISION_MODEL_PROFILE}.env` (default
`qwen2.5-vl-72b-instruct`), on host port `${VISION_PORT:-8001}` (main uses
`${PORT:-8000}`), with its own triton/torchinductor/tilelang cache dirs
(`~/.cache/{triton,torchinductor,tilelang}_vision`) so the two containers
never share mutable compile-cache state. The aiter JIT dir is shared on
purpose: `just prewarm` builds the kernels once in the main service's env and
both containers then only read them (concurrent *builds* are the hazard).

`just up` starts both services; it waits for and warms up the main one while
the (much larger) vision model keeps loading — check it with
`just logs vllm-vision`. Select profiles with `MODEL_PROFILE` /
`VISION_MODEL_PROFILE` (or `just --set model ...` / `just --set vision_model
...`). Both containers see all four GPUs
(`env/2xr9700.vllm.common`); when running them at the same time, give them
non-overlapping GPU sets at runtime (e.g. `HIP_VISIBLE_DEVICES=0,1` vs `2,3`)
and match `VLLM_TP` to the set size.

### Chat template

All profiles mount [froggeric's Qwen-Fixed-Chat-Templates]
(`chat-templates/qwen.jinja`, pinned to **v22.3** — `qwen3.8-froggeric-v22.3`,
fetched from the repo's `main`). The `vllm` service sets
`VLLM_CHAT_TEMPLATE` to that file, so `entrypoint.sh` applies it to the
qwen3.x models, overriding each model's bundled template. The `vllm-vision`
service leaves `VLLM_CHAT_TEMPLATE` unset and uses the tokenizer's built-in
template (the froggeric template targets the qwen3.x models).
The template fixes rendering bugs, KV-cache invalidation, token waste, and
agentic stalling in the official Qwen templates, and adds tool-error retry
warnings plus optional `tool_call_format="json"` / reasoning-effort steering
(`reasoning_effort`) kwargs. Thinking is partitioned into the `reasoning` field
and the final answer into `content`; `--reasoning-parser qwen3` (see below) is
required for the split to work.

Since **v22.3**, history re-rendering is byte-identical to generated tokens for
thinking-off turns (past ` thinking` blocks are always preserved, even when
empty), which keeps `--enable-prefix-caching` hits intact across multi-turn
conversations — this stack runs thinking-off by default.

Refresh the overlay from upstream when a newer version ships (compare the
`template_version` line of `chat-templates/qwen.jinja` against the repo's
`main`):

```sh
curl -L -o chat-templates/qwen.jinja \
  https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja
```

Template bumps need no image rebuild: after refreshing, bump the pin note
above and `just down && just up` (the in-memory prefix cache is cleared on
restart anyway).

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
- **`--enable-prompt-tokens-details`**: include the `prompt_tokens_details`
  breakdown (cached/text/image/audio/video) in API responses so front-ends like
  LiteLLM can attribute cache hits for accurate cost tracking.
- **`--max-model-len`** (default `131072`; the Qwen3.8-27B profile overrides to
  `262144` for 256K contexts), **`-tp 2`**,
  **`--gpu-memory-utilization 0.95`**. **`--max-num-seqs`** defaults to `2`
  (`entrypoint.sh`), and `qwen3.6.env.common` keeps it `2` explicitly on all
  Qwen profiles to stay below the #35288 corruption threshold (see "MTP
  concurrency bug").
 - **`--kv-cache-dtype bfloat16`** (`VLLM_KV_CACHE_DTYPE`, default `bfloat16`).
   bf16 KV is now the default across all Qwen profiles. The AITER BF16 LDS-fit
   patch (`patches/aiter/unified-attention-bf16-kv.patch`), which caps
   `TILE_SIZE` and `attn_stages` to fit 64 KiB LDS, is required for it. Prior
   "garbage" output was caused by MTP token loops (see
   [`archive/DEADENDS.md`](archive/DEADENDS.md)), not the patch. To opt back into
   fp8 KV, set `VLLM_KV_CACHE_DTYPE=fp8` (halves KV-cache memory, useful when
   context length is the constraint).
 - **fp8 KV is the opt-in path; when used, the calibrated copy matters.** The
   stock `Qwen3.8-27B-FP8`/`Qwen3.6-27B-FP8` checkpoints ship no
   `k_scale`/`v_scale`/`q_scale`, so vLLM ≥0.28 (which removed
   `--calculate-kv-scales`) would otherwise serve fp8 KV at **scale 1.0** (boot
   log: `Using KV cache scaling factor 1.0 for fp8_e4m3`). Scale 1.0 is
   miscalibrated — a calibration run records deep-layer V amax up to **~130-132**
   (layer 63) vs the ~1-24 range scale 1.0 assumes, wasting the e4m3 dynamic
   range. The calibrated copy (`~/models-local/<model>-kvscales`, built by
   `just up` -> `ensure-kvscales` via `tools/setup_kvscales.py` +
   `tools/calibrate_kv_scales.py`) covers the main full-attention layers **and
   the MTP prediction-head layer(s)** (`mtp.layers.*`, which caches fp8 KV
   separately). Measured effect: calibrated vs scale-1.0 diverge ~20-27% (scale
   1.0 corrupts KV numerics), no throughput regression. Recalibrate with
   `just clear-kvscales` then `just up`. Note the 2026-08-22 quality A/B
   (`benchmarks/2026-08-22_kv_calibration_quality_ab.md`) found calibrated vs
   scale-1.0 **indistinguishable** on both PPL and long-context recall, so
   calibration is a correctness fix, not a measured quality win — hence bf16 is
   the safer default.
- **`--attention-backend ROCM_AITER_UNIFIED_ATTN`** + `--speculative-config`
  (MTP4 on Qwen3.6-27B, **MTP3** on Qwen3.8-27B, disabled on 35B-A3B — see
  [`archive/DEADENDS.md`](archive/DEADENDS.md)).
- **MTP3 is the Qwen3.8-27B default.** DFlash2 was tried and rejected: a clean
  2026-08-22 depth A/B on the same calibrated build showed DFlash2's decode win
  is short-context only (MTP3 holds ~50-60 t/s out to d64K and ~37@d128K vs
  DFlash2 37/24/15, with far lower variance). See
  [`archive/DEADENDS.md`](archive/DEADENDS.md) and
  [`benchmarks/2026-08-22_qwen3.8-27b_fp8kv_depth_mtp3_dflash.md`](benchmarks/2026-08-22_qwen3.8-27b_fp8kv_depth_mtp3_dflash.md).
- **PCIe P2P must stay disabled on this host.** `NCCL_P2P_DISABLE=1` is
  required: the two R9700s sit on separate PCIe root ports, and enabling P2P
  (`NCCL_P2P_DISABLE=0`, even with `HSA_ENABLE_IPC_MODE_LEGACY=0`) collapses
  DFlash decode from ~92 t/s to ~9 t/s (10× regression) despite RCCL
  establishing P2P channels. `HSA_ENABLE_IPC_MODE_LEGACY` is irrelevant once
  P2P is off. This also rules out the P2P all-reduce HIP kernels on this box.

### Runtime overlays (bind-mounted source fixes)

Version-locked patches applied at runtime by read-only bind-mounts in
`compose.yaml` (no image rebuild needed). Refresh the overlay files when bumping
the pinned dependency.

- **Tolerate empty `tools` arrays** (`patches/vllm/protocol.py`, pinned to
  `VLLM_REF` v0.28.0rc2): some clients send `{"tools": [], "tool_choice": "none"}`
  on chat completions, which upstream vLLM's `check_tool_usage` rejects with a
  400 (it treats any empty tools array as malformed, even when `tool_choice` is
  `"none"`). The overlay of
  `vllm/entrypoints/openai/chat_completion/protocol.py` treats `tools: []` as a
  no-tools request when `tool_choice` is `"none"`/omitted, while still rejecting
  genuinely invalid combos (`auto`/`required`/named tool_choice with empty
  tools).

### Source-build patches (applied at image build time)

Local backport of an upstream fix not in `VLLM_REF` v0.28.0rc2. Applied by
`Dockerfile.fullbuild` from `patches/vllm/*.patch` (mirrors the aiter patch
loop). Verified to apply cleanly on the v0.28.0rc2 tree; re-verify when
bumping `VLLM_REF`.

(#51812 GDN spec-gate alignment and #51837 ROCm KV-first page separation were
carried as patches on v0.27.1; both merged upstream 2026-08-11 and are in
v0.28.0rc2, so their patches were dropped with the bump.)

- **Honor `drop_eagle_block` in `MambaManager`**
  (`patches/vllm/48375-mamba-drop-eagle-block.patch`,
  [#48375](https://github.com/vllm-project/vllm/pull/48375), adapted for
  v0.28.0rc2): `MambaManager.find_longest_cache_hit` accepted `drop_eagle_block`
  and ignored it, so on hybrid GDN models with MTP/EAGLE + prefix caching the
  final matched page of a cache hit could hold recurrent state written over
  draft positions that verification later rejects — silent corruption that
  cache hits then spread to every later request sharing the prefix (#43559,
  #50188). Fix lowers the search ceiling by one page (a literal pop would
  delete Mamba's rightmost real state block). Still open upstream; carried as
  a local patch.

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
| `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` | `0,1,2,3` | all four R9700s visible to every container; partition at runtime when running both |
| `HIP_ARCHITECTURES`/`AMDGPU_TARGETS`/etc. | `gfx1201` | target the R9700 ISA |

The `VLLM_ROCM_USE_AITER_*` flags in `env/aiter-unified-attention.env` enable
only AITER's unified attention; MoE/linear/RMSNorm stay on stock vLLM kernels
(AITER's MoE/FP8 backends don't support `gfx1201` yet).

Key tuning decisions:
- **MTP speculative decoding** (dense profiles): MTP4 on Qwen3.6-27B (~72%
  acceptance, ~doubles decode). Qwen3.8-27B peaks at **MTP3** (tg32 72.1 on the
  current fp8-KV stack; bf16-KV sweep: MTP3 57.6, MTP2 56.0, MTP1 45.6, MTP4
  49.2, no-MTP 32.0) because its MTP head accepts drafts poorly past position 3,
  so more drafts waste compute and fewer lose throughput. MTP is **disabled on
  35B-A3B** (pending a re-test after the upstream #47087 fix — see
  [`archive/DEADENDS.md`](archive/DEADENDS.md)).
- **`GPU_MAX_HW_QUEUES=1`** is required. Multiple queues cause a 55-63% decode
  throughput regression on RDNA4 — one queue per process avoids kernel launch
  scheduling overhead.
- **NCCL channels pinned to 4** (`NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`):
  the bandwidth sweet spot for two GPUs on separate PCIe 5.0 x8 root ports with
  P2P disabled.
- **bf16 KV cache** (current default, `VLLM_KV_CACHE_DTYPE=bfloat16`): higher
  KV fidelity than fp8. It uses more K/V bytes than fp8, so fp8 (with the
  calibrated-copy scale fix, halving KV memory) remains the option when context
  length is the binding constraint. The 2026-08-22 quality A/B found fp8
  (calibrated or scale-1.0) indistinguishable from bf16 on PPL and long-context
  recall, so bf16's extra fidelity costs nothing measurable and it is the
  safer default; the prior "garbage" output was MTP token loops, not the KV
  dtype.
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

### MTP concurrency bug (dense profiles)

The one upstream bug currently affecting this stack:
[#35288](https://github.com/vllm-project/vllm/issues/35288) — MTP spec-decode
produces corrupted output when 4+ decode sequences share a batch (garbage
header → repetition loop → `max_tokens`).

**Workaround**: `env/qwen3.6.env.common` sets `VLLM_MAX_NUM_SEQS=2`, so vLLM
never forms a ≥4-sequence decode batch — concurrent decode stays below the
corruption threshold regardless of incoming concurrency. Verified with the
#35288 repro (4/6/8 concurrent requests → all coherent) and the 400-request
stress test. Dense MTP stays enabled (MTP4 on Qwen3.6-27B, MTP3 on
Qwen3.8-27B); 35B-A3B runs MTP disabled.

### Upstream issues

The live upstream-issue watchlist, the known-to-ignore list, and the update-check
workflow (pin bumps, patch re-verification, triage filters) are maintained in
`AGENTS.md` ("Checking for Updates"). Resolved/superseded issues, dead ends, and
stale triage snapshots live in
[`archive/DEADENDS.md`](archive/DEADENDS.md).

## Performance

Measured on 2× R9700 (gfx1201), single request, thinking off, vLLM 0.28.0rc2 +
the local patch under "Source-build patches" (#48375), torch 2.13, triton 3.8.0
(ROCm 7.14.0), tuned MoE/dense GEMM configs. The Qwen3.8-27B row is the
current default stack (**MTP3**, 256K context, **bf16 KV**; measured 2026-08-22
after the bf16 default switch). MTP3 was
made the default over DFlash2 after the 2026-08-22 depth A/B: MTP3 holds decode
far better at depth (tg32 60@d32K, 52@d64K, 37@d128K vs DFlash 37/24/15) at the
cost of short-context decode (MTP3 tg32 ~53-60 vs DFlash ~88 @d0). The other rows are the
latest available measurements for those profiles (2026-08-12, pre-patch
build).
Full methodology, per-run files, and upgrade history in
[`BENCHMARKS.md`](BENCHMARKS.md) and [`archive/`](archive/).

| model                     | MTP (draft #) | KV   | pp2048 t/s | tg32 t/s | tg128 t/s |
|:--------------------------|:--------------|:-----|-----------:|---------:|----------:|
| Qwen3.8-27B-FP8 (default, 2026-08-22) | **MTP3** | bf16 |    2628 |   ~57 |    ~65 |
| Qwen3.6-27B-FP8 (2026-08-12)²         | MTP4 | bf16 |   ~2500 |   **90.8** |    ~69 |
| Qwen3.6-35B-A3B-FP8 (2026-08-12)      | off  | bf16 |   ~8510 |   **91.0** |   **91.3** |

### Depth sweep (Qwen3.8-27B-FP8, 2026-08-22, current default: bf16 KV)

Current default stack: bf16 KV + **MTP3** + 256K max-model-len, full-context
prefill at depth:

| depth | pp2048 (t/s) | tg32 (t/s) | e2e TTFT (s) |
|------:|-------------:|-----------:|-------------:|
| 4096  |     2708.14 |     57.49 |          2.27 |
| 8192  |     2654.98 |     54.53 |          3.86 |
| 16384 |     2531.50 |     64.71 |          7.28 |
| 32768 |     2290.63 |     59.00 |         15.20 |
| 65536 |     1909.03 |     51.50 |         35.40 |
| 128000|     1445.16 |     61.12 |         89.99 |
| 200000|     1119.89 |     50.46 |        180.42 |
| 256000|      953.17 |     44.97 |        270.73 |

bf16 KV doubles the K/V bytes per attention step, so **deep prefill/TTFT is
slower than the fp8 sweep** (pp256K 953 vs 1563 t/s, ~39% down; TTFT 271 vs
165 s at d256K). Decode holds 51–65 t/s out to d200K and 45 t/s at d256K
(recorded here higher than the old fp8 figures — see the cross-build caveat in
the depth doc). Coherence passed at every depth; MTP acceptance ~33% unchanged.
Full tables and the fp8-vs-bf16 comparison in
[`benchmarks/2026-08-22_qwen3.8-27b_bf16kv_depth_mtp3.md`](benchmarks/2026-08-22_qwen3.8-27b_bf16kv_depth_mtp3.md);
the earlier fp8-KV depth sweeps are in
[`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md)
and [`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_d0.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_d0.md).

### 35B-A3B depth sweep and concurrency (archived)

The 35B-A3B depth sweep (tuned dense vs stock) and the long-context
concurrency head-to-head (serial/c1 wins; the `--max-num-seqs` 2 cap exists
for the #35288 MTP bug, not throughput) are archived in
[`archive/BENCHMARKS.md`](archive/BENCHMARKS.md), with per-run tables in
[`archive/benchmarks/`](archive/benchmarks/).

## Stability tests

Smoke tests, sustained-load stress, and long-context generation checks for
catching crashes, memory errors, and token-loop degeneration.

See [`benchmarks/STABILITY_TESTS.md`](benchmarks/STABILITY_TESTS.md) for scripts
and baseline results. Quick health check:

```sh
just check && curl -sf http://localhost:8000/health && echo "OK"
```
