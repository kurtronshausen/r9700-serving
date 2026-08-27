# AGENTS

## Rules

- **Never commit, push, open PRs, or make any remote changes targeting `andysalerno/r9700-serving` (remote: `upstream`/`andysalerno`) without explicit, direct instructions from the user — and confirm intent before doing so.**
- Always target `origin` (your own fork) for commits, pushes, and PRs unless explicitly told otherwise.

## Tools

Use `just` for all build/run workflows. Commands are defined in `justfile`.

| recipe | purpose |
|:-------|:--------|
| `just check` | validate the compose config for both model profiles (main + flash-next) |
| `just build` | build the main Docker image (`vllm` service only) |
| `just build-flashnext` | build the Qwen3.8-Flash-Next image (Dockerfile.flashnext, `vllm-qwen-flashnext` service) |
| `just rebuild` / `just rebuild-flashnext` | force-rebuild (no cache), main / flashnext image |
| `just up` | start the main `vllm` service (runs `check`, `ensure-cache-dirs`, `prewarm`, waits for readiness, runs warmup) |
| `just up-flashnext` | start the `vllm-qwen-flashnext` service (needs `build-flashnext` first) |
| `just prewarm` | build shared aiter JIT kernels in one throwaway container (runs automatically before every `up`) |
| `just bench` | benchmark the selected model via `llama-benchy` (pp2048, tg32+128) |
| `just tune` | tune Triton GEMM kernel configs for the selected profile (dense W8A8 block-FP8 + fused MoE) in a throwaway container on one GPU; independent of the running `vllm` service, results picked up on next `just up` |
| `just logs [service]` | follow container logs (all services, or one: `just logs vllm-qwen-flashnext`) |
| `just down` | stop and remove the containers |
| `just exec <cmd>` | run a command inside the running container (e.g. `just exec bash`) |
| `just compose <args>` | raw compose passthrough (e.g. `just compose up -d vllm`, `just compose ps`) |
| `just ensure-cache-dirs` | pre-create host cache dirs owned by the current user |
| `just clear-vllm-caches` | wipe compile caches (triton, torchinductor, aiter, etc., incl. the `*_flashnext` dirs) |

There is no `command:` in `compose.yaml`: `entrypoint.sh` (the image
ENTRYPOINT) assembles the `vllm serve` arguments from each service's own
environment at startup — the `env/<profile>.env` stack the recipes pass to
compose via `--env-file` (alongside `.env` for the build pins). This is what
lets `vllm` and `vllm-qwen-flashnext` run different profiles from one compose
file. Bare `docker compose up` fails with a required-variable error
(`MODEL_PROFILE` unset) by design — always go through `just`. `.env` is
untracked; create it with `cp .env.example .env`.

To switch the MAIN model, set `MODEL_PROFILE` or use `--set`:

```
MODEL_PROFILE=qwen3.6-27b just up
just --set model qwen3.6-27b up
```

`vllm-qwen-flashnext` is a second, concurrent container (same entrypoint) on
host port `${FLASHNEXT_PORT:-8001}` (main: `${PORT:-8000}`), fed by
`env/qwen3.8-flash-next.env`, with its own
`~/.cache/{triton,torchinductor,tilelang}_flashnext` dirs. It is built from a
SEPARATE image: `Dockerfile.flashnext` is byte-identical to
`Dockerfile.fullbuild` except the vLLM stage applies
`patches/vllm-flashnext/` (newer vLLM head carrying qwen4_exp + the local PLE
CPU-offload patch) at `FLASHNEXT_VLLM_REF`/`FLASHNEXT_VLLM_VERSION`. The
separate pins mean the two images build/cache independently — switching back
to the qwen3.8-27b stack never rebuilds the flashnext image. `just up` starts
only `vllm`; `just up-flashnext` starts the flashnext service (and its PLE
n-gram table needs ~110 GiB of free host RAM, `VLLM_PLE_CPU_OFFLOAD=1`). The
aiter JIT dir is shared between the two containers (`just prewarm` builds it
once). Both containers see all four GPUs; assign non-overlapping GPU sets at
runtime when running them together (see README "Multiple containers").

## Build Caveats

### Rebuild Timeouts
- `just rebuild` (no-cache Docker build) takes **40-60+ minutes** for a full vLLM
  compilation cycle (framework-base → flash-attention → aiter → vllm → runtime).
- When invoking rebuild via automation, set timeout to **at least 3600s (1h)**;
  budget **4500s (75m)** for safety on first-run or after base-image changes.
- Incremental `just build` (layer-cached) is significantly faster. Only use
  `rebuild` when base images, dependency pins, or source patches change.

### Cache Clearing
- The vLLM container runs as the host user (`compose.yaml` `user:` + `HOME` env),
  and `just up` pre-creates the host cache dirs via `just ensure-cache-dirs`, so
  Docker's daemon never recreates them as root. Cache dirs stay user-owned and
  `just clear-vllm-caches` needs no sudo.
- If the dirs were ever created by an older root-running setup (or by a bare
  `docker compose`), they may be root-owned; `just ensure-cache-dirs` detects
  this and asks for a one-time sudo to chown them back.
- `compose.yaml` mounts the whole home (`${HOME}:${HOME}`) plus
  `${HOME}/.vllm-workspace:/workspace`; cache dirs are redirected under
  `~/.cache` via env (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `AITER_JIT_DIR`, `TILELANG_CACHE_DIR`) and created lazily by the container as
  the host user, so `just ensure-cache-dirs` only pre-creates `~/.cache` and
  `~/.vllm-workspace`.
- Cache dirs managed by `just clear-vllm-caches`: `~/.cache/{vllm,triton,
  torchinductor,aiter,comgr,tvm-ffi,tilelang}` plus the vllm-qwen-flashnext
  container's `~/.cache/{triton,torchinductor,tilelang}_flashnext` (huggingface
  kept: model weights).
- Always clear caches after updating `VLLM_REF`/`VLLM_VERSION`
  (or `FLASHNEXT_VLLM_REF`/`FLASHNEXT_VLLM_VERSION`) or changing `AITER_REF`
  to avoid stale kernel artifacts causing runtime errors.

### Non-root aiter JIT (required)
- aiter's JIT build falls back to `~/.aiter/jit` when site-packages isn't
  writable (non-root runs), but that dir is only added to `sys.path` when
  `AITER_JIT_DIR` is set. Without it, `import aiter.jit.module_aiter_core` and
  `aiter.ops.triton.unified_attention` fail with `ModuleNotFoundError` and the
  unified-attention backend breaks. `compose.yaml` sets
  `AITER_JIT_DIR=${HOME}/.cache/aiter/jit`.
- `just up` runs `just prewarm` first: it builds the shared aiter kernels
  (module_aiter_core, unified-attention) in one throwaway container before the
  server starts. On a fresh cache this is required — otherwise the model-
  inspection subprocess and both TP workers race to build them
  (`ModuleNotFoundError: aiter.ops.triton.unified_attention`) and a subprocess
  dying at exit leaves a stale aiter baton lock that deadlocks startup.
- A stale baton lock (`~/.cache/aiter/jit/build/lock_*` referencing a dead
  PID/container) is not auto-cleared by aiter; `just prewarm` removes it first.
- The `vllm-qwen-flashnext` container shares this aiter JIT dir with `vllm`
  (deliberate: concurrent *readers* of built kernels are safe; only concurrent
  *builds* race, and prewarm has already done the building). The other
  compile caches are per-container (`*_flashnext` dirs) for the same reason.

### Version Pins
- All build pins live in `.env` (untracked) and `.env.example` (tracked template).
- When upgrading vLLM, update both `VLLM_REF` and `VLLM_VERSION` in **both files**.
- The flashnext image has INDEPENDENT pins: `FLASHNEXT_VLLM_REF`/
  `FLASHNEXT_VLLM_VERSION` (currently a vLLM main-head commit carrying
  qwen4_exp; see `patches/vllm-flashnext/` version-lock comments). Bump them
  only with `just build-flashnext`; the main image is unaffected.
- Key dependencies to cross-check against vLLM release notes:
  - `AITER_REF` — ROCm kernels library
  - `FLASH_ATTN_REF` — flash attention (commit pin)
  - `TORCH_VERSION` / `TORCHVISION_VERSION` — PyTorch stack
- If `just rebuild` terminates (signal 15 / timeout), check the image with
  `docker inspect localhost/vllm-fullbuild:latest` to verify completion before
  attempting `just up`.

### GEMM Kernel Config Tuning
- vLLM keys its Triton GEMM kernel configs on the **per-GPU shape** (dense W8A8
  block-FP8: `N=*,K=*`; fused MoE: `E=*,N=*`), so the tuned files in
  `fp8_configs/` / `fused_moe_configs/` **go stale whenever a profile's TP
  changes** — the server then silently runs stock defaults (warning:
  "Using default W8A8 Block FP8 kernel config"). After a TP change, re-tune.
- `just tune` does this in a throwaway container (same image, `tools/run_tune.py`
  driver): dense shapes are auto-discovered from the `vllm` container's
  startup warnings (`--set shapes "N:K,..."` when the server was already
  stopped), and MoE tuning runs automatically for MoE profiles only, at the
  profile's `VLLM_TP`. It benchmarks on a single GPU
  (`just --set tune_gpu 3 tune`) so a running `vllm` service can be
  stopped/started freely mid-tune; results land in the repo config dirs and
  are picked up on the next `just up`.
- Budget time: on a cold `~/.cache/triton` the first shape takes ~1-5h
  (one-time kernel compiles), each later shape ~30-90min (cached). It is
  interrupt/resume-safe and writes each shape's file as it completes. Expected
  gain (README A/B, 27B at TP=2): ~+19% tg32 decode, ~+3% pp2048 — decode
  wins dominate for conversational workloads.

## Checking for Updates

When asked to "check for updates" (or when a new release is suspected), compare
the pins in `.env`/`.env.example` against upstream, then look for **patches
that affect this GPU setup and model combo** before recommending a bump.

### 1. Upstream release state

```sh
# vLLM — current pin VLLM_REF=v0.28.0rc2
gh release list -R vllm-project/vllm --limit 8

# AITER — current pin AITER_REF=v0.1.20
gh release list -R ROCm/aiter --limit 8

# Flash Attention — pinned to a commit, so compare HEAD to FLASH_ATTN_REF
git ls-remote https://github.com/ROCm/flash-attention.git HEAD

# ROCm base image — current ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full
curl -s "https://hub.docker.com/v2/repositories/rocm/dev-ubuntu-24.04/tags?page_size=100&name=7.1" | jq -r '.results[].name' | sort -V | tail

# Froggeric chat template — current pin is the first line of chat-templates/qwen.jinja
# (template_version = "qwen3.8-froggeric-vXX.X"). Compare against upstream main:
curl -sL https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja | head -1
head -1 chat-templates/qwen.jinja
```

Report what's newer than the current pins and whether the bump is worth it
(see relevance filters below). Do **not** auto-bump pins. Chat-template bumps
are lower-risk than build pins: it's a pure Jinja swap, no rebuild — refresh
`chat-templates/qwen.jinja`, bump the README pin note, `just down && just up`
(the in-memory prefix cache is cleared on restart anyway).

### 2. Scan for open issues affecting this setup

Beyond checking the watchlist (below), actively search for **new** open
issues/PRs each time. Report anything that changes the picture; do **not**
auto-apply fixes.

```sh
# Re-check watchlist status (open/closed/resolved) + any new labels:
for n in 35288 47087 48375 52872 47602 51250 52520 45238 51562 51812 51837 40707 52527 52789 48815 52817 52959 51198 49125; do
  gh issue view $n -R vllm-project/vllm --json state,title,updatedAt 2>/dev/null \
    | jq -r '"\(.state) | \(.updatedAt) | \(.title)"'
done

# New open issues by theme (MTP, hybrid, ROCm, prefix caching):
gh search issues -R vllm-project/vllm --state open --limit 25 "MTP" \
  --json number,title,updatedAt | jq -r '.[] | "\(.number) | \(.updatedAt) | \(.title)"'
gh search issues -R vllm-project/vllm --state open --limit 25 "hybrid" --json number,title
gh search issues -R vllm-project/vllm --state open --limit 25 "ROCm" --json number,title
gh search issues -R vllm-project/vllm --state open --limit 25 "mamba" --json number,title
```

Apply the relevance filters from step 3 when triaging results: track **only**
issues that affect this stack (`gfx1201`/ROCm, hybrid GDN/Mamba path, MTP/
speculative decoding, prefix caching align mode, fp8 KV, AITER unified
attention) **and** one of the tracked models (Qwen3.6-27B, Qwen3.6-35B-A3B,
Qwen3.8-27B). Everything else is out of scope — NVIDIA/CUDA-only issues
included, even if the model matches — and goes to the not-applicable list
below. For a candidate issue, read its body and comments: confirm the root
cause matches a path this stack actually reaches (e.g. check whether an
option the issue requires — async scheduling, KV connectors, DSpark,
turboquant KV, NVFP4 weights, explicit `--block-size` — is even enabled here)
before adding it to the watchlist.

### 3. Relevance filters — does the update matter here?

This stack is not a stock vLLM install. A fix/perf change only matters if it
touches one of:

- **GPU**: `gfx1201` (RDNA4, 2× R9700), ROCm 7.14.0. ROCm-only issues and
  AITER unified-attention paths are in scope; NVIDIA/CUDA-only fixes are not.
- **Models**: Qwen3.6-27B (dense, MTP4), Qwen3.6-35B-A3B (MoE, MTP off),
  Qwen3.8-27B (hybrid GDN, MTP3, 256K context, bf16 KV). Anything touching:
  hybrid Mamba/GDN models, MTP/speculative decoding, prefix caching
  (align mamba cache mode), fp8 KV, or `ROCM_AITER_UNIFIED_ATTN` is in scope.
- **Chat template**: froggeric `chat-templates/qwen.jinja` (pinned, e.g.
  v22.3). A newer version matters when it changes prompt rendering in ways
  this stack hits: history re-render must stay byte-identical to generated
  tokens (KV-cache/prefix-cache invariance) for thinking-off multi-turn,
  tool-argument formatting for the `qwen3_coder` XML parser, or reasoning/
  tool-error heuristics. Template bumps need no rebuild (see step 1).
- **Known-bug watchlist** (search/check these before recommending a vLLM bump):
  - `#35288` MTP concurrency corruption (still mitigated by `max-num-seqs 2`)
  - `#47087` MTP token loops on Qwen3-MoE (resolved by #51113, in v0.27.1 —
    pending 35B MTP re-test)
  - `#51812` Qwen GDN gate/spec-token alignment — **resolved**: merged
    upstream 2026-08-11 (`5af7c8d`), present in v0.28.0rc2; local patch
    dropped 2026-08-21
  - `#51837` ROCm KV-first attention blocks sharing pages with Mamba —
    **resolved**: merged upstream 2026-08-11 (`3e372c5`), present in
    v0.28.0rc2; local patch dropped 2026-08-21. Inert on this stack (AITER
    unified attn is blocks-first, `block_dim == 0`); only matters if a
    KV-first backend is ever selected
  - `#48375` MambaManager ignores `drop_eagle_block` (MTP + prefix caching
    corrupts hybrid recurrent state, #43559/#50188) — **carried as a local patch**
  - `#52872` GDN/hybrid prefill peak under-predicted; `--max-num-batched-tokens`
    also sizes the CUDA-graph pool
  - `#47602` MTP draft acceptance decays with context length (Qwen3.6-27B)
  - `#51250` prefix caching is a silent no-op on GDN hybrid (same family as
    `#45238`)
  - `#51198` newer restatement (2026-08-21) of the `#45238`/`#51250` hybrid
    prefix-cache 0%-hit no-op — confirms the family is still open (monitor)
  - `#49125` stale partial prefix-cache hash resurrected after full-block
    promotion (pure-Python `BlockPool` bug in the fine-grained/partial
    prefix-caching path `#45939`/`#46384`). Only reachable once `#45238` is
    fixed and prefix caching actually hits — monitor post-fix
  - `#52520` align-mode admission livelock near KV-pool ceiling (open)
  - `#45238` hybrid prefix caching drops to 0% in align mode (open) — the
    binding constraint on this stack. Root cause in v0.27.1:
    `BlockPool.cache_full_blocks` skips Mamba align-mode null blocks
    (block_pool.py, "Mamba models with prefix-caching in align mode"), so only
    ~1 checkpoint hash per request is registered; a missing Mamba checkpoint
     vetoes all attention-group hits (every group must hit). Live geometry:
     `block_size=832` on the bf16-KV default (was `1600` on the old fp8-KV
     default — the 2-byte KV halves tokens-per-block; see
     `vllm:cache_config_info`), so hits are 0% whenever
     `floor((prompt_len-1)/832)*832 > shared_prefix_len` — measured **0% on
     the qwen3.8-27b multi-turn probe (2026-08-20, fp8; re-confirmed 2026-08-22
     on bf16-KV, still 0% across 4 turns)**. Fix in flight, none
    merged: `#52527` (metrics for shared-prefix tokens lost to missing
    checkpoints), `#52789` (internal prefill checkpoints, 9–25% TTFT),
    `#48815` (MTP align retention). **When one merges**: carry it as a local
    patch only if the current `VLLM_REF` release does not already contain it —
    if it's in an available vLLM bump, prefer the bump (step 5), not a patch.
  - `#52817` RFC: hybrid SSM + SpecDec + APC re-runs the last full block on a
    prefix hit (832 tokens here on the bf16-KV default; was 1600 on fp8),
    bounding the prefix-cache win for MTP even after `#45238` is fixed. Monitor
    for a merged implementation.
  - `#51562` GDN metadata misclassifies stateless first chunk (open)
  - `#52959` RFC: internal state checkpoints for Mamba align mode (same
    family as `#52789`; in flight, not merged)
  - `#40707` hybrid Mamba scheduling deadlock with 2+ large images in one
    prompt (align block-split collapses to 0 → request hangs forever, engine
    never recovers). Reachable here: all profiles pass
    `--limit-mm-per-prompt image: 99` and both 27B models are this GDN hybrid.
    Fix PR `#40709` is **not merged** (absent from v0.28.0rc2). Only workaround
    is avoiding 2+ large (~11.8K vision-token) images in a single request.

Issues known **not** to apply (checked; re-check only if the stack changes):
NVIDIA-only (#52475, #52583 VL, #51571 async-MTP — async is auto-disabled for
MTP), non-Qwen models (#52833/#48568 GLM, #51530 DeepSeek), or paths not
reached here (PP ranks #51752, DP attention #51957, KV connectors #51805/
#51766/#40017, GPTQ #51971, gfx950 MLA #52312). #52793 (fp8 KV scale-1.0 on
hybrids) was previously logged as a "non-issue" because a d200K/d256K probe
passed coherence at 258k tokens. **Revisited 2026-08-22**: the stock FP8
checkpoints ship no k/v/q scales, so fp8 KV serves at scale 1.0, and a
calibration run (see `benchmarks/2026-08-22_lifted_tested.md`) records deep-layer
V amax up to ~132 vs scale 1.0's ~1-24 assumption — i.e. scale 1.0 is genuinely
miscalibrated, not just unprecise. It does not cause coherence failures, but it
does change KV numerics (calibrated vs scale-1.0 deterministic outputs diverge
 ~20-27%). **Made opt-in 2026-08-22**: the quality A/B
 (`benchmarks/2026-08-22_kv_calibration_quality_ab.md`) found calibrated and
 scale-1.0 fp8 KV **indistinguishable** on both PPL and long-context recall
 (PPL gap ~0.02-0.08%, direction flips with data; recall byte-identical), so
 calibration is a correctness fix, not a measured quality win — and **bf16 KV is
 now the default** (`VLLM_KV_CACHE_DTYPE=bfloat16` in `compose.yaml` +
 `env/qwen3.6.env.common`; `VLLM_GPU_MEM_UTIL=0.95`). When fp8 KV is re-enabled,
 the default profiles still point `VLLM_MODEL` at the calibrated local copy that
 `just up` builds via the `ensure-kvscales` recipe
 (`tools/setup_kvscales.py` + `tools/calibrate_kv_scales.py`, `amax/448`
 sidecar + index entries; recalibrate with `just clear-kvscales`), whose
 coverage includes the **MTP prediction-head layer(s)** (`mtp.layers.*`): the
 calibrator enables MTP spec-decode while capturing so the head's own full-attn
 layer(s) exist and get `mtp.layers.*.self_attn.attn.{q,k,v}_scale` too (they
 cache fp8 KV just like the main layers and were silently at scale 1.0). A
 residual `prob_scale 1.0` warning (kv_cache.py, fp8 attention) is separate from
 the KV cache scales — it's the softmax-probability scale for the fp8 attention
 kernel; q_scale is calibrated and consumed, prob_scale stays 1.0 by default and
 did not cause coherence issues at 258K. Re-visit scale calibration whenever KV
 precision matters (long-context recall), not just on coherence. Also
checked 2026-08-21: #53180 (turboquant_k8v4 + MTP degeneration on hybrid
GDN — NVIDIA Ada/AWQ, we use fp8 KV; same silent-corruption family, so
re-check if turboquant KV is ever tried), #52480 (qwen3_5_mtp TP≥2 load
failure — NVFP4/ModelOpt checkpoints on NVIDIA; our FP8 MTP head loads fine
at TP=2), #53142 (align pre-copy IMA on prefix-cache resume — requires
explicit `--block-size`, which we never pass).

### 4. Local patches vs upstream

`patches/vllm/*.patch` (main image), `patches/vllm-flashnext/*.patch`
(flashnext image) and `patches/aiter/*.patch` are cherry-picks/overrides
applied at build time by `Dockerfile.fullbuild` / `Dockerfile.flashnext`
respectively; `patches/*/protocol.py` files are compose volume-mount
overlays, version-locked per image. Before bumping any pin:

- Check whether a newer `VLLM_REF`/`FLASHNEXT_VLLM_REF` **already contains**
  a carried patch (the fix landed upstream). If so, the patch should be
  **dropped**, not kept. Verify: `gh pr view <pr> --repo vllm-project/vllm`
  and check the PR's merged status + which release tag includes it (compare
  tag commits via `git ls-remote --tags https://github.com/vllm-project/vllm.git`).
- After any pin change, verify each patch still applies cleanly on the new
  ref before building; a failed `git apply` in the Dockerfile aborts the
  build. Bump the version-lock comment in each patch header too. If a fix is
  needed for BOTH images, carry it in BOTH patch dirs (the dirs are NOT
  symlinked — copy the patch and adjust the version-lock header).
- Always `just clear-vllm-caches` after a `VLLM_REF`/`VLLM_VERSION`/
  `FLASHNEXT_VLLM_REF`/`FLASHNEXT_VLLM_VERSION`/`AITER_REF` change, then
  `just rebuild` / `just rebuild-flashnext` (see Rebuild Timeouts).

### 5. Recommended bump checklist

1. Diff `.env.example` vs `.env` — keep both in sync.
2. Update `VLLM_REF` + `VLLM_VERSION` together; verify `AITER_REF` and
   `FLASH_ATTN_REF` are compatible with the new vLLM release notes.
3. Check `TORCH_VERSION`/`TORCHVISION_VERSION` against the vLLM release's
   supported ROCm/PyTorch stack.
4. Re-check the patch watchlist (step 2/3) and drop/rebase local patches.
5. Refresh `chat-templates/qwen.jinja` if froggeric shipped a newer version
   (step 1): curl from upstream `main`, bump the README pin note, then
   `just down && just up`. No rebuild or cache clearing needed.
6. `just clear-vllm-caches && just rebuild && just up`, then `just bench` to
   confirm no regression vs `README.md`/`benchmarks/` baselines.
7. Re-run the prefix-cache hit-rate probe (`benchmarks/prefix_cache_probe.py`)
   after any vLLM bump/restart and record whether
   `vllm:prefix_cache_hits_total` moves off 0% (see `#45238`). A non-zero hit
   rate on the multi-turn probe is the signal the align-mode checkpoint fix
   landed and is worth carrying/keeping.
8. Update `README.md` (patches, pins, bench tables) and commit to `origin`.
