# AGENTS

## Rules

- **Never commit, push, open PRs, or make any remote changes targeting `andysalerno/r9700-serving` (remote: `upstream`/`andysalerno`) without explicit, direct instructions from the user — and confirm intent before doing so.**
- Always target `origin` (your own fork) for commits, pushes, and PRs unless explicitly told otherwise.

## Tools

Use `just` for all build/run workflows. Commands are defined in `justfile`.

| recipe | purpose |
|:-------|:--------|
| `just check` | validate the compose config for the selected model profile |
| `just build` | build the Docker image |
| `just rebuild` | force-rebuild (no cache) |
| `just up` | start the vLLM server (runs `check`, `ensure-cache-dirs`, `prewarm`, starts container, waits for readiness, runs warmup) |
| `just prewarm` | build shared aiter JIT kernels in one throwaway container (runs automatically before every `up`) |
| `just bench` | benchmark the selected model via `llama-benchy` (pp2048, tg32+128) |
| `just logs` | follow container logs (compose `logs -f`) |
| `just down` | stop and remove the container |
| `just exec <cmd>` | run a command inside the running container (e.g. `just exec bash`) |
| `just ensure-cache-dirs` | pre-create host cache dirs owned by the current user |
| `just clear-vllm-caches` | wipe compile caches (triton, torchinductor, aiter, etc.) |

`compose.yaml` interpolates `VLLM_MODEL`/`VLLM_TOKENIZER`/`VLLM_SERVED_NAME`/
`VLLM_SPEC_DECODE` from `env/<profile>.env`, which the recipes pass to compose
via `--env-file` (alongside `.env` for the build pins). Bare `docker compose up`
fails with a required-variable error by design — always go through `just`.
`.env` is untracked; create it with `cp .env.example .env`.

To switch models, set `MODEL_PROFILE` or use `--set`:

```
MODEL_PROFILE=qwen3.6-27b just up
just --set model qwen3.6-27b up
```

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
  torchinductor,aiter,comgr,tvm-ffi,tilelang}` (huggingface kept: model weights).
- Always clear caches after updating `VLLM_REF`/`VLLM_VERSION` or changing
  `AITER_REF` to avoid stale kernel artifacts causing runtime errors.

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

### Version Pins
- All build pins live in `.env` (untracked) and `.env.example` (tracked template).
- When upgrading vLLM, update both `VLLM_REF` and `VLLM_VERSION` in **both files**.
- Key dependencies to cross-check against vLLM release notes:
  - `AITER_REF` — ROCm kernels library
  - `FLASH_ATTN_REF` — flash attention (commit pin)
  - `TORCH_VERSION` / `TORCHVISION_VERSION` — PyTorch stack
- If `just rebuild` terminates (signal 15 / timeout), check the image with
  `docker inspect localhost/vllm-fullbuild:latest` (or `podman inspect`) to
  verify completion before attempting `just up`.

### Podman (verified on podman 5.7 / buildah 1.42, docker-compose provider)
- `just` takes a runtime: `RUNTIME=podman just up` / `just --set runtime
  podman up`. `just check` (compose config) works out of the box.
- Runtime ops (`up`/`down`/`logs`/`run`/`exec`) need the podman API service:
  `systemctl --user enable --now podman.socket` (or `podman system service &`).
  Without it, `podman compose` fails to connect to
  `/run/user/$UID/podman/podman.sock`. `up`/`ps`/`down` were verified working
  with the service running.
- Builds work, but buildah silently ignores `RUN --mount=type=cache` (target
  dirs are pre-created but not shared across steps), so `just rebuild` loses
  pip/ccache cache reuse and runs slower than the Docker estimate above.

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
    binding constraint on this stack. Root cause:
    `BlockPool.cache_full_blocks` skips Mamba align-mode null blocks, so only
    ~1 checkpoint hash per request is registered and a missing Mamba
    checkpoint vetoes every attention-group hit. Live geometry:
    `block_size=832` on the bf16-KV default (was `1600` on fp8), so
    incremental multi-turn prefixes never hit — measured **0% on the 30-turn
    qwen3.8-27b probe (re-confirmed 2026-08-23)**. Note the cumulative
    `vllm:prefix_cache_hits_total` is non-zero: caching *does* hit on
    repeated-identical-prompt workloads — the failure is specific to the
    incremental shared-prefix pattern, not a global no-op. Fixes in flight:
    `#52527` (metrics), `#48815` (MTP align retention); `#52789` (internal
    prefill checkpoints) merged 2026-08-22 but postdates the rc2 tag
    (main-only, Kimi-K3/FlashKDA-specific TTFT win, not a fix for the 0%-hit
    geometry — not worth cherry-picking). **When a real fix merges**: prefer
    the version bump; carry a local patch only if no available release
    contains it.
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
#51766/#40017, GPTQ #51971, gfx950 MLA #52312). #52688/#53397 (multi-layer MTP
spec_step_idx — all K draft steps re-execute layers[0]; both 27B models have
`mtp_num_hidden_layers=1`, so layer-0-only is correct and N/A). #53136 (ROCm
all-reduce 8–16 MiB dead zone → RCCL generic-kernel launch fault; requires
`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` and the fault was gfx942 TP=8-specific —
we never set QUICK_REDUCE and gfx1201 TP=2 is stable; re-check only if that
changes). #52793 (fp8 KV scale-1.0 on hybrids): no coherence failures
 (d258K probe passed), but the stock FP8 checkpoints ship no k/v/q scales, so
 fp8 KV serves at scale 1.0, which is genuinely miscalibrated (deep-layer V
 amax ~132 vs the ~1-24 range scale 1.0 assumes; calibrated vs scale-1.0
 outputs diverge ~20-27%). The 2026-08-22 quality A/B
 (`benchmarks/2026-08-22_kv_calibration_quality_ab.md`) found calibrated and
 scale-1.0 fp8 KV **indistinguishable** on PPL and long-context recall — so
 calibration is a correctness fix, not a measured quality win, and **bf16 KV
 is the default**. When fp8 KV is re-enabled, the default profiles still point
 `VLLM_MODEL` at the calibrated local copy that `just up` builds via
 `ensure-kvscales` (recalibrate with `just clear-kvscales`), whose coverage
 includes the **MTP prediction-head layer(s)** (`mtp.layers.*`, which cache
 fp8 KV separately and were silently at scale 1.0). The residual `prob_scale
 1.0` warning is the fp8-attention softmax-probability scale, separate from
 the KV cache scales, and caused no coherence issues at 258K. Re-visit
 calibration whenever KV precision matters (long-context recall). Also
 checked 2026-08-21: #53180 (turboquant_k8v4 + MTP degeneration on hybrid
GDN — NVIDIA Ada/AWQ, we use fp8 KV; same silent-corruption family, so
re-check if turboquant KV is ever tried), #52480 (qwen3_5_mtp TP≥2 load
failure — NVFP4/ModelOpt checkpoints on NVIDIA; our FP8 MTP head loads fine
at TP=2), #53142 (align pre-copy IMA on prefix-cache resume — requires
explicit `--block-size`, which we never pass).

### 4. Local patches vs upstream

`patches/vllm/*.patch` and `patches/aiter/*.patch` are cherry-picks/overrides
applied at build time. Before bumping any pin:

- Check whether a newer `VLLM_REF` **already contains** a carried patch (the
  fix landed upstream). If so, the patch should be **dropped**, not kept.
  Verify: `gh pr view <pr> --repo vllm-project/vllm` and check the PR's merged
  status + which release tag includes it (compare tag commits via
  `git ls-remote --tags https://github.com/vllm-project/vllm.git`).
- After any pin change, verify each patch still applies cleanly on the new
  ref before building; a failed `git apply` in `Dockerfile.fullbuild` aborts
  the build. Bump the version-lock comment in each patch header too.
- Always `just clear-vllm-caches` after a `VLLM_REF`/`VLLM_VERSION`/`AITER_REF`
  change, then `just rebuild` (see Rebuild Timeouts).

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
