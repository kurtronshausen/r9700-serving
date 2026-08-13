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
| `just up` | start the vLLM server (runs `check`, starts container, waits for readiness, runs warmup) |
| `just logs` | follow container logs (`docker compose logs -f`) |
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
  `docker inspect localhost/vllm-fullbuild:latest` to verify completion before
  attempting `just up`.
