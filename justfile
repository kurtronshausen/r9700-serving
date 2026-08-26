set positional-arguments

# Container runtime: "docker" (default) or "podman". Override with
# `just --set runtime podman <recipe>` or `RUNTIME=podman just <recipe>`.
runtime := env_var_or_default('RUNTIME', 'docker')

# Model profile: "qwen3.8-27b" (default), "qwen3.6-27b", or "qwen3.6-35b-a3b".
# Selects env/${model}.env for model-specific server arguments.
# Override with `just --set model qwen3.6-27b <recipe>` or
# `MODEL_PROFILE=qwen3.6-27b just <recipe>`.
model := env_var_or_default('MODEL_PROFILE', 'qwen3.8-27b')

# Exported so compose can resolve the `env_file:` path for the model profile.
export MODEL_PROFILE := model

# Physical GPU index the `tune` container benchmarks on. All four GPUs are
# visible to every container, so a running `vllm` server shares the box; pick
# `tune_gpu` to steer contention (or stop the server — tuning is independent
# of it). Override: `just --set tune_gpu 3 tune` or TUNE_GPU=3.
tune_gpu := env_var_or_default('TUNE_GPU', '0')

# Explicit dense GEMM shapes ("N:K,N:K") for `tune`. Leave empty to
# auto-discover from the `vllm` container's startup warnings. Needed only when
# the server has already been stopped and its logs are gone.
# Override: `just --set shapes "5120:4352,5120:5120" tune`
shapes := ''

# Vision model profile: selects env/${vision_model}.env for the `vllm-vision`
# service (a second, concurrent container on its own port and cache dirs).
# Override with `just --set vision_model <profile>` or
# `VISION_MODEL_PROFILE=<profile> just <recipe>`.
vision_model := env_var_or_default('VISION_MODEL_PROFILE', 'qwen2.5-vl-72b-instruct')

# Exported so compose can resolve the `env_file:` path for the vision profile.
export VISION_MODEL_PROFILE := vision_model

# Every compose call must load `.env` (build pins) plus the env_file stack
# that compose.yaml declares (common → aiter → qwen3.6 → profile), because
# passing --env-file disables the implicit .env resolution, so each must be
# listed explicitly.
compose := runtime + " compose --env-file .env --env-file env/2xr9700.vllm.common --env-file env/aiter-unified-attention.env --env-file env/qwen3.6.env.common --env-file env/" + model + ".env"

_default:
    @just --list

# Validate the compose config for the selected model profile.
check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -r .env ]; then
        printf 'error: no .env found. Copy the template: cp .env.example .env\n' >&2
        exit 1
    fi
    for p in "env/{{model}}.env" "env/{{vision_model}}.env"; do
        if [ ! -r "$p" ]; then
            printf 'error: no such model profile: %s\n' "$p" >&2
            printf 'available profiles:\n' >&2
            for f in env/*.env; do
                case "$f" in env/aiter-*|env/*.common) continue ;; esac
                printf '  %s\n' "$(basename "$f" .env)" >&2
            done
            exit 1
        fi
    done
    {{compose}} config --quiet
    printf 'Config OK (model: %s, vision: %s).\n' \
        "$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2-)" \
        "$(grep -m1 '^VLLM_MODEL=' "env/{{vision_model}}.env" | cut -d= -f2-)"

# Ensure the whole-home mount sources exist and are owned by the current user.
# Docker's daemon pre-creates missing bind-mount sources as root, which breaks
# non-root container writes and forces sudo for clear-vllm-caches; pre-creating
# them here prevents that. Individual ~/.cache/* dirs are created lazily by the
# container as the user, so only the top-level dirs need pre-creation. Safe to
# run repeatedly.
ensure-cache-dirs:
    #!/usr/bin/env bash
    set -euo pipefail
    cache_dirs=(
        "$HOME/.cache"
        "$HOME/.vllm-workspace"
    )
    uid="$(id -u)"
    gid="$(id -g)"
    mkdir -p "${cache_dirs[@]}"
    needs_sudo=0
    for dir in "${cache_dirs[@]}"; do
        if [ "$(stat -c '%u' "$dir")" != "$uid" ]; then
            needs_sudo=1
        fi
    done
    if [ "$needs_sudo" -eq 1 ]; then
        printf 'Some cache dirs are root-owned; fixing ownership (one-time sudo required).\n' >&2
        sudo chown -R "$uid:$gid" "$HOME/.cache" "$HOME/.vllm-workspace"
    fi
    printf 'Cache dirs ready (owner: %s).\n' "$(id -un)"

# Remove host cache directories written by the container (under ~/.cache).
clear-vllm-caches:
    #!/usr/bin/env bash
    set -euo pipefail

    cache_dirs=(
        # "$HOME/.cache/huggingface"
        "$HOME/.cache/vllm"
        "$HOME/.cache/triton"
        "$HOME/.cache/torchinductor"
        "$HOME/.cache/aiter"
        "$HOME/.cache/comgr"
        "$HOME/.cache/tvm-ffi"
        "$HOME/.cache/tilelang"
        # vllm-vision service's per-container compile caches
        "$HOME/.cache/triton_vision"
        "$HOME/.cache/torchinductor_vision"
        "$HOME/.cache/tilelang_vision"
    )

    printf 'Removing vLLM host cache directories:\n'
    for dir in "${cache_dirs[@]}"; do
        if [ ! -e "$dir" ]; then
            printf '  %s (absent, skipped)\n' "$dir"
        elif [ ! -w "$dir" ]; then
            printf '  %s (not writable, using sudo)\n' "$dir"
            sudo rm -rf -- "$dir"
        else
            printf '  %s\n' "$dir"
            rm -rf -- "$dir"
        fi
    done

build: check
    @{{compose}} build

rebuild: check
    @{{compose}} build --no-cache

# Build the shared aiter JIT infrastructure (module_aiter_core, the
# unified-attention triton kernels) in a single throwaway process BEFORE the
# server starts. On a fresh cache the first `vllm serve` run would otherwise
# have the model-inspection subprocess and both TP workers racing to build
# these kernels: losers get `ModuleNotFoundError: aiter.ops.triton.unified_attention`
# and a subprocess that dies at exit can leave a stale aiter baton lock that
# deadlocks startup. Warm them first so every later build is a 0s cache hit.
prewarm: check ensure-cache-dirs
    #!/usr/bin/env bash
    set -euo pipefail
    # A process that dies mid-build can leave a stale aiter baton lock that
    # `file_baton` never detects as dead, hanging the next build forever. No
    # aiter process runs between container stops, so clearing any leftover lock
    # before building is always safe here.
    rm -f "$HOME/.cache/aiter/jit/build/lock_"* 2>/dev/null || true
    printf 'Pre-warming aiter JIT kernels ...\n'
    {{compose}} run -T --rm --no-deps --entrypoint python vllm \
        -c "import aiter; from aiter.ops.triton.unified_attention import unified_attention"
    printf 'aiter pre-warm complete.\n'

# Tune the Triton GEMM kernel configs for the selected model profile (dense
# w8a8 block-FP8 + fused MoE) in a throwaway container, independent of the
# running `vllm` service: `just down`/`just up` mid-tune won't interrupt it
# (separate container; it shares only the GPUs and the compile caches).
#
# Dense shapes are auto-discovered from the `vllm` container's startup
# warnings ("Using default W8A8 Block FP8 kernel config ... N=*,K=*"), so the
# server must have been started once with this profile+TP. If it has been
# stopped and the logs are gone, override with `--set shapes "N:K,N:K"`.
# MoE tuning runs automatically for MoE profiles only, at the profile's TP
# (VLLM_TP). Results land in ./fp8_configs and ./fused_moe_configs (rw
# mounts); the next `just up` picks them up. Slow on a cold Triton cache
# (hours) — it is interrupt/resume-safe (compiles persist in
# ~/.cache/triton), and each shape writes its file as it completes.
tune: check ensure-cache-dirs
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{shapes}}" ]; then
        dense_shapes="{{shapes}}"
    else
        dense_shapes="$( {{runtime}} logs vllm 2>&1 \
            | grep -oE 'N=[0-9]+,K=[0-9]+' \
            | sed 's/N=//; s/,K=/:/' | sort -u | paste -sd, - || true )"
        if [ -z "$dense_shapes" ]; then
            printf 'error: no missing W8A8 block-FP8 shapes in the `vllm` container logs.\n' >&2
            printf 'Start the server once (just up) so it logs them, or pass them explicitly:\n' >&2
            printf '  just --set shapes "N:K,N:K" tune\n' >&2
            exit 1
        fi
    fi
    printf 'Tuning on GPU {{tune_gpu}} (dense shapes: %s) ...\n' "$dense_shapes"
    {{compose}} run -T --rm --no-deps \
        -e DENSE_SHAPES="$dense_shapes" \
        -e REPO_DIR=/app \
        -e HIP_VISIBLE_DEVICES="{{tune_gpu}}" \
        --entrypoint python vllm /app/tools/run_tune.py
    printf 'Tune complete. Run `just up` to pick up the new configs, then `just bench`.\n'

# Ensure the selected profile's calibrated KV-scale model copy exists. Profiles
# that want calibrated fp8 KV declare VLLM_MODEL_ID (HF source id) + VLLM_MODEL
# (local copy path) in their env file. If the local copy's sidecar is missing,
# create the copy from the HF cache and calibrate it in one throwaway container.
# Idempotent: skips once the sidecar exists. Runs before the server starts, so
# the GPUs are free for calibration.
ensure-kvscales:
    #!/usr/bin/env bash
    set -euo pipefail
    profile="env/{{model}}.env"
    model_id="$(grep -m1 '^VLLM_MODEL_ID=' "$profile" | cut -d= -f2- || true)"
    local_model="$(grep -m1 '^VLLM_MODEL=' "$profile" | cut -d= -f2- || true)"
    if [ -z "$model_id" ]; then
        printf 'profile %s has no VLLM_MODEL_ID; skipping kv-scale setup.\n' "$profile"
        exit 0
    fi
    sidecar="$local_model/model-kvscales.safetensors"
    if [ -f "$sidecar" ]; then
        printf 'kv-scales up to date: %s\n' "$local_model"
        exit 0
    fi
    printf 'setting up calibrated KV-scale copy: %s -> %s\n' "$model_id" "$local_model"
    python3 tools/setup_kvscales.py "$model_id" "$local_model"
    printf 'calibrating fp8 KV scales ...\n'
    {{compose}} run -T --rm --no-deps -e CALIB_KV_LOG=/workspace/kvscale.log \
        --entrypoint python vllm \
        "$PWD/tools/calibrate_kv_scales.py" "$local_model"
    printf 'kv-scales calibrated: %s\n' "$local_model"

# Remove the calibrated model copy so the next `just up` recalibrates it.
clear-kvscales:
    #!/usr/bin/env bash
    set -euo pipefail
    local_model="$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2- || true)"
    if [ -n "$local_model" ] && [ -d "$local_model" ]; then
        rm -rf -- "$local_model"
        printf 'removed %s\n' "$local_model"
    else
        printf 'no local model copy for profile %s.\n' "{{model}}"
    fi

up: check ensure-cache-dirs prewarm ensure-kvscales
    #!/usr/bin/env bash
    set -euo pipefail
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    # Host ports are overridable in .env (PORT / VISION_PORT); compose defaults
    # are 8000 (vllm) and 8001 (vllm-vision).
    port="$(grep -m1 '^PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    port="${port:-8000}"
    vision_port="$(grep -m1 '^VISION_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    vision_port="${vision_port:-8001}"
    # Starts BOTH services (`vllm` + `vllm-vision`); the vision container
    # keeps loading in the background while we wait on the main one.
    {{compose}} up -d
    printf 'Waiting for server to be ready ...\n'
    ready=0
    for _ in $(seq 1 150); do
        if curl -sf --max-time 3 "http://localhost:${port}/health" > /dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        printf 'error: server did not become ready within 300s. Run `just logs`.\n' >&2
        exit 1
    fi
    printf 'Warming up Triton kernels ...\n'
    if curl -sf --max-time 300 "http://localhost:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${served_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":100,\"temperature\":0}" \
        > /dev/null; then
        printf 'Warmup complete. Serving %s at http://localhost:%s/v1\n' "$served_name" "$port"
    else
        printf 'error: warmup request failed. Run `just logs`.\n' >&2
        exit 1
    fi
    printf 'Vision profile %s starting at http://localhost:%s/v1 (large model —\n' "{{vision_model}}" "$vision_port"
    printf 'check readiness with `just logs vllm-vision` or `just compose ps`).\n'

# Run a command inside the running vLLM container (e.g. `just exec bash`).
exec *args:
    @{{compose}} exec vllm {{args}}

# `just logs` follows all services; `just logs vllm-vision` one of them.
logs *args:
    @{{compose}} logs -f {{args}}

# Raw compose passthrough for service-level ops
# (e.g. `just compose up -d vllm`, `just compose ps`, `just compose down vllm-vision`).
compose *args:
    @{{compose}} {{args}}

bench: check
    #!/usr/bin/env bash
    set -euo pipefail
    model="$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2-)"
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    tokenizer="$(grep -m1 '^VLLM_TOKENIZER=' "env/{{model}}.env" | cut -d= -f2-)"
    port="$(grep -m1 '^PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    port="${port:-8000}"
    printf 'Benchmarking %s (pp2048, tg32+128) ...\n\n' "$model"
    uvx llama-benchy@0.4.0 --base-url "http://localhost:${port}/v1" \
        --model "$served_name" \
        --tokenizer "$tokenizer" \
        --pp 2048 \
        --tg 32 128 \
        --runs 3 \
        --enable-prefix-caching \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
        --format md

down:
    @{{compose}} down
