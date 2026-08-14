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
    profile="env/{{model}}.env"
    if [ ! -r "$profile" ]; then
        printf 'error: no such model profile: %s\n' "$profile" >&2
        printf 'available profiles:\n' >&2
        for f in env/*.env; do
            case "$f" in env/aiter-*) continue ;; esac
            printf '  %s\n' "$(basename "$f" .env)" >&2
        done
        exit 1
    fi
    {{compose}} config --quiet
    printf 'Config OK (profile: {{model}}, model: %s).\n' \
        "$(grep -m1 '^VLLM_MODEL=' "$profile" | cut -d= -f2-)"

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

up: check ensure-cache-dirs prewarm
    #!/usr/bin/env bash
    set -euo pipefail
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    {{compose}} up -d
    printf 'Waiting for server to be ready ...\n'
    ready=0
    for _ in $(seq 1 150); do
        if curl -sf --max-time 3 http://localhost:8180/health > /dev/null 2>&1; then
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
    if curl -sf --max-time 300 http://localhost:8180/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${served_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":100,\"temperature\":0}" \
        > /dev/null; then
        printf 'Warmup complete. Serving %s at http://localhost:8180/v1\n' "$served_name"
    else
        printf 'error: warmup request failed. Run `just logs`.\n' >&2
        exit 1
    fi

# Run a command inside the running vLLM container (e.g. `just exec bash`).
exec *args:
    @{{compose}} exec vllm {{args}}

logs:
    @{{compose}} logs -f

bench: check
    #!/usr/bin/env bash
    set -euo pipefail
    model="$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2-)"
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    tokenizer="$(grep -m1 '^VLLM_TOKENIZER=' "env/{{model}}.env" | cut -d= -f2-)"
    printf 'Benchmarking %s (pp2048, tg32+128) ...\n\n' "$model"
    uvx llama-benchy@0.4.0 --base-url http://localhost:8180/v1 \
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
