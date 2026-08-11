set positional-arguments

# Container runtime: "docker" (default) or "podman". Override with
# `just --set runtime podman <recipe>` or `RUNTIME=podman just <recipe>`.
runtime := env_var_or_default('RUNTIME', 'docker')

# Model profile: "qwen3.6-35b-a3b" (default) or "qwen3.6-27b".
# Selects env/${model}.env for model-specific server arguments.
# Override with `just --set model qwen3.6-27b <recipe>` or
# `MODEL_PROFILE=qwen3.6-27b just <recipe>`.
model := env_var_or_default('MODEL_PROFILE', 'qwen3.6-35b-a3b')

_default:
    @just --list

# Remove host cache directories mounted into vLLM containers.
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
    )

    printf 'Removing vLLM host cache directories:\n'
    for dir in "${cache_dirs[@]}"; do
        if [ ! -w "$dir" ]; then
            printf '  %s (not writable, using sudo)\n' "$dir"
            sudo rm -rf -- "$dir"
        else
            printf '  %s\n' "$dir"
            rm -rf -- "$dir"
        fi
    done

build:
    @MODEL_PROFILE={{model}} {{runtime}} compose build

rebuild:
    @MODEL_PROFILE={{model}} {{runtime}} compose build --no-cache

up:
    #!/usr/bin/env bash
    set -a
    source "env/{{model}}.env"
    MODEL_PROFILE={{model}}
    {{runtime}} compose up -d
    printf 'Waiting for server to be ready ...\n'
    for _ in $(seq 1 90); do
        if curl -sf --max-time 3 http://localhost:8180/health > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    printf 'Warming up Triton kernels ...\n'
    curl -s --max-time 120 http://localhost:8180/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":100,"temperature":0}' \
        > /dev/null 2>&1 && printf 'Warmup complete.\n' || printf 'Warmup failed (server may still be starting).\n'

# Run a command inside the running vLLM container (e.g. `just exec bash`).
exec *args:
    @{{runtime}} compose exec vllm {{args}}

# Send a chat completion request to the running server.
chat prompt max_tokens="100":
    #!/usr/bin/env bash
    curl -s http://localhost:8180/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"{{model}}\",\"messages\":[{\"role\":\"user\",\"content\":\"{{prompt}}\"}],\"max_tokens\":{{max_tokens}},\"temperature\":0}" \
        | python3 -m json.tool 2>/dev/null || cat

logs:
    @{{runtime}} compose logs -f

down:
    @{{runtime}} compose down
