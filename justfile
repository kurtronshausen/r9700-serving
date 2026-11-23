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

# The second, concurrent service is fixed to the Qwen3.8-Flash-Next profile
# (env/qwen3.8-flash-next.env, `vllm-qwen-flashnext` service, port 8001,
# built from Dockerfile.flashnext). It is not started by `just up` — see
# `just up-flashnext`.
#
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
    for p in "env/{{model}}.env" "env/qwen3.8-flash-next.env"; do
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
    printf 'Config OK (model: %s, flashnext: %s, radiance: qwen3.8-27b-radiance).\n' \
        "$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2-)" \
        "$(grep -m1 '^VLLM_MODEL=' "env/qwen3.8-flash-next.env" | cut -d= -f2-)"

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
        # vllm-qwen-flashnext service's per-container compile caches
        "$HOME/.cache/triton_flashnext"
        "$HOME/.cache/torchinductor_flashnext"
        "$HOME/.cache/tilelang_flashnext"
        # vllm-radiance service's per-container compile caches
        "$HOME/.cache/vllm_radiance"
        "$HOME/.cache/triton_radiance"
        "$HOME/.cache/torchinductor_radiance"
        "$HOME/.cache/aiter_radiance"
        # vllm-mxfp4 service's per-container compile caches
        "$HOME/.cache/vllm_mxfp4"
        "$HOME/.cache/triton_mxfp4"
        "$HOME/.cache/torchinductor_mxfp4"
        "$HOME/.cache/aiter_mxfp4"
        # vllm-gptq service's per-container compile caches
        "$HOME/.cache/vllm_gptq"
        "$HOME/.cache/triton_gptq"
        "$HOME/.cache/torchinductor_gptq"
        "$HOME/.cache/aiter_gptq"
        # vllm-gptq-aa service's per-container compile caches
        "$HOME/.cache/vllm_gptqaa"
        "$HOME/.cache/triton_gptqaa"
        "$HOME/.cache/torchinductor_gptqaa"
        "$HOME/.cache/aiter_gptqaa"
        # image-gen service's compile caches (NOT its HF_HOME — the checkpoint
        # is a local /srv/llm dir, so HF_HOME holds little and wiping it only
        # forces re-fetches of tokenizer/processor assets).
        "$HOME/.cache/triton_imagegen"
        "$HOME/.cache/torchinductor_imagegen"
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

# Build the MAIN image (Dockerfile.fullbuild, VLLM_REF) for the `vllm`
# service only. The flashnext image is separate — `just build-flashnext` —
# so day-to-day 27b work never rebuilds (or needs) the Qwen4 stack.
build: check
    @{{compose}} build vllm

# Build the Qwen3.8-Flash-Next image (Dockerfile.flashnext,
# FLASHNEXT_VLLM_REF + patches/vllm-flashnext/) for the
# `vllm-qwen-flashnext` service. Stages identical to the main build hit the
# shared docker layer cache; expect ~15-30 min on a warm cache (the vLLM
# stage only), 40-60 min cold.
build-flashnext: check
    @{{compose}} build vllm-qwen-flashnext

rebuild: check
    @{{compose}} build --no-cache vllm

rebuild-flashnext: check
    @{{compose}} build --no-cache vllm-qwen-flashnext

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
        -e REPO_DIR="$PWD" \
        -e HIP_VISIBLE_DEVICES="{{tune_gpu}}" \
        --entrypoint python vllm "$PWD/tools/run_tune.py"
    printf 'Tune complete. Run `just up` to pick up the new configs, then `just bench`.\n'

# Tune the fused-MoE Triton kernel config for the Qwen3.8-Flash-Next profile
# (int4_w4a16, expert-parallel geometry — auto-detected by run_tune.py from
# the model's quantization_config + VLLM_EXTRA_ARGS) in a throwaway
# container, independent of the running `vllm-qwen-flashnext` service.
# Dense (attention-projection) tuning is out of scope here: flashnext's
# dense layers use RDNAHybridW4A16LinearKernel, a hand-written kernel tuned
# via a hardcoded override table in vLLM source, not a JSON config file —
# see plan notes / Track B. Results land in ./fused_moe_configs (rw mount);
# the next `just up-flashnext` picks them up. Faster than the fp8 MoE tune
# (smaller shapes): budget on the order of tens of minutes across the M
# sweep on a cold Triton cache.
tune-flashnext: check ensure-cache-dirs
    #!/usr/bin/env bash
    set -euo pipefail
    printf 'Tuning flashnext fused MoE on GPU {{tune_gpu}} ...\n'
    {{compose}} run -T --rm --no-deps \
        -e REPO_DIR="$PWD" \
        -e HIP_VISIBLE_DEVICES="{{tune_gpu}}" \
        --entrypoint python vllm-qwen-flashnext "$PWD/tools/run_tune.py"
    printf 'Tune complete. Run `just up-flashnext` to pick up the new config, then `just bench`.\n'

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

# Start the MAIN `vllm` service (the selected 27b/35b profile). The
# Qwen3.8-Flash-Next service is separate: `just up-flashnext` (requires
# `just build-flashnext` first).
up: check ensure-cache-dirs prewarm ensure-kvscales
    #!/usr/bin/env bash
    set -euo pipefail
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    # Host ports are overridable in .env (PORT / FLASHNEXT_PORT); compose
    # defaults are 8000 (vllm) and 8001 (vllm-qwen-flashnext).
    port="$(grep -m1 '^PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    port="${port:-8000}"
    {{compose}} up -d vllm
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

# Start the Qwen3.8-Flash-Next service (concurrent with `vllm`; assign it
# non-overlapping GPUs at runtime when running both — see README). Needs
# `just build-flashnext` first, and ~110 GiB of free host RAM for the
# pinned PLE table.
up-flashnext: check ensure-cache-dirs
    #!/usr/bin/env bash
    set -euo pipefail
    flashnext_port="$(grep -m1 '^FLASHNEXT_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    flashnext_port="${flashnext_port:-8001}"
    {{compose}} up -d vllm-qwen-flashnext
    printf 'vllm-qwen-flashnext starting at http://localhost:%s/v1 (large model —\n' "$flashnext_port"
    printf 'check readiness with `just logs vllm-qwen-flashnext` or `just compose ps`).\n'

# Start the vllm-radiance service (Qwen3.8-27B-FP8 via the prebuilt
# stilldeadcode/vllm-radiance image — no build step, just `docker pull`).
# Not intended to run concurrently with `vllm`/`vllm-qwen-flashnext` — no
# GPU partitioning is applied by default (RADIANCE_HIP_VISIBLE_DEVICES
# defaults to 0,1 for the TP=2 bring-up phase; see plan.md).
up-radiance: check
    #!/usr/bin/env bash
    set -euo pipefail
    radiance_port="$(grep -m1 '^RADIANCE_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    radiance_port="${radiance_port:-8002}"
    {{compose}} up -d vllm-radiance
    printf 'vllm-radiance starting at http://localhost:%s/v1 (fresh image —\n' "$radiance_port"
    printf 'first boot compiles Triton/inductor kernels, can take several\n'
    printf 'minutes; check readiness with `just logs vllm-radiance` or\n'
    printf '`just compose ps`).\n'

# Stop just the vllm-radiance service (leaves `vllm`/`vllm-qwen-flashnext`
# running if they are up).
down-radiance:
    @{{compose}} down vllm-radiance

# Start the vllm-mxfp4 service (Qwen3.8-Flash-Next-MXFP4-FP8 via the prebuilt
# tcclaviger/vllm:dev image — no build step, just `docker pull`). TP=4 wants
# all four GPUs and the PLE table pins ~110 GiB of host RAM, so it is not
# intended to run concurrently with `vllm`/`vllm-qwen-flashnext`/`vllm-radiance`.
up-mxfp4: check
    #!/usr/bin/env bash
    set -euo pipefail
    mxfp4_model="/srv/llm/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8"
    if [ ! -f "$mxfp4_model/model.safetensors.index.json" ]; then
      printf 'error: model not found at %s\n' "$mxfp4_model" >&2
      printf 'Download it first (125.8 GB, resumable):\n' >&2
      printf '  hf download tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8 \\\n' >&2
      printf '    --local-dir %s\n' "$mxfp4_model" >&2
      exit 1
    fi
    mxfp4_port="$(grep -m1 '^MXFP4_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    mxfp4_port="${mxfp4_port:-8003}"
    {{compose}} up -d vllm-mxfp4
    printf 'vllm-mxfp4 starting at http://localhost:%s/v1 (fresh image —\n' "$mxfp4_port"
    printf 'first boot loads the 125.8 GB checkpoint from /srv/llm and compiles\n'
    printf 'Triton/inductor kernels; check readiness with `just logs vllm-mxfp4` or\n'
    printf '`just compose ps`).\n'

# Stop just the vllm-mxfp4 service (leaves the other services running).
down-mxfp4:
    @{{compose}} down vllm-mxfp4

# Start the vllm-gptq service — the GPTQ ("AA2") checkpoint of the same model,
# as a separate service so switching checkpoints needs no compose edit and no
# cache clear (each has its own cache suffix). It resolves the GPTQ_VLLM_TAG
# image by tag with pull_policy: always, so every start re-pulls the newest
# build of that tag. Not intended to run concurrently with any other service:
# TP=4 wants all four GPUs and the PLE table pins ~116 GiB of host RAM.
up-gptq: check
    #!/usr/bin/env bash
    set -euo pipefail
    gptq_model="$(grep -m1 '^GPTQ_MODEL_DIR=' .env 2>/dev/null | cut -d= -f2- || true)"
    gptq_model="${gptq_model:-/srv/llm/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ}"
    if [ ! -f "$gptq_model/model.safetensors.index.json" ]; then
      printf 'error: model not found at %s\n' "$gptq_model" >&2
      printf 'Download it first (109 GB, resumable):\n' >&2
      printf '  hf download tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ \\\n' >&2
      printf '    --local-dir %s\n' "$gptq_model" >&2
      exit 1
    fi
    gptq_port="$(grep -m1 '^GPTQ_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    gptq_port="${gptq_port:-8004}"
    gptq_tag="$(grep -m1 '^GPTQ_VLLM_TAG=' .env 2>/dev/null | cut -d= -f2- || true)"
    gptq_tag="${gptq_tag:-29.02.1}"
    {{compose}} up -d vllm-gptq
    printf 'vllm-gptq starting at http://localhost:%s/v1 (re-pulled %s first).\n' "$gptq_port" "$gptq_tag"
    printf 'First boot loads the 109 GB checkpoint from /srv/llm and compiles\n'
    printf 'Triton/inductor kernels into a cold cache — can take 20+ min;\n'
    printf 'check readiness with `just logs vllm-gptq`.\n'

# Stop just the vllm-gptq service (leaves the other services running).
down-gptq:
    @{{compose}} down vllm-gptq

# Start the vllm-gptq-aa service — tcclaviger's own TP4 recipe for the "AA2"
# GPTQ build, reproduced as faithfully as possible on his current image, so any
# issue can be quoted back to him verbatim. It resolves the QFN_AA_VLLM_TAG
# image by tag with pull_policy: always, so every start re-pulls that tag.
# VLLM_PLE_CPU_OFFLOAD is gone from this image; its --engram-config host-RAM
# default pins via cudaHostRegister, which registers 0 bytes on this box and
# crashes. So this service runs --ple-nvme-offload: the PLE table comes from
# NVMe with a small RAM row cache instead of ~116 GiB of pinned host RAM. It
# still wants all four GPUs — one service at a time.
up-gptq-aa: check
    #!/usr/bin/env bash
    set -euo pipefail
    aa_model="$(grep -m1 '^QFN_AA_MODEL_DIR=' .env 2>/dev/null | cut -d= -f2- || true)"
    aa_model="${aa_model:-/srv/llm/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ-AA2}"
    if [ ! -f "$aa_model/model.safetensors.index.json" ]; then
      printf 'error: model not found at %s\n' "$aa_model" >&2
      printf 'Download it first (~115 GB, resumable):\n' >&2
      printf '  hf download tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ \\\n' >&2
      printf '    --local-dir %s\n' "$aa_model" >&2
      exit 1
    fi
    aa_port="$(grep -m1 '^QFN_AA_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    aa_port="${aa_port:-8006}"
    aa_tag="$(grep -m1 '^QFN_AA_VLLM_TAG=' .env 2>/dev/null | cut -d= -f2- || true)"
    aa_tag="${aa_tag:-29.04.15}"
    {{compose}} up -d vllm-gptq-aa
    printf 'vllm-gptq-aa starting at http://localhost:%s/v1 (re-pulled %s first).\n' "$aa_port" "$aa_tag"
    printf 'First boot loads the ~115 GB checkpoint from /srv/llm and compiles\n'
    printf 'Triton/inductor kernels into a cold cache — can take 20+ min;\n'
    printf 'check readiness with `just logs vllm-gptq-aa`.\n'

# Stop just the vllm-gptq-aa service (leaves the other services running).
down-gptq-aa:
    @{{compose}} down vllm-gptq-aa

# Download a radiance .rad container for the vllm-radiance-engine service
# (113.6 GiB either way). Resumable, so re-run after an interrupted transfer.
# Written to /srv/llm (the 1.3T volume), never ~/.cache -- see the disk-layout
# note in AGENTS.md. Takes ~35 min at ~57 MB/s.
#
# `which` is `served` (what we run today, and what radiance 2.x still serves:
# verified 2026-11-19 with rad-info + a GPU-less load under 2.2.0, which opened
# it as `arch=qwen4exp quant=bf16` and printed no format refusal) or `iqr-v4`
# (the newer IQR quantisation, which needs > 1.17.0 -- i.e. the 2.x line). They
# are different weight files, not a version bump: switching is a config edit
# (RADIANCE_ENGINE_MODEL) plus a restart, so keep both until one wins.
get-radiance-rad which="served":
    #!/usr/bin/env bash
    set -euo pipefail
    case '{{which}}' in
      served)
        repo='qwen3.8-next-flash-fp8-iq4r-moe'
        sha='' ;;
      iqr-v4)
        repo='qwen3.8-next-flash-iqr-v4'
        sha='25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0' ;;
      *)
        printf 'error: which must be `served` or `iqr-v4`, not `%s`.\n' '{{which}}' >&2
        exit 1 ;;
    esac
    dest_dir="/srv/llm/StillDeadcode"
    dest="$dest_dir/$repo.rad"
    mkdir -p "$dest_dir"
    printf 'Downloading 113.6 GiB to %s (resumes if interrupted)...\n' "$dest"
    # Sends the `hf auth login` token when present. A single long-lived HF
    # connection can decay to ~2 MB/s; interrupting and rerunning resumes on a
    # fresh one.
    auth=()
    tok="${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
    [ -n "$tok" ] && auth=(-H "Authorization: Bearer $tok")
    curl -L -C - --retry 10 --retry-delay 5 "${auth[@]}" -o "$dest" \
      "https://huggingface.co/StillDeadcode/$repo/resolve/main/$repo.rad"
    if [ -n "$sha" ]; then
      printf 'Checking sha256 (the model card publishes one; ~2 min over 113 GiB)...\n'
      echo "$sha  $dest" | sha256sum -c -
    fi
    printf 'Verifying it is a readable .rad container...\n'
    {{compose}} run --rm --no-deps --entrypoint rad-info vllm-radiance-engine "$dest"

# Render the radiance ENGINE's serve config.
#
# radiance 2.0.0 removed every server CLI flag: the binary takes only
# `--config FILE` (plus `--check`). compose.yaml therefore holds no flags for
# this service, and `.env` keeps driving it through RADIANCE_ENGINE_* -- this
# recipe is the one place that turns those into the YAML the engine reads, at
# ${RADIANCE_ENGINE_STATE}/radiance.yaml, bind-mounted read-only by compose.
#
# Re-run it after any .env edit (up-radiance-engine does it for you) and check
# the result with `just radiance-check`, which is the engine's own validator and
# allocates nothing. Comments below ride into the generated file on purpose: it
# is what someone on the host reads when the service misbehaves.
radiance-config:
    #!/usr/bin/env bash
    set -euo pipefail
    ev() { local v; v="$(grep -m1 "^RADIANCE_ENGINE_$1=" .env 2>/dev/null | cut -d= -f2- || true)"; printf '%s' "${v:-$2}"; }

    tag="$(ev TAG 2.2.0)"
    case "$tag" in
      1.*)
        printf 'error: radiance %s takes flags, not --config; this renderer is for 2.x.\n' "$tag" >&2
        printf 'To stay on 1.x: revert the vllm-radiance-engine `command:` block in\n'
        printf 'compose.yaml to its flag list (git show HEAD~1:compose.yaml has it).\n' >&2
        exit 1 ;;
    esac

    tp="$(ev TP 4)"
    # host-pool-mib is a TOTAL across the ranks in 2.x ("pinned host tier for
    # weights, in all": core/config.cpp divides it by tp). The flag meant
    # per-rank up to 1.17, so .env keeps the per-rank number we tuned and this
    # is the one place that converts it. Under-provisioning here is silent: the
    # experts just spill further down the tiers.
    pool_per_rank="$(ev HOST_POOL_MIB 6144)"
    pool_total="$(( pool_per_rank * tp ))"
    model="$(ev MODEL /srv/llm/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe.rad)"
    state="$(ev STATE /srv/llm/radiance)"
    out="$state/radiance.yaml"
    mkdir -p "$state"

    cat > "$out.new" <<EOF
    # GENERATED by \`just radiance-config\` -- edit .env (RADIANCE_ENGINE_*), not here.
    # Schema: radiance 2.x reads exactly one YAML file, no flags. \`radiance --config
    # this-file --check\` (\`just radiance-check\`) validates it without loading.
    server:
      host: 0.0.0.0
      # The engine's own port, INSIDE the container; compose maps it to
      # RADIANCE_ENGINE_PORT on the host.
      port: 8000
    memory:
      # VRAM held back at boot for what the budget cannot count: a kernel's code
      # object and scratch (sized on first launch), plugin hipMalloc, and whatever
      # another process takes between the card read and the allocations landing.
      # Shortfall shows up as an OOM mid-request, on whichever request triggers an
      # unseen code path. The engine says what it needs on the
      # \`settled at ... against ...\` line and raises the margin one for one.
      gpu-headroom-mib: $(ev GPU_HEADROOM_MIB 112)
    models:
      # The entry's name IS the served model id (2.x dropped --served-model-name);
      # add \`aliases:\` for more. This is what the LiteLLM gateway calls.
      $(ev SERVED_NAME qwen3.8-flash-next-rad):
        model: $model
        tp: $tp
        # \`exact\` = upstream's 4-rank row. \`wht6\` at 4 ranks is a lossy tiered-int8
        # all-reduce wire (at 2 ranks it is rotated 6-bit); A/B it deliberately.
        tp-wire: $(ev TP_WIRE exact)
        p2p: $(ev P2P auto)
        # 524288 = YaRN factor 2 over the container's trained 262144, applied at
        # LOAD (no reconversion). The factor is where upstream's retrieval numbers
        # were measured (docs/YARN.md). Changing it invalidates the on-disk prefix
        # cache, which is rotary-fingerprinted.
        rope-scaling: "yarn:$(ev YARN_FACTOR 2):$(ev ROPE_ORIG 262144)"
        max-model-len: $(ev MAX_MODEL_LEN 524288)
        # 8, not upstream's 32: Flash-Next is hybrid, so every sequence carries its
        # own recurrent state, and upstream keeps Flash-Next at 8 for exactly that
        # reason -- 32 of them take the memory back from the attention cache and
        # from the expert slab.
        max-num-seqs: $(ev MAX_NUM_SEQS 8)
        max-num-batched-tokens: $(ev MAX_NUM_BATCHED_TOKENS 2048)
        # PREFILL PACING. Default: every step carries a prefill chunk while one is
        # queued, so decode is paced at batch/step-latency (~2048-token chunk ~= 230
        # ms -> ~10-35 t/s aggregate during a prefill storm). prefill-interleave=N
        # runs up to N decode-only steps after each chunk (max 64; 0 = the old
        # every-step behaviour). A chunk that has other work in the step is capped at
        # busy-prefill-chunk tokens.
        #
        # A prefill step streams every routed layer's non-resident experts onto the
        # card, and the stager hides that PCIe transfer behind the step's compute
        # only at steps of >= 1025 tokens: below it, a held-back chunk reads those
        # experts synchronously, which costs more than the smaller step saves. So
        # busy-prefill-chunk must be >= 1025, and <= max-num-batched-tokens; decode
        # still gets what is left of the step budget.
        prefill-interleave: $(ev PREFILL_INTERLEAVE 8)
        busy-prefill-chunk: $(ev BUSY_PREFILL_CHUNK 1536)
        # i8, NOT fp8: an E4M3 KV cache is one plugin's (RadArchProbe::kv_fp8_ok) and
        # the qwen4exp plugin does not declare it, so 2.x refuses \`kv-cache-dtype:
        # fp8\` at boot for this model. i8 (int8 K and V, a scale per token and head)
        # is the half-size cache now, and it costs MORE per token than the packed
        # fp8_v4/k6_v4 caches of 1.x -- so the pool holds fewer tokens at a given
        # window. Read the \`kv:\` line of the boot log: it says what share of
        # max-num-seqs at max-model-len the pool backs, and lower one of the two (or
        # raise host-pool-mib) if it comes in under 100%.
        kv-cache-dtype: $(ev KV_DTYPE i8)
        placement: $(ev PLACEMENT expert_tiered)
        # TOTAL over the $tp ranks = $pool_per_rank per rank (see the note above).
        host-pool-mib: $pool_total
        # 47.7 GiB n-gram PLE table; auto tries vram, then ram, then disk.
        ngram-placement: $(ev NGRAM_PLACEMENT auto)
        num-speculative-tokens: $(ev NUM_SPEC_TOKENS 3)
        # Two-tier prefix cache (VRAM -> host RAM). memory.disk-tier is back in 2.x
        # and off unless asked for; the 1.x disk tier's leftovers under /data are
        # dead weight. ~4096 MiB parked a 1.x fp8 cache's worth of ~143k tokens; i8
        # costs more per token, so re-read the boot log's prefix-cache line here too.
        prefix-cache-mib: $(ev PREFIX_CACHE_MIB 24576)
        # exact = resume only where a cold prefill would start a chunk (bit-identical
        # to cold); nearest = resume at the deepest snapshot, re-prefills up to a
        # chunk less. Upstream notes exactness only holds on an IDLE server, so on a
        # shared one exact buys less than it costs -- which is why .env sets nearest.
        prefix-resume: $(ev PREFIX_RESUME exact)
        # Same vetted froggeric template as every other service here, in place of the
        # one baked into the .rad. qwen.jinja maps OpenAI's \`high\` (which this
        # template family lacks) to \`low\` rather than raising; the map below is the
        # server-side override of that, kept so a client asking for \`high\` gets the
        # xhigh thinking budget the agentic clients expect.
        override-chat-template: /app/chat-templates/qwen.jinja
        reasoning-effort-map: $(ev REASONING_EFFORT_MAP high=xhigh)
        # The loop symptom from CLAUDE.md's open issue: this repo's other Flash-Next
        # services carry presence_penalty=1.0 as a mitigation, so mirror it here to
        # keep the comparison honest. (docs attribute empty/length-capped output to
        # reasoning-effort, not to penalties -- that lead is still open.)
        presence-penalty: $(ev PRESENCE_PENALTY 1.0)
    EOF

    # Optional keys, emitted only when .env asks for them: every one below is a
    # 2.x addition, so the default file stays exactly the shape --check passed.
    # The server- and scheduler-level ones belong in their own section, so they
    # are spliced in rather than appended.
    ka="$(grep -m1 '^RADIANCE_ENGINE_KEEP_ALIVE_TIMEOUT=' .env 2>/dev/null | cut -d= -f2- || true)"
    if [ -n "$ka" ]; then
      out_tmp="$out.new.tmp"
      awk -v add="  keep-alive-timeout: $ka" '/^  port:/{print;print add;next} {print}' "$out.new" > "$out_tmp"
      mv "$out_tmp" "$out.new"
    fi
    sched=""
    for pair in "KEEPALIVE:keepalive" "STUCK_EXIT:stuck-exit" "MIXED_STEP_TARGET_MS:mixed-step-target-ms" "PREFILL_ROOM_GATE:prefill-room-gate" "PREEMPT_WAIT_MS:preempt-wait-ms"; do
      k="${pair%%:*}"; y="${pair##*:}"
      v="$(grep -m1 "^RADIANCE_ENGINE_$k=" .env 2>/dev/null | cut -d= -f2- || true)"
      [ -n "$v" ] && sched="${sched}  ${y}: ${v}"$'\n'
    done
    if [ -n "$sched" ]; then
      out_tmp="$out.new.tmp"
      awk -v add="$sched" 'BEGIN{done=0} /^models:/{if(!done){print "scheduler:";printf "%s",add;done=1}} {print}' "$out.new" > "$out_tmp"
      mv "$out_tmp" "$out.new"
    fi
    for pair in "STATE_CACHE_DTYPE:state-cache-dtype" "DRAFT_COUPLING:draft-coupling" "DRAFT_VERIFY:draft-verify" "DRAFT_DEPTH:draft-depth" "DRAFT_DEPTH_LOAD:draft-depth-load" "DRAFT_SUFFIX:draft-suffix" "PREFILL_SPAN:prefill-span" "PREFILL_SPAN_PASS:prefill-span-pass" "PREFILL_SPAN_MIN:prefill-span-min" "PREFILL_SPAN_FIRST:prefill-span-first"; do
      k="${pair%%:*}"; y="${pair##*:}"
      v="$(grep -m1 "^RADIANCE_ENGINE_$k=" .env 2>/dev/null | cut -d= -f2- || true)"
      [ -n "$v" ] && printf '    %s: %s\n' "$y" "$v" >> "$out.new"
    done
    # A sequence, not a string: the engine reads each entry as one id and refuses
    # any alias holding ':' or whitespace, so a quoted "[a, b]" is one bad alias.
    aliases="$(grep -m1 '^RADIANCE_ENGINE_ALIASES=' .env 2>/dev/null | cut -d= -f2- || true)"
    if [ -n "$aliases" ]; then
      printf '    aliases:\n' >> "$out.new"
      printf '%s\n' "$aliases" | tr ',' '\n' | sed 's/^ *//; s/ *$//; /^$/d' \
        | while read -r a; do printf '      - %s\n' "$a"; done >> "$out.new"
    fi

    mv "$out.new" "$out"
    printf 'wrote %s (radiance %s, tp %s, host-pool %s MiB total)\n' "$out" "$tag" "$tp" "$pool_total"
    printf 'Validate it without loading anything: `just radiance-check`\n'

# Validate the rendered config with the engine itself. `--check` reads and
# validates the file and starts nothing -- it opens no card, so this is safe to
# run while a live engine holds all the VRAM.
radiance-check:
    #!/usr/bin/env bash
    set -euo pipefail
    state="$(grep -m1 '^RADIANCE_ENGINE_STATE=' .env 2>/dev/null | cut -d= -f2- || true)"
    state="${state:-/srv/llm/radiance}"
    [ -f "$state/radiance.yaml" ] || { printf 'render it first: `just radiance-config`\n' >&2; exit 1; }
    {{compose}} run --rm --no-deps --entrypoint /opt/radiance/bin/radiance vllm-radiance-engine \
      --config /etc/radiance/radiance.yaml --check

# The prefix cache's disk tier lives under ${RADIANCE_ENGINE_STATE}:/data and
# the container runs as the host user, so the dir must be user-owned or the
# engine cannot write it. Mirrors ensure-cache-dirs, for the one service whose
# state is NOT under ~/.cache.
ensure-radiance-state:
    #!/usr/bin/env bash
    set -euo pipefail
    state="$(grep -m1 '^RADIANCE_ENGINE_STATE=' .env 2>/dev/null | cut -d= -f2- || true)"
    state="${state:-/srv/llm/radiance}"
    mkdir -p "$state/radiance/prefix-cache"
    if [ "$(stat -c '%u' "$state")" != "$(id -u)" ]; then
      printf '%s is not owned by %s; fixing ownership (one-time sudo).\n' "$state" "$(id -un)" >&2
      sudo chown -R "$(id -u):$(id -g)" "$state"
    fi
    printf 'radiance state dir ready: %s (owner: %s)\n' "$state" "$(id -un)"

# Start the radiance ENGINE service (stilldeadcode/radiance -- NOT the
# vllm-radiance vLLM service on 8002). Pulls the pinned tag, checks the .rad is
# in place, then starts. Not concurrent with anything else: TP=4 wants all four
# cards and the host pool pins 4 x 6 GiB of RAM.
up-radiance-engine: check ensure-radiance-state radiance-config
    #!/usr/bin/env bash
    set -euo pipefail
    rad="$(grep -m1 '^RADIANCE_ENGINE_MODEL=' .env 2>/dev/null | cut -d= -f2- || true)"
    rad="${rad:-/srv/llm/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe.rad}"
    if [ ! -f "$rad" ]; then
      printf 'error: %s not found.\n' "$rad" >&2
      printf 'Download it first (113.6 GiB, resumable): `just get-radiance-rad served`\n' >&2
      printf '(the newer IQR quant is `just get-radiance-rad iqr-v4`, and needs\n'
      printf ' RADIANCE_ENGINE_MODEL pointed at it plus radiance >= 2.0)\n' >&2
      exit 1
    fi
    printf 'Free host RAM: %s GiB (the host pool pins 24 GiB at the defaults)\n' \
      "$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
    {{compose}} pull vllm-radiance-engine
    {{compose}} up -d vllm-radiance-engine
    port="$(grep -m1 '^RADIANCE_ENGINE_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    printf 'vllm-radiance-engine starting at http://localhost:%s/v1\n' "${port:-8007}"
    printf 'First boot loads 113.6 GiB across four cards; watch the log for the\n'
    printf '`rope:` line (should say YaRN factor 2 -- the rotary reaches 524288),\n'
    printf 'the `elastic` line (re-derive RADIANCE_ENGINE_HOST_POOL_MIB from it),\n'
    printf 'and the `kv:` line (i8 costs more per token than 1.x fp8 did -- it\n'
    printf 'says what share of max-num-seqs x max-model-len the pool actually backs).\n'
    printf 'Config lives in .env: edit it, then `just radiance-config` + restart.\n'
    printf 'Readiness: `just logs vllm-radiance-engine` or `just compose ps`.\n'

# Stop just the vllm-radiance-engine service (leaves the others running).
down-radiance-engine:
    @{{compose}} down vllm-radiance-engine

# Build the image-generation image (Dockerfile.imagegen). Thin: it reuses the
# ROCm/torch base from Dockerfile.fullbuild's framework-base stage but installs
# diffusers (from git at DIFFUSERS_REF) + transformers instead of vLLM. Separate
# image, so building it never rebuilds (or is rebuilt by) the LLM images.
build-imagegen: check
    @{{compose}} build image-gen

# Start the image-gen service (Qwen-Image-2.1 text-to-image via diffusers, NOT
# vLLM). The ~32 GB of weights fill a whole R9700 and the default device_map
# placement spreads them over all four, so this is not meant to run alongside
# any LLM service — stop one first (`just down` / `just down-gptq`), same
# one-at-a-time rule. First boot loads the checkpoint from /srv/llm.
up-imagegen: check
    #!/usr/bin/env bash
    set -euo pipefail
    imagegen_model="$(grep -m1 '^IMAGEGEN_MODEL_DIR=' .env 2>/dev/null | cut -d= -f2- || true)"
    imagegen_model="${imagegen_model:-/srv/llm/Qwen/Qwen-Image-2.1}"
    if [ ! -f "$imagegen_model/model_index.json" ]; then
      printf 'error: model not found at %s\n' "$imagegen_model" >&2
      printf 'Download it first (33.1 GB, resumable):\n' >&2
      printf '  hf download Qwen/Qwen-Image-2.1 --local-dir %s\n' "$imagegen_model" >&2
      exit 1
    fi
    imagegen_port="$(grep -m1 '^IMAGEGEN_PORT=' .env 2>/dev/null | cut -d= -f2- || true)"
    imagegen_port="${imagegen_port:-8005}"
    {{compose}} up -d image-gen
    printf 'image-gen starting at http://localhost:%s/generate (and /v1/images/generations).\n' "$imagegen_port"
    printf 'First boot loads ~32 GB of weights across the GPUs; poll readiness with\n'
    printf '`just logs image-gen` (GET /health returns 503 until the pipeline is resident).\n'

# Stop just the image-gen service (leaves the other services running).
down-imagegen:
    @{{compose}} down image-gen

# Run a command inside the running vLLM container (e.g. `just exec bash`).
exec *args:
    @{{compose}} exec vllm {{args}}

# `just logs` follows all services; `just logs vllm-qwen-flashnext` one of them.
logs *args:
    @{{compose}} logs -f {{args}}

# Raw compose passthrough for service-level ops
# (e.g. `just compose up -d vllm`, `just compose ps`, `just compose down vllm-qwen-flashnext`).
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
