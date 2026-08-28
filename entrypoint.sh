#!/bin/sh
#
# Container entrypoint: assembles the `vllm serve` invocation from the
# service's environment (the compose env_file stack:
# 2xr9700.vllm.common -> aiter-unified-attention.env -> qwen3.6.env.common ->
# <profile>.env) and execs it.
#
# Compose services set no `command:` — every flag is driven by env vars so the
# same image/entrypoint serves every profile, and each container expands its
# OWN profile's values at startup (compose-time interpolation can't distinguish
# the profiles of two services in one file). Missing required vars abort with
# a message instead of starting a server with no model. Extra arguments are
# ignored.
#
# Defaults in "${VAR:-default}" below mirror the previous compose.yaml
# interpolation defaults. The max-num-seqs default of 2 is the universal
# vllm-project/vllm#35288 cap (MTP spec-decode corrupts output at 4+ decode
# sequences); do not raise until the upstream bug is fixed.
#
# --limit-mm-per-prompt image cap is 1 (not the vLLM default): 2+ large images
# in one hybrid-Mamba prompt deadlocks the engine (vllm-project/vllm#40707,
# unfixed — #40709 not merged). Raise only if that lands upstream.
#
# --no-async-scheduling tracks VLLM_SPEC_DECODE: on vLLM v0.28.0+ async
# scheduling resolves to ON by default when the spec method is MTP, but
# async+MTP+hybrid-GDN+align has an open accepted-count race
# (vllm-project/vllm#51571) and vLLM's own ROCm CI disables async for MTP
# pending an unroot-caused hang (#54039/#32275). Non-MTP profiles keep async
# scheduling (decode gap overlap).
#
# PREFIX_CACHE_RETENTION_INTERVAL (set per-profile, not a vLLM-managed env
# var name — see the note in env/qwen3.8-27b.env) pins
# --prefix-cache-retention-interval to the profile's attention block size.
# Only takes effect on vLLM v0.28.1rc0+ (#52216 introduced the flag); a no-op
# on older builds. Restores dense Mamba retention that rc0's sparse default
# would otherwise drop under MTP (vllm-project/vllm#53504).

set -eu

# (No backticks in the :? messages: the word is subject to command
# substitution in POSIX sh.)
: "${VLLM_MODEL:?set in env/<profile>.env - start the server with: just up}"
: "${VLLM_TOKENIZER:?set in env/<profile>.env - start the server with: just up}"
: "${VLLM_SERVED_NAME:?set in env/<profile>.env - start the server with: just up}"

# VLLM_TOOL_CHOICE and VLLM_EXTRA_ARGS are intentionally unquoted: they hold
# argument strings (e.g. "--enable-expert-parallel") that must word-split
# into separate flags.
exec vllm serve \
    "$VLLM_MODEL" \
    --tokenizer "$VLLM_TOKENIZER" \
    ${VLLM_MODEL_REVISION:+--revision "$VLLM_MODEL_REVISION"} \
    --served-model-name "$VLLM_SERVED_NAME" \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1, "audio": 0, "video": 0}' \
    ${VLLM_CHAT_TEMPLATE:+--chat-template "$VLLM_CHAT_TEMPLATE"} \
    ${VLLM_TOOL_CHOICE:-} \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-131072}" \
    --max-num-batched-tokens "${VLLM_MAX_BATCHED_TOKENS:-8192}" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}' \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.95}" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-2}" \
    --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE:-bfloat16}" \
    ${VLLM_QUANTIZATION:+--quantization "$VLLM_QUANTIZATION"} \
    -tp "${VLLM_TP:-2}" \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    ${VLLM_SPEC_DECODE:+--speculative-config "$VLLM_SPEC_DECODE"} \
    ${VLLM_SPEC_DECODE:+--no-async-scheduling} \
    ${PREFIX_CACHE_RETENTION_INTERVAL:+--prefix-cache-retention-interval "$PREFIX_CACHE_RETENTION_INTERVAL"} \
    ${VLLM_EXTRA_ARGS:-}
