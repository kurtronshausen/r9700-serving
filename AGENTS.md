# AGENTS

## Rules

- **Never commit, push, open PRs, or make any remote changes targeting `andysalerno/r9700-serving` (remote: `upstream`/`andysalerno`) without explicit, direct instructions from the user — and confirm intent before doing so.**
- Always target `origin` (your own fork) for commits, pushes, and PRs unless explicitly told otherwise.

## Tools

Use `just` for all build/run workflows. Commands are defined in `justfile`.

| recipe | purpose |
|:-------|:--------|
| `just build` | build the Docker image |
| `just rebuild` | force-rebuild (no cache) |
| `just up` | start the vLLM server (sources env, starts container, waits for readiness, runs warmup) |
| `just logs` | follow container logs (`docker compose logs -f`) |
| `just down` | stop and remove the container |
| `just exec <cmd>` | run a command inside the running container (e.g. `just exec bash`) |
| `just chat <prompt>` | send a chat completion to the running server |
| `just clear-vllm-caches` | wipe compile caches (triton, torchinductor, aiter, etc.) |

To switch models, set `MODEL_PROFILE` or use `--set`:

```
MODEL_PROFILE=qwen3.6-27b just up
just --set model qwen3.6-27b up
```
