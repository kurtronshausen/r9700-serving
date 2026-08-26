"""Drive the full GEMM-config tuning for the active model profile.

Runs inside the throwaway `just tune` container (same image as the server).
Two stages:

1. Dense w8a8 block-FP8: tunes every shape in $DENSE_SHAPES ("N:K,N:K") via
   tune_fp8_dense.py. The shapes are the per-GPU GEMM geometries the server
   reported as "Config file not found" at startup (see the `just tune`
   recipe), so this stage is a no-op when nothing is missing.
2. Fused MoE: for MoE profiles only, derives the per-GPU geometry at the
   profile's TP (E = global experts — vLLM replicates experts across TP
   ranks; N = moe_intermediate_size / TP; K = hidden_size) and tunes via
   tune_fused_moe.py.

Outputs land in /app/fp8_configs and /app/fused_moe_configs (repo dirs
mounted rw by compose); the next `just up` picks them up.
"""

import json
import os
import subprocess
import sys

REPO = os.environ.get("REPO_DIR", "/app")
DENSE_SHAPES = os.environ.get("DENSE_SHAPES", "")


def read_text_config():
    model = os.environ.get("VLLM_MODEL", "")
    if not model:
        print("VLLM_MODEL unset; skipping MoE stage")
        return None
    if os.path.isdir(model):
        path = os.path.join(model, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def main():
    if DENSE_SHAPES:
        print(f"== dense w8a8 block-FP8 tuning: {DENSE_SHAPES} ==")
        subprocess.run(
            [sys.executable, f"{REPO}/tools/tune_fp8_dense.py",
             "--shapes", DENSE_SHAPES],
            check=True,
        )
    else:
        print("== dense w8a8 block-FP8: no shapes; nothing to tune ==")

    print("== fused MoE check ==")
    tc = read_text_config()
    if tc is None:
        return
    experts = tc.get("num_experts", 0)
    if not experts:
        print("no MoE experts in this profile; skipping MoE tuning")
        return
    tp = int(os.environ.get("VLLM_TP", "2"))
    moe = {
        "E": experts,
        "N": tc["moe_intermediate_size"] // tp,
        "K": tc["hidden_size"],
        "topk": tc.get("num_experts_per_tok", 8),
    }
    print(f"tuning fused MoE at TP={tp}: {moe}")
    subprocess.run(
        [sys.executable, f"{REPO}/tools/tune_fused_moe.py",
         "--E", str(moe["E"]), "--N", str(moe["N"]),
         "--K", str(moe["K"]), "--topk", str(moe["topk"])],
        check=True,
    )
    print("== tuning complete: run `just up` to pick up the new configs ==")


if __name__ == "__main__":
    main()
