"""Drive the full GEMM-config tuning for the active model profile.

Runs inside the throwaway `just tune` container (same image as the server).
Two stages:

1. Dense w8a8 block-FP8: tunes every shape in $DENSE_SHAPES ("N:K,N:K") via
   tune_fp8_dense.py. The shapes are the per-GPU GEMM geometries the server
   reported as "Config file not found" at startup (see the `just tune`
   recipe), so this stage is a no-op when nothing is missing.
2. Fused MoE: for MoE profiles only, derives the per-GPU geometry and tunes
   via tune_fused_moe.py (fp8_w8a8 profiles) or tune_fused_moe_int4.py
   (int4_w4a16 profiles, auto-detected from the model's quantization_config).
   TP-sliced profiles (the default) divide moe_intermediate_size by
   VLLM_TP — vLLM replicates experts across TP ranks and slices the
   intermediate dim. Expert-parallel profiles (VLLM_EXTRA_ARGS containing
   `--enable-expert-parallel`, e.g. flashnext) instead slice E (global
   experts / EP-rank-count == TP) and leave moe_intermediate_size whole,
   since EP replicates the intermediate dim per-rank rather than
   tensor-slicing it.

Outputs land in /app/fp8_configs and /app/fused_moe_configs (repo dirs
mounted rw by compose); the next `just up` picks them up.
"""

import json
import os
import subprocess
import sys

REPO = os.environ.get("REPO_DIR", "/app")
DENSE_SHAPES = os.environ.get("DENSE_SHAPES", "")


def read_full_config():
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
    return cfg


def detect_quant(cfg):
    """Return ("int4_w4a16", group_size) or ("fp8_w8a8", None)."""
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get(
        "quantization_config"
    )
    if not qc:
        return "fp8_w8a8", None
    groups = qc.get("config_groups", {})
    weights = next(iter(groups.values()), {}).get("weights", {}) if groups else {}
    num_bits = weights.get("num_bits")
    fmt = qc.get("format", "")
    if num_bits == 4 and fmt == "pack-quantized":
        return "int4_w4a16", weights.get("group_size", 128)
    return "fp8_w8a8", None


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
    cfg = read_full_config()
    if cfg is None:
        return
    tc = cfg.get("text_config", cfg)
    experts = tc.get("num_experts", 0)
    if not experts:
        print("no MoE experts in this profile; skipping MoE tuning")
        return

    tp = int(os.environ.get("VLLM_TP", "2"))
    extra_args = os.environ.get("VLLM_EXTRA_ARGS", "")
    expert_parallel = "--enable-expert-parallel" in extra_args
    dtype, group_size = detect_quant(cfg)

    if expert_parallel:
        # EP: experts are sliced across ranks, intermediate dim stays whole.
        moe = {
            "E": experts // tp,
            "N": tc["moe_intermediate_size"],
            "K": tc["hidden_size"],
            "topk": tc.get("num_experts_per_tok", 8),
        }
        print(f"tuning fused MoE (expert-parallel, EP={tp}): {moe}")
    else:
        # TP: experts replicated per rank, intermediate dim sliced by TP.
        moe = {
            "E": experts,
            "N": tc["moe_intermediate_size"] // tp,
            "K": tc["hidden_size"],
            "topk": tc.get("num_experts_per_tok", 8),
        }
        print(f"tuning fused MoE (tensor-parallel, TP={tp}): {moe}")

    if dtype == "int4_w4a16":
        print(f"detected int4_w4a16 quantization (group_size={group_size})")
        cmd = [sys.executable, f"{REPO}/tools/tune_fused_moe_int4.py",
               "--E", str(moe["E"]), "--N", str(moe["N"]),
               "--K", str(moe["K"]), "--topk", str(moe["topk"]),
               "--group-size", str(group_size)]
    else:
        cmd = [sys.executable, f"{REPO}/tools/tune_fused_moe.py",
               "--E", str(moe["E"]), "--N", str(moe["N"]),
               "--K", str(moe["K"]), "--topk", str(moe["topk"])]
    subprocess.run(cmd, check=True)
    print("== tuning complete: run `just up` to pick up the new configs ==")


if __name__ == "__main__":
    main()
