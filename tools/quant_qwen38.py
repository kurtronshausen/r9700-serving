#!/usr/bin/env python3
"""RTN W4A16 quantizer for Qwen3.8-27B-FP8 (compressed-tensors format).

Dequantizes the source's block-FP8 (e4m3 + 128x128 weight_scale_inv) linear
weights and re-quantizes them to symmetric int4 / group-128 in the exact
layout vLLM's compressed-tensors WNA16 path expects (same format as the
flashnext RTN checkpoint):

    name.weight (F8_E4M3 [N, K]) + name.weight_scale_inv (BF16 [N/128, K/128])
        -> name.weight_packed  I32  [N, K/8]
        -> name.weight_scale   BF16 [N, K/128]
        -> name.weight_shape   I32  [2]  (= [N, K])

Quantized: self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_qkv,in_proj_z,
out_proj}, mlp.{gate,up,down}_proj — main layers only (mtp.* stays BF16 via
the ignore list, mirroring the flashnext checkpoint).

Copied unchanged: all BF16 tensors (embeds, norms, GDN A_log/dt_bias/conv1d/
in_proj_a/b, k_norm/q_norm, visual.*, mtp.*), vocab files, chat template.

CPU-only, streams source shards (peak RAM a few GiB), resumable. Run inside
the flashnext image (torch + safetensors):

    docker run --rm -v ~/.cache/huggingface:/hf \
               -v /srv/llm/kurt:/srv/llm/kurt \
               -v $PWD:/workspace localhost/vllm-flashnext:latest \
      python3 /workspace/tools/quant_qwen38.py \
        /hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/<hash> \
        /srv/llm/kurt/qwen3.8-27b-w4a16-rtng128
"""

import argparse
import json
import mmap
import os
import re
import shutil
import struct
import sys
import time

import torch
from safetensors.torch import save_file

GROUP_SIZE = 128
NUM_BITS = 4
BIAS = 7  # uint4b8
PACK = 32 // NUM_BITS  # 8 values per int32
# F8_E4M3 block scale in the source checkpoint (HuggingFace fp8 convention:
# weight_scale_inv has shape [N/128, K/128], value = 1/scale).
FP8_BLOCK = 128

# name.weight that gets W4-quantized (main layers only).
# PHASE A: full-attn q/k/v/o + MLP only. GDN (linear_attn) projections stay
# BF16: W4A16 G128 on the recurrent GDN path degenerates the model
# ("terastera" loop, 2026-08-28) — the recurrent state amplifies the ~12%
# per-layer rel-RMS.
QUANT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.("
    r"self_attn\.(?:q|k|v|o)_proj"
    r"|mlp\.(?:gate|up|down)_proj"
    r")\.weight$"
)
# config.json "ignore" list: same family as the flashnext RTN checkpoint,
# minus the MoE-specific entries, plus GDN small-state tensors.
IGNORE_LIST = [
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*mtp\\..*",
    "re:.*visual\\..*",
    "re:.*linear_attn\\.(?:A_log|dt_bias|conv1d|in_proj_a|in_proj_b|norm)$",
    "re:.*self_attn\\.(?:q_norm|k_norm)$",
]
QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": GROUP_SIZE,
                "num_bits": NUM_BITS,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        }
    },
    "format": "pack-quantized",
    "global_compression_ratio": None,
    "kv_cache_scheme": None,
    "ignore": IGNORE_LIST,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
}
COPY_FILES = [
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processing_utils.json",
    "LICENSE",
    "README.md",
]

_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "I8": torch.int8,
}


def parse_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return 8 + hlen, header


def rtng128(w: torch.Tensor):
    """Symmetric RTN int4 / group-128. Returns (packed I32 [N, K/8],
    scale BF16 [N, K/128])."""
    assert w.dim() == 2 and w.shape[1] % GROUP_SIZE == 0
    N, K = w.shape
    wf = w.to(torch.float32)
    g = wf.reshape(N, K // GROUP_SIZE, GROUP_SIZE)
    scale = g.abs().amax(dim=2).clamp_min(1e-8) / BIAS  # [N, K/128]
    q = (torch.round(g / scale.unsqueeze(2)) + BIAS).clamp(0, 2 * BIAS)
    q8 = q.reshape(N, -1).to(torch.int32)  # [N, K]
    words = torch.zeros(N, K // PACK, dtype=torch.int32)
    for i in range(PACK):
        words |= (q8[:, i::PACK] & 0xF) << (4 * i)
    return words, scale.to(torch.bfloat16)


def fp8_dequant(w8: torch.Tensor, scale_inv: torch.Tensor):
    """Block-128 dequant: w = w8 * scale_inv tiled over 128x128 blocks."""
    N, K = w8.shape
    assert N % FP8_BLOCK == 0 and K % FP8_BLOCK == 0, (N, K)
    assert scale_inv.shape == (N // FP8_BLOCK, K // FP8_BLOCK), (
        scale_inv.shape, N, K)
    wb = w8.to(torch.float32).reshape(N // FP8_BLOCK, FP8_BLOCK,
                                      K // FP8_BLOCK, FP8_BLOCK)
    si = scale_inv.to(torch.float32).reshape(N // FP8_BLOCK, 1, K // FP8_BLOCK, 1)
    return (wb * si).reshape(N, K)


class ShardWriter:
    def __init__(self, dst_dir, max_bytes):
        self.dst_dir = dst_dir
        self.max_bytes = max_bytes
        self.idx = 0
        self.cur = {}
        self.cur_bytes = 0

    def _path(self, i):
        return os.path.join(self.dst_dir, f"model-{i:05d}.safetensors")

    def add(self, name, tensor):
        self.cur[name] = tensor
        self.cur_bytes += tensor.numel() * tensor.element_size()
        if self.cur_bytes > self.max_bytes:
            self.flush()

    def flush(self):
        if not self.cur:
            return
        save_file(dict(self.cur), self._path(self.idx),
                  metadata={"format": "pt"})
        print(f"  wrote {os.path.basename(self._path(self.idx))}", flush=True)
        self.idx += 1
        self.cur = {}
        self.cur_bytes = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    args = ap.parse_args()

    src, dst = args.src_dir, args.dst_dir
    os.makedirs(dst, exist_ok=True)
    state_path = os.path.join(dst, ".quant_state.json")
    index_path = os.path.join(dst, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print("index.json already present — checkpoint complete, nothing to do")
        return

    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    # Pass 1: collect the tensors to dequant so each weight+scale pair can be
    # matched within one shard (they are co-located per layer file).
    total = 0
    for sh in shards:
        _, h = parse_header(os.path.join(src, sh))
        total += len(h) - 1
    print(f"source: {len(shards)} shards, {total} tensors")

    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"resuming: {len(state['done_src'])}/{len(shards)} shards done")
    else:
        state = {"done_src": [], "out_counter": 0, "n_quant": 0}

    if not os.path.exists(os.path.join(dst, "tokenizer_config.json")):
        for fn in COPY_FILES:
            s = os.path.join(src, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))
        with open(os.path.join(src, "config.json")) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = QUANT_CONFIG
        with open(os.path.join(dst, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("copied static files + wrote quantization_config")

    max_bytes = int(args.shard_gb * 1e9)
    w = ShardWriter(dst, max_bytes)
    w.idx = state["out_counter"]
    t0 = time.time()

    for sh in shards:
        if sh in state["done_src"]:
            continue
        path = os.path.join(src, sh)
        data_off, header = parse_header(path)
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                dtype, shape, off = meta["dtype"], meta["shape"], meta["data_offsets"]
                raw = mm[data_off + off[0]: data_off + off[1]]
                t = torch.frombuffer(raw, dtype=_DTYPE_MAP[dtype]).reshape(shape)
                scale_name = name[:-len(".weight")] + ".weight_scale_inv"
                if QUANT_RE.match(name):
                    scale_inv = torch.frombuffer(
                        mm[data_off + header[scale_name]["data_offsets"][0]:
                        data_off + header[scale_name]["data_offsets"][1]],
                        dtype=torch.bfloat16).reshape(header[scale_name]["shape"])
                    wf = fp8_dequant(t, scale_inv)
                    packed, scale = rtng128(wf)
                    w.add(name.replace(".weight", ".weight_packed"), packed)
                    w.add(name.replace(".weight", ".weight_scale"), scale)
                    w.add(name.replace(".weight", ".weight_shape"),
                          torch.tensor(list(shape), dtype=torch.int32))
                    state["n_quant"] += 1
                elif name.endswith(".weight_scale_inv"):
                    # consumed by its paired FP8 weight (or by mtp/visual
                    # pairs that we copy below)
                    continue
                elif name.endswith(".weight") and scale_name in header:
                    # an FP8 tensor whose pair we did NOT quantize (mtp.*,
                    # visual.*): dequant to BF16 so the checkpoint is uniform.
                    si = torch.frombuffer(
                        mm[data_off + header[scale_name]["data_offsets"][0]:
                        data_off + header[scale_name]["data_offsets"][1]],
                        dtype=torch.bfloat16).reshape(header[scale_name]["shape"])
                    w.add(name, fp8_dequant(t, si).to(torch.bfloat16).contiguous())
                else:
                    w.add(name, t.contiguous())
            mm.close()
        w.flush()
        state["done_src"].append(sh)
        state["out_counter"] = w.idx
        with open(state_path, "w") as f:
            json.dump(state, f)
        el = time.time() - t0
        pct = 100.0 * len(state["done_src"]) / len(shards)
        rate = el / len(state["done_src"])
        print(f"[{pct:5.1f}%] {sh}  ({el/60:.1f}m elapsed, "
              f"~{rate * len(shards) / 60:.0f}m projected total)", flush=True)

    with open(index_path, "w") as f:
        index = {"metadata": {"total_size": _total_size(dst)}, "weight_map": {}}
        for fn in sorted(os.listdir(dst)):
            if not (fn.startswith("model-") and fn.endswith(".safetensors")):
                continue
            _, h = parse_header(os.path.join(dst, fn))
            for k in h:
                if k != "__metadata__":
                    index["weight_map"][k] = fn
        json.dump(index, f)
    os.remove(state_path)
    print(f"DONE: {state['n_quant']} linear weights quantized, "
          f"{(time.time()-t0)/60:.1f} min")


def _total_size(dst):
    total = 0
    for fn in os.listdir(dst):
        if not (fn.startswith("model-") and fn.endswith(".safetensors")):
            continue
        data_off, h = parse_header(os.path.join(dst, fn))
        total += max(m["data_offsets"][1] for m in h.values()
                     if m.get("data_offsets")) - data_off
    return total


if __name__ == "__main__":
    sys.exit(main())
