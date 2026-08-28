#!/usr/bin/env python3
"""RTN W4A16 quantizer for Qwen3.8-Flash-Next (compressed-tensors format).

Quantizes the model's routed MoE experts to symmetric int4 / group-128 in the
exact layout vLLM's compressed-tensors WNA16 path (and the aixiaoma W4A16
reference checkpoint) expect, and copies every other tensor unchanged:

    experts.E.{gate,up,down}_proj.weight      BF16 [N, K]
        -> experts.E.{gate,up,down}_proj.weight_packed  I32  [N, K/8]
        -> experts.E.{gate,up,down}_proj.weight_scale   BF16 [N, K/128]
        -> experts.E.{gate,up,down}_proj.weight_shape   I32  [2]  (= [N, K])

Packing convention (vllm .../quant_utils.py::pack_quantized_values_into_int32,
packed_dim = K): element k of row n lives at bits (k%8)*4 of
packed[n, k//8]. uint4b8 bias = 7: q = clamp(round(w/scale) + 7, 0, 15).

CPU-only, streams the source shards via mmap (peak RAM a few GiB), resumable.
Run inside the flashnext image (needs torch + safetensors):

    python /workspace/tools/quant_flashnext.py \
        /srv/llm/Qwen/Qwen3.8-Flash-Next \
        /srv/llm/kurt/flashnext-w4a16-rtng128
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
# Exactly the tensors the reference (aixiaoma) W4A16 checkpoint quantizes.
QUANT_RE = re.compile(
    r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+)"
    r"\.(gate_proj|up_proj|down_proj)\.weight$"
)
# config.json "ignore" list copied verbatim from the reference checkpoint so
# vLLM's compressed-tensors scheme selection behaves identically.
IGNORE_LIST = [
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*mtp\\..*",
    "re:.*\\.ple\\..*",
    "re:.*visual\\..*",
    "re:.*\\.gate$",
    "re:.*hyper_connection.*",
    "re:.*indexer.*",
    "re:.*\\.linear_attn\\..*",
    "re:.*shared_expert.*",
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
# Small metadata / config files to carry over (everything else, e.g. the
# source's own index and .gitattributes, is regenerated or dropped).
COPY_FILES = [
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "README.md",
]


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
    q = torch.round(g / scale.unsqueeze(2)).clamp(0, 2 * BIAS)
    q8 = q.reshape(N, -1).to(torch.int32)  # [N, K]
    words = torch.zeros(N, K // PACK, dtype=torch.int32, device=w.device)
    for i in range(PACK):
        words |= (q8[:, i::PACK] & 0xF) << (4 * i)
    return words, scale.to(torch.bfloat16)


class ShardWriter:
    def __init__(self, dst_dir, max_bytes):
        self.dst_dir = dst_dir
        self.max_bytes = max_bytes
        self.idx = 0
        self.cur = {}

    def _path(self, i):
        return os.path.join(self.dst_dir, f"model-{i:05d}.safetensors")

    def _cur_size(self):
        return sum(t.numel() * t.element_size() for t in self.cur.values())

    def add(self, name, tensor):
        self.cur[name] = tensor
        if self._cur_size() > self.max_bytes:
            self.flush()

    def flush(self):
        if not self.cur:
            return
        tensors = {k: v for k, v in self.cur.items()}
        save_file(tensors, self._path(self.idx), metadata={"format": "pt"})
        print(f"  wrote {os.path.basename(self._path(self.idx))}", flush=True)
        self.idx += 1
        self.cur = {}


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

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    src_shards = sorted(set(weight_map.values()))
    print(f"source: {len(src_shards)} shards, {len(weight_map)} tensors")

    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"resuming: {len(state['done_src'])}/{len(src_shards)} shards done")
    else:
        state = {"done_src": [], "out_counter": 0}

    # Carry over the static files once.
    if not os.path.exists(os.path.join(dst, "tokenizer_config.json")):
        for fn in COPY_FILES:
            s = os.path.join(src, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))
        cfg_path = os.path.join(src, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = QUANT_CONFIG
        with open(os.path.join(dst, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("copied static files + wrote quantization_config")

    max_bytes = int(args.shard_gb * 1e9)
    w = ShardWriter(dst, max_bytes)
    w.idx = state["out_counter"]

    quant_tensors = sum(1 for k in weight_map if QUANT_RE.match(k))
    total_experts = len({QUANT_RE.match(k).group(1) for k in weight_map if QUANT_RE.match(k)})
    done_experts = 0
    t0 = time.time()

    for shard in src_shards:
        if shard in state["done_src"]:
            continue
        path = os.path.join(src, shard)
        data_off, header = parse_header(path)
        fsize = os.path.getsize(path)
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                dtype, shape, off = meta["dtype"], meta["shape"], meta["data_offsets"]
                raw = mm[data_off + off[0]: data_off + off[1]]
                m = QUANT_RE.match(name)
                if m:
                    t = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape)
                    packed, scale = rtng128(t)
                    base = name[: -len(".weight")]
                    w.add(f"{base}.weight_packed", packed.contiguous())
                    w.add(f"{base}.weight_scale", scale.contiguous())
                    w.add(f"{base}.weight_shape",
                          torch.tensor(shape, dtype=torch.int32))
                    done_experts += 3
                else:
                    t = torch.frombuffer(raw, dtype=_DTYPE_MAP[dtype]).reshape(shape)
                    w.add(name, t.contiguous())
            mm.close()
        w.flush()
        state["done_src"].append(shard)
        state["out_counter"] = w.idx
        with open(state_path, "w") as f:
            json.dump(state, f)
        el = time.time() - t0
        pct = 100.0 * len(state["done_src"]) / len(src_shards)
        rate = el / len(state["done_src"])
        print(f"[{pct:5.1f}%] {shard}  ({el/60:.1f}m elapsed, "
              f"~{rate*len(state['done_src']):.0f}s total projected)", flush=True)

    # Final index + config sanity.
    with open(index_path, "w") as f:
        index = {
            "metadata": {"total_size": _total_size(dst)},
            "weight_map": {},
        }
        for fn in sorted(os.listdir(dst)):
            if not (fn.startswith("model-") and fn.endswith(".safetensors")):
                continue
            _, h = parse_header(os.path.join(dst, fn))
            for k in h:
                if k != "__metadata__":
                    index["weight_map"][k] = fn
        json.dump(index, f)
    os.remove(state_path)
    print(f"DONE: {quant_tensors} expert tensors, {total_experts} experts, "
          f"{(time.time()-t0)/60:.1f} min")


_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "I8": torch.int8,
}


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
