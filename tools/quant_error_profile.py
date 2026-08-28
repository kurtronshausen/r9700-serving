#!/usr/bin/env python3
"""Post-quant error profile: our RTN W4A16 output vs the BF16 source.

Samples experts across layers and reports the relative quantization error
(||w - dequant(w_q)|| / ||w||) per projection, since the aixiaoma reference
turns out to carry a different expert arrangement (not comparable per index).

    python quant_error_profile.py <src_bf16_dir> <our_quant_dir> [n_layers n_experts]
"""
import json
import math
import random
import struct
import sys

import numpy as np

SRC = sys.argv[1]
OURS = sys.argv[2]
N_LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
N_EXPERTS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
random.seed(0)


def load(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        pos = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            nb = 4 if v["dtype"] in ("I32", "I64") else 2
            sz = int(math.prod(v["shape"])) * nb
            if k == key:
                f.seek(pos)
                b = f.read(sz)
                dt = np.int32 if v["dtype"] == "I32" else np.int64 if v["dtype"] == "I64" else np.uint16
                u = np.frombuffer(b, dt)
                if v["dtype"] == "BF16":
                    u = (u.astype(np.uint32) << 16).view(np.float32)
                return u.reshape(v["shape"])
            pos += sz
    raise KeyError(key)


def index_of(d):
    return json.load(open(d + "/model.safetensors.index.json"))["weight_map"]


src_wm = index_of(SRC)
our_wm = index_of(OURS)

# which main layers have MoE experts
moelayers = sorted(
    {int(k.split(".layers.")[1].split(".")[0])
     for k in src_wm
     if k.startswith("model.language_model.layers.")
     and k.endswith(".mlp.experts.gate_up_proj")}
)
print(f"MoE layers in source: {len(moelayers)} (mtp excluded from sampling)")
layers = random.sample(moelayers, min(N_LAYERS, len(moelayers)))

rel = {"gate_proj": [], "up_proj": [], "down_proj": []}
for l in layers:
    pre = f"model.language_model.layers.{l}.mlp.experts"
    fused = load(SRC + "/" + src_wm[f"{pre}.gate_up_proj"], f"{pre}.gate_up_proj")
    down = load(SRC + "/" + src_wm[f"{pre}.down_proj"], f"{pre}.down_proj")
    E = fused.shape[0]
    for e in random.sample(range(E), min(N_EXPERTS, E)):
        for proj, w in (("gate_proj", fused[e, : fused.shape[1] // 2]),
                        ("up_proj", fused[e, fused.shape[1] // 2:]),
                        ("down_proj", down[e])):
            base = f"{pre}.{e}.{proj}"
            wp = load(OURS + "/" + our_wm[base + ".weight_packed"], base + ".weight_packed")
            s = load(OURS + "/" + our_wm[base + ".weight_scale"], base + ".weight_scale")
            N, K8 = wp.shape
            q = np.empty((N, K8 * 8), np.int64)
            for j in range(8):
                q[:, j::8] = (wp.astype(np.int64) >> (4 * j)) & 15
            dq = (q - 7) * np.repeat(s, 128, axis=1)
            err = float(np.linalg.norm(w - dq) / np.linalg.norm(w))
            rel[proj].append(err)

print(f"samples: {len(layers)} layers x {N_EXPERTS} experts x 3 projections")
for proj, v in rel.items():
    v = np.array(v)
    print(f"{proj:12s} rel-rms  mean={v.mean():.4f} median={np.median(v):.4f} "
          f"p99={np.percentile(v, 99):.4f} max={v.max():.4f}  n={len(v)}")
agg = np.concatenate(list(rel.values()))
print(f"{'ALL':12s} rel-rms  mean={agg.mean():.4f}  (typical 4-bit RTN: 0.05-0.15)")
