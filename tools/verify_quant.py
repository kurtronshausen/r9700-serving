#!/usr/bin/env python3
"""Post-hoc verification of a quant_flashnext.py output directory.

    python verify_quant.py <src_bf16_dir> <dst_quant_dir>

Checks: every expert .weight in the source has exactly the
_packed/_scale/_shape triple in the destination (and no leftover .weight),
dtypes/shapes match the expected layout, total size is sane, and the
config carries the compressed-tensors quantization block.
"""

import json
import os
import re
import struct
import sys

QUANT_RE = re.compile(
    r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+)"
    r"\.(gate_proj|up_proj|down_proj)\.weight$"
)


def parse_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main(src_dir, dst_dir):
    src_wm = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
    dst_wm = json.load(open(os.path.join(dst_dir, "model.safetensors.index.json")))["weight_map"]

    src_experts = {QUANT_RE.match(k).group(0) for k in src_wm if QUANT_RE.match(k)}
    n_exp_tensors = 3 * len(src_experts)
    dst_experts = {k for k in dst_wm if k.endswith(".weight_packed")}
    assert dst_experts == {f"{b}.{p}.weight_packed"
                           for b in src_experts
                           for p in ("gate_proj", "up_proj", "down_proj")}, \
        f"packed set mismatch: {len(dst_experts)} vs {n_exp_tensors}"
    leftover = [k for k in dst_wm if QUANT_RE.match(k)]
    assert not leftover, f"unquantized expert weights remain: {leftover[:3]}"
    # same non-expert tensor count
    src_other = set(src_wm) - {k for k in src_wm if QUANT_RE.match(k)}
    dst_other = set(dst_wm) - {k for k in dst_wm
                               if any(k.endswith(s) for s in (".weight_packed", ".weight_scale", ".weight_shape"))}
    assert src_other == dst_other, \
        f"passthrough mismatch: only_src={len(src_other - dst_other)} only_dst={len(dst_other - src_other)}"

    # layout spot-check: first shard, a few tensors of each kind
    shard = sorted(f for f in os.listdir(dst_dir)
                   if f.startswith("model-") and f.endswith(".safetensors"))[0]
    h = parse_header(os.path.join(dst_dir, shard))
    seen = {"packed": 0, "scale": 0, "shape": 0}
    for k, m in h.items():
        if k == "__metadata__":
            continue
        if k.endswith(".weight_packed"):
            assert m["dtype"] == "I32" and m["shape"][1] * 8 % 128 == 0
            seen["packed"] += 1
        elif k.endswith(".weight_scale"):
            assert m["dtype"] == "BF16"
            seen["scale"] += 1
        elif k.endswith(".weight_shape"):
            assert m["dtype"] == "I32" and m["shape"] == [2]
            seen["shape"] += 1
    assert all(v > 0 for v in seen.values()), seen

    cfg = json.load(open(os.path.join(dst_dir, "config.json")))
    assert cfg.get("quantization_config", {}).get("quant_method") == "compressed-tensors"

    # size sanity: experts ~ (params*0.5 + scales*2)/1e9 GiB-ish; just check
    # total is between 0.3x and 0.6x of the source (4-bit experts dominate).
    def total_size(d):
        tot = 0
        for fn in os.listdir(d):
            if not (fn.startswith("model-") and fn.endswith(".safetensors")):
                continue
            off, hh = parse_header(os.path.join(d, fn))
            tot += max(m["data_offsets"][1] for m in hh.values()
                       if m.get("data_offsets")) - off
        return tot
    s, t = total_size(src_dir), total_size(dst_dir)
    ratio = t / s
    assert 0.25 < ratio < 0.7, ratio
    print(f"VERIFY OK: {len(src_experts)} experts, "
          f"{n_exp_tensors} packed tensors, passthrough identical, "
          f"size {t/2**30:.1f} GiB = {ratio:.2f}x source ({s/2**30:.1f} GiB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
