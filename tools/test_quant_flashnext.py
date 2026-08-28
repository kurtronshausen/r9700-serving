#!/usr/bin/env python3
"""Self-check for quant_flashnext.py (run inside the flashnext image).

Builds a synthetic BF16 mini-checkpoint with the real key layout, runs the
quantizer on it, then verifies:
  1. layout: packed I32 [N, K/8], scale BF16 [N, K/128], shape I32 [2],
     passthrough tensors byte-identical
  2. round-trip: unpack(packed) with scale matches the source within
     0.5*group_scale (plus bf16 epsilon)
  3. packing equivalence with vLLM's own pack_quantized_values_into_int32
     (same values, same bit order)
"""

import json
import os
import shutil
import struct
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_flashnext as qf  # noqa: E402

torch.manual_seed(0)

L = 2  # layers
E = 4  # experts
H, IM = 640, 256  # both divisible by 128


def mk_tensors():
    """Real source layout: per-layer 3D expert tensors (main + MTP)."""
    t = {}
    for l in range(L):
        pre = f"model.language_model.layers.{l}.mlp.experts"
        # gate/up stored fused: [E, 2*IM, H], gate in the first half (the
        # split vLLM's RoutedExperts.load_weights does: chunk(2, dim=1))
        t[f"{pre}.gate_up_proj"] = torch.randn(E, 2 * IM, H, dtype=torch.bfloat16) * 0.02
        t[f"{pre}.down_proj"] = torch.randn(E, H, IM, dtype=torch.bfloat16) * 0.02
        t[f"model.language_model.layers.{l}.mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(H, H, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight"] = torch.randn(H, H, dtype=torch.bfloat16)
        t[f"model.language_model.layers.{l}.ple.ngram_embedding.weight"] = torch.randn(32, H, dtype=torch.bfloat16)
    mpre = "mtp.layers.0.mlp.experts"
    t[f"{mpre}.gate_up_proj"] = torch.randn(E, 2 * IM, H, dtype=torch.bfloat16) * 0.02
    t[f"{mpre}.down_proj"] = torch.randn(E, H, IM, dtype=torch.bfloat16) * 0.02
    t["model.language_model.embed_tokens.weight"] = torch.randn(512, H, dtype=torch.bfloat16)
    t["model.language_model.lm_head.weight"] = torch.randn(512, H, dtype=torch.bfloat16)
    t["visual.proj.weight"] = torch.randn(H, H, dtype=torch.bfloat16)
    return t


def expert_source(tensors, name):
    """Original per-expert [N, K] weight behind an output key."""
    parts = name.split(".")
    l = int(parts[parts.index("layers") + 1])
    e = int(parts[parts.index("experts") + 1])
    pre = (f"model.language_model.layers.{l}.mlp.experts"
           if parts[0] == "model" else "mtp.layers.0.mlp.experts")
    fused = tensors[f"{pre}.gate_up_proj"]
    down = tensors[f"{pre}.down_proj"]
    if name.endswith("gate_proj"):
        return fused[e, :IM]
    if name.endswith("up_proj"):
        return fused[e, IM:]
    return down[e]


def main():
    work = tempfile.mkdtemp(prefix="qfn-test-")
    try:
        src, dst = os.path.join(work, "src"), os.path.join(work, "dst")
        os.makedirs(src)
        tensors = mk_tensors()
        # two source shards, split tensors arbitrarily
        keys = sorted(tensors)
        for i, chunk in enumerate((keys[: len(keys) // 2], keys[len(keys) // 2:])):
            save_file({k: tensors[k] for k in chunk}, f"{src}/model-{i+1:05d}.safetensors")
        wm = {k: (f"model-{1 if k < keys[len(keys)//2] else 2:05d}.safetensors") for k in keys}
        with open(f"{src}/model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {}, "weight_map": wm}, f)
        with open(f"{src}/config.json", "w") as f:
            json.dump({"model_type": "qwen4_exp", "x": 1}, f)
        for fn in ("tokenizer_config.json", "generation_config.json"):
            with open(f"{src}/{fn}", "w") as f:
                f.write("{}")

        # invoke quantizer
        sys.argv = ["quant_flashnext.py", src, dst, "--shard-gb", "100"]
        qf.main()

        # ---- 1. layout ----
        with open(f"{dst}/model.safetensors.index.json") as f:
            out_wm = json.load(f)["weight_map"]

        def read_tensor(name):
            fn = out_wm[name]
            with open(f"{dst}/{fn}", "rb") as g:
                (n,) = struct.unpack("<Q", g.read(8))
                hh = json.loads(g.read(n))
                m = hh[name]
                g.seek(8 + n + m["data_offsets"][0])
                b = g.read(m["data_offsets"][1] - m["data_offsets"][0])
            dt = {"I32": torch.int32, "BF16": torch.bfloat16}[m["dtype"]]
            return torch.frombuffer(b, dtype=dt).reshape(m["shape"]), m["dtype"]

        base0 = "model.language_model.layers.0.mlp.experts.0"
        for proj, shape in (("gate_proj", (IM, H)), ("up_proj", (IM, H)),
                            ("down_proj", (H, IM))):
            n, k = shape
            p, dt = read_tensor(f"{base0}.{proj}.weight_packed")
            s, sdt = read_tensor(f"{base0}.{proj}.weight_scale")
            sh, sdt2 = read_tensor(f"{base0}.{proj}.weight_shape")
            assert dt == "I32" and list(p.shape) == [n, k // 8], (proj, p.shape, n, k)
            assert sdt == "BF16" and list(s.shape) == [n, k // 128], (proj, s.shape, n, k)
            assert sh.tolist() == [n, k]
        assert f"{base0}.gate_proj.weight" not in out_wm
        # MTP experts quantized too
        assert "mtp.layers.0.mlp.experts.0.gate_proj.weight_packed" in out_wm
        assert "mtp.layers.0.mlp.experts.3.down_proj.weight_scale" in out_wm
        # gate/up actually split (not swapped / not whole-fused)
        pg, _ = read_tensor(f"{base0}.gate_proj.weight_packed")
        pu, _ = read_tensor(f"{base0}.up_proj.weight_packed")
        assert not torch.equal(pg, pu)
        # passthrough byte-identical (incl. the 3D expert tensors are GONE)
        assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in out_wm
        for k in ("model.language_model.layers.0.self_attn.q_proj.weight",
                  "model.language_model.embed_tokens.weight",
                  "model.language_model.layers.0.mlp.gate.weight",
                  "model.language_model.layers.0.ple.ngram_embedding.weight",
                  "visual.proj.weight"):
            t, dt = read_tensor(k)
            assert dt == "BF16" and torch.equal(t, tensors[k]), k
        with open(f"{dst}/config.json") as f:
            cfg = json.load(f)
        assert cfg["quantization_config"]["quant_method"] == "compressed-tensors"
        assert cfg["x"] == 1  # original config preserved
        assert os.path.exists(f"{dst}/tokenizer_config.json")
        print("1. layout: OK")

        # ---- 2. round-trip (main + MTP layers) ----
        max_err_ratio = 0.0
        pres = ([f"model.language_model.layers.{l}.mlp.experts" for l in range(L)]
                + ["mtp.layers.0.mlp.experts"])
        for pre in pres:
            for e in range(E):
                for proj, shape in (("gate_proj", (IM, H)), ("up_proj", (IM, H)),
                                    ("down_proj", (H, IM))):
                    n, k = shape
                    name = f"{pre}.{e}.{proj}"
                    p, _ = read_tensor(f"{name}.weight_packed")
                    s, _ = read_tensor(f"{name}.weight_scale")
                    w = expert_source(tensors, name)
                    # uint32 view: int32 arithmetic shift breaks when bit 31 set
                    pu = p.to(torch.int64) & 0xFFFFFFFF  # unsigned view
                    q = torch.zeros(n, k, dtype=torch.int32)
                    for i in range(8):
                        q[:, i::8] = (pu >> (4 * i)) & 0xF
                    deq = ((q - 7).to(torch.float32) * s.to(torch.float32)
                           .repeat_interleave(128, dim=1))
                    err = (deq - w.to(torch.float32)).abs().max().item()
                    # bound: 0.5*group_scale (rounding) + bf16 scale-quant error
                    g = w.to(torch.float32).reshape(n, k // 128, 128)
                    gscale = g.abs().amax(dim=2).clamp_min(1e-8) / 7
                    max_err_ratio = max(max_err_ratio, err / gscale.max().item())
        assert max_err_ratio < 0.55, max_err_ratio
        print(f"2. round-trip: OK (max_err/max_scale = {max_err_ratio:.3f} < 0.55)")

        # ---- 3. packing equivalence with vLLM ----
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            pack_quantized_values_into_int32,
            unpack_quantized_values_into_int32,
        )
        from vllm.scalar_type import scalar_types

        w = expert_source(tensors, f"{base0}.gate_proj")
        packed, scale = qf.rtng128(w)
        # unpack mine with vLLM's unpacker, re-pack with vLLM's packer
        vals = unpack_quantized_values_into_int32(packed, scalar_types.uint4b8, packed_dim=1)
        repacked = pack_quantized_values_into_int32(vals, scalar_types.uint4b8, packed_dim=1)
        assert torch.equal(repacked, packed), "vLLM pack/unpack round trip mismatch"
        # and direct: quantize the same values with vLLM packer -> identical words
        wf = w.to(torch.float32)
        g = wf.reshape(w.shape[0], -1, 128)
        s = g.abs().amax(dim=2).clamp_min(1e-8) / 7
        qv = (torch.round(g / s.unsqueeze(2)) + 7).clamp(0, 14).reshape(w.shape).to(torch.int32)
        vllm_packed = pack_quantized_values_into_int32(qv, scalar_types.uint4b8, packed_dim=1)
        assert torch.equal(vllm_packed, packed), "word-level mismatch vs vLLM packer"
        print("3. vLLM packing equivalence: OK")

        # dequant sanity: vLLM-unpacked values dequantize back to the weights
        # (bound: 0.5 group-scale rounding + bf16 rounding of the scale itself)
        deq = ((vals.float() - 7) * scale.float().repeat_interleave(128, dim=1))
        assert (deq - w.float()).abs().max().item() < 0.56 * s.max().item()
        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
