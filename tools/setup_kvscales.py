#!/usr/bin/env python3
"""Create a local, calibratable copy of an HF model directory.

The stock Qwen3.5/3.6/3.8 FP8 checkpoints ship without fp8 KV scales, so fp8
KV serves at scale 1.0. Calibration writes a `model-kvscales.safetensors`
sidecar plus index entries into a *local* copy (the HF snapshot blobs are
content-addressed and should not be modified in place).

This creates that copy: every file in the resolved HF snapshot becomes a
symlink into the hub (cheap, no extra disk), except `model.safetensors.index.json`
which is dereferenced to a real file so the calibrator can edit it. Files that
the calibrator writes (`model-kvscales.safetensors`) are new, so the hub is
never touched.

Usage:
    python tools/setup_kvscales.py Qwen/Qwen3.8-27B-FP8 ~/models-local/Qwen3.8-27B-FP8-kvscales
"""
import argparse
import os
import sys


def resolve_snapshot(model_id: str, cache_root: str | None = None) -> str:
    root = cache_root or os.path.expanduser("~/.cache/huggingface")
    hub_dir = os.path.join(root, "hub", "models--" + model_id.replace("/", "--"))
    snapshots = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots):
        raise SystemExit(f"HF hub dir not found: {snapshots}")
    # Prefer the branch/commit the refs file points at, else the latest snapshot.
    refs = os.path.join(hub_dir, "refs", "main")
    if os.path.isfile(refs):
        sha = open(refs).read().strip()
        cand = os.path.join(snapshots, sha)
        if os.path.isdir(cand):
            return cand
    subs = [os.path.join(snapshots, d) for d in os.listdir(snapshots)]
    subs = [d for d in subs if os.path.isdir(d)]
    if not subs:
        raise SystemExit(f"no snapshots under {snapshots}")
    return max(subs, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="HF repo id, e.g. Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("out_dir", help="local copy destination, e.g. ~/models-local/...")
    ap.add_argument("--force", action="store_true", help="recreate even if exists")
    args = ap.parse_args()

    src = resolve_snapshot(args.model_id)
    dst = os.path.expanduser(args.out_dir)
    index = os.path.join(dst, "model.safetensors.index.json")
    if os.path.isdir(dst) and not args.force:
        if os.path.isfile(index):
            print(f"copy already exists: {dst}")
            return
        print(f"WARN: {dst} exists but has no index; will populate missing files")

    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.islink(s) or os.path.isfile(s):
            if os.path.exists(d) or os.path.islink(d):
                continue
            # The index is dereferenced so the calibrator can edit it in place;
            # everything else is a symlink into the hub (no disk cost).
            if name == "model.safetensors.index.json":
                import shutil
                shutil.copyfile(s, d)
            else:
                os.symlink(s, d)
        else:
            import shutil
            if not os.path.exists(d):
                shutil.copytree(s, d)
    if not os.path.isfile(index):
        raise SystemExit(f"model.safetensors.index.json missing after copy: {dst}")
    print(f"local copy ready: {dst}")
    print(f"  source snapshot: {src}")


if __name__ == "__main__":
    main()
