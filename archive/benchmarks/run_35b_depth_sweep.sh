#!/bin/bash
# Run llama-benchy depth sweep for 35B-A3B
# Outputs results to benchmarks/ directory

set -e

PROFILE="qwen3.6-35b-a3b"
OUTDIR="benchmarks"
MODE="MD"  # JSON or MD

MODE_NAME="BF16+MoeTuned+MtPOff"

just down >/dev/null 2>&1
MODEL_PROFILE=$PROFILE just up >/dev/null 2>&1
sleep 3

cat >> "$OUTDIR/08_11_${PROFILE}_${MODE_NAME}_128k_depth.md" <<EOF
# Depth Sweep: $PROFILE
# Date: $(date '+%Y-%m-%d %H:%M')
# Config: BF16 KV + tuned MOE + MTP disabled
# Model: Qwen/Qwen3.6-35B-A3B-FP8
# Stack: vLLM 0.27.0, ROCm 7.14.0, Torch 2.13, Triton 3.8.0
# Hardware: 2x R9700 (gfx1201), tp=2
EOF

for DEPTH in 4096 8192 16384 32768 65536 128000; do
  echo ""
  echo "=== Running depth=$DEPTH ==="
  
  just exec uvx llama-benchy --base-url http://localhost:8180/v1 \
    --served-model-name $PROFILE \
    --tg 32 \
    --depth $DEPTH \
    --runs 2 \
    --save-result /tmp/bench_${PROFILE}_pp2048_tg32_d${DEPTH}.json \
    --format json \
    -q 2>&1 | grep -E "tg_throughput|tg_req_throughput" | head -2
  
  just exec python3 - << 'PYEOF'
import json, re
with open('/tmp/bench_${PROFILE}_pp2048_tg32_d${DEPTH}.json') as f:
    data = json.load(f)
for b in data.get('benchmarks', []):
    tg_mean = b.get('tg_throughput', {}).get('mean', 0)
    pp_mean = b.get('pp_throughput', {}).get('mean', 0)
    ttft = b.get('ttfr', {}).get('mean', 0) / 1000 if 'ttfr' in b else 0
    print(f"depth=${DEPTH}: tg={tg_mean:.1f} t/s, pp={pp_mean:.0f} t/s, ttft={ttft:.3f}s")
PYEOF
done

echo ""
echo "=== All results written to benchmarks/ ==="
