#!/bin/bash
# Run depth sweep for one model
PROFILE=$1
just down >/dev/null 2>&1 || true
MODEL_PROFILE=$PROFILE just up >/dev/null 2>&1
sleep 3

for D in 4096 8192 16384 32768 65536 128000; do
  echo "=== depth=$D ==="
  just exec uvx llama-benchy \
    --base-url http://localhost:8180/v1 \
    --served-model-name "$PROFILE" \
    --tg 32 \
    --depth "$D" \
    --runs 2 \
    --save-result /tmp/bench.json \
    --format json 2>/dev/null || true
  just exec cat /tmp/bench.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for b in data.get('benchmarks', []):
    tg = b.get('tg_throughput', {}).get('mean', 0)
    pp = b.get('pp_throughput', {}).get('mean', 0)
    ttft = b.get('ttfr', {}).get('mean', 0) / 1000 if 'ttfr' in b else 0
    print(f'$D: tg={tg:.1f} t/s pp={pp:.0f} t/s ttft={ttft:.3f}s')
"
done
