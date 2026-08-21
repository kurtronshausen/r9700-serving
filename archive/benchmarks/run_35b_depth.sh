#!/bin/bash
set -e
PROFILE="qwen3.6-35b-a3b"

just down >/dev/null 2>&1
MODEL_PROFILE=$PROFILE just up >/dev/null 2>&1
sleep 3

RESULTS=""
for DEPTH in 4096 8192 16384 32768 65536 128000; do
  echo "=== depth=$DEPTH ==="
  just exec uvx llama-benchy --base-url http://localhost:8180/v1 \
    --served-model-name $PROFILE \
    --tg 32 \
    --depth $DEPTH \
    --runs 2 \
    --save-result /tmp/bench.json \
    --format json 2>/dev/null || true
  
  just exec bash -c "python3 -c \"
import json
with open('/tmp/bench.json') as f:
    data = json.load(f)
for b in data.get('benchmarks', []):
    tg = b.get('tg_throughput', {}).get('mean', 0)
    pp = b.get('pp_throughput', {}).get('mean', 0)
    ttft = b.get('ttfr', {}).get('mean', 0) / 1000 if 'ttfr' in b else 0
    print(f'depth=\$1 tg32={tg:.1f} t/s | pp={pp:.0f} t/s | ttft={ttft:.3f}s' % {'\$1': \$2})
\" _ \$DEPTH" 
done
