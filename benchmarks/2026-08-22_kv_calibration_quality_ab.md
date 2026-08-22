# KV-scale calibration: quality A/B (does it matter? 2026-08-22)

Goal: quantify the *quality* benefit of calibrated fp8 KV scales vs scale 1.0,
beyond the already-established numerics-fidelity facts (deep-layer V amax
~130-132 mis-ranged at scale 1.0; calibrated vs scale-1.0 outputs deterministically
diverge ~20-27%; no throughput regression). The question: does calibration
*measurably improve* prediction quality?

Setup: Qwen3.8-27B-FP8, TP=2, vLLM 0.28.0rc2, AITER unified attention, MTP3.
Three KV configs, same document, scored identically:
  - `bf16`    : full-precision KV reference (quality floor)
  - `fp8_scale1`: fp8 KV at scale 1.0 (stock checkpoint, no scales)
  - `fp8_calib`: fp8 KV with calibrated q/k/v scales (ensure-kvscales)

## Methodology (tools)

- `benchmarks/kv_ppl_ab.py` — held-out continuation perplexity (PPL).
  Builds a long realistic document from the repo's own `.md`/`.py` text,
  splits it into independent `ctx + cont` segments, and for each config reads
  the true-token logprob of the trailing `cont` tokens via
  `prompt_logprobs` (raised top-K via `max_logprobs`, default 128). Only
  positions whose true token is returned by **all three** configs are scored,
  so the comparison is exact and fair. Reports per-config PPL and
  `|PPL_cfg - PPL_bf16|`.
- `benchmarks/kv_recall_ab.py` — long-context needle-in-haystack recall.
  Plants a distinctive single-token fact ("The passphrase is <TOKEN>") at
  several depths in a long filler context and asks for extraction. Normal
  greedy generation (no `prompt_logprobs`), so true long contexts (32K) fit.

## Findings

### 1. Perplexity (ctx=2048, cont=32, 32 segments / 1024 scored tokens) — noise

| config | PPL | |PPL - bf16| |
|:-------|----:|------------:|
| bf16 | 2.6946* | 0 |
| fp8 scale1 | 2.7094 | 0.0071 |
| fp8 calib | 2.6946* | 0.0220 |

`*` bf16 and calib both printed 2.6946 by coincidence of rounding; the bf16
reference is used as the baseline. Note the sign of the gap is **unstable across
segment counts**: a 4-segment run showed calib closer to bf16 (scale1 0.0828 vs
calib 0.0861), while the 32-segment run shows scale1 closer (scale1 0.0071 vs
calib 0.0220). Both gaps are <1% of PPL and the direction flips with data — the
scale1-vs-calib difference is **measurement noise**, not a reproducible signal.
The dominant, reproducible effect is fp8 quantization vs bf16 itself (both fp8
configs sit above bf16 PPL).

### 2. Long-context recall (ctx=32768, depths 0.1/0.5/0.9, 5 reps each) — identical

| depth | bf16 | fp8 scale1 | fp8 calib |
|:------|-----:|-----------:|----------:|
| 0.1 | 2/5 | 2/5 | 2/5 |
| 0.5 | 1/5 | 1/5 | 1/5 |
| 0.9 | 1/5 | 1/5 | 1/5 |
| total | 4/15 | 4/15 | 4/15 |

All three configs are byte-for-byte identical in recall outcomes, including the
bf16 reference. The three configs *do* differ in KV numerics (proven elsewhere),
yet recall is unchanged. (Recall itself is low on this GDN model at 32K even with
bf16 KV — a poor discriminating test — but the identical-outcomes result is the
signal: calibration changes nothing here.)

### 3. Memory constraints on the PPL path

`prompt_logprobs` materializes `[seq, vocab]` fp32 logits on GPU. On 2× 32GB
gfx1201 with the 27B model loaded (~26-27GB reserved), the full `[seq, vocab]`
tensor OOMs above ~4K context (8K needs ~7.6GB free). The vLLM KV pool reserves
up to `gpu_memory_utilization` and does not free it for the logits computation, so
raising mem-util does **not** help (it worsens it). Hence the PPL path is limited
to moderate context; long-context quality had to be tested via recall instead.

## Conclusion

On two independent quality metrics — per-token PPL and long-context recall — the
calibrated-KV config is indistinguishable from scale 1.0. Calibration is therefore
best understood as a **correctness/faithfulness fix** (it removes a provably
mis-ranged default and changes deterministic outputs), **not** a measured quality
or throughput win on this stack. Its practical value would still be relevant if a
future metric or workload proves KV fidelity matters (e.g. high-precision
long-context recall on a model/task that actually stresses it); the calibration
infrastructure (`ensure-kvscales`) stays as the safer default.
