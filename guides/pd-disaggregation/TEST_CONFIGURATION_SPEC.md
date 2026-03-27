# TEST CONFIGURATION

## 1. Scope

This document is the single source of truth for PD break-even benchmarking.

- Compare deploy modes:
  - baseline: colocated vLLM serving (no PD) [BASELINE_NON_PD_DEPLOYMENT.md](BASELINE_NON_PD_DEPLOYMENT.md)
  - baseline-pd: separated prefill/decode without llm-d [BASELINE_PD_DEPLOYMENT.md](BASELINE_PD_DEPLOYMENT.md)
  - llm-d-pd: separated prefill/decode with llm-d [README.xpu.md](README.xpu.md)

- Goal: identify crossover region where PD improves TTFT and throughput

## 2. Global Fixed Parameters

Use the same fixed settings for all test cases.

```yaml
sampling:
  temperature: 0.0
  top_p: 1.0
  stream: false

execution:
  warmup_requests: 30
  measured_requests: 300
  max_retries: 0
  random_seed: 20260325

metrics_required:
  - ttft_ms_avg
  - ttft_ms_p50
  - ttft_ms_p99
  - tpot_ms_avg
  - tpot_ms_p50
  - tpot_ms_p99
  - throughput_req_per_s
  - throughput_token_per_s
  - success_rate
```

## 3. xPyD Design for Multi-Model on PVC and BMG

This section defines a practical xPyD (XPU Prefill-Decode) starting design for the following models:

- GLM-4-9B
- Qwen3-30B-A3B-Instruct-2507
- DeepSeek-R1-Distill-Llama-70B
- Qwen3-235B-A22B-Instruct-2507

### 3.1 Hardware Envelope

- PVC: Intel Data Center GPU Max class, 64GB HBM per card
- BMG: Intel Battlemage class, 24GB VRAM per card

### 3.2 Recommended xPyD Topology Matrix

Design principle for this matrix:
- Prefill (P): lower tensor parallelism, higher replica count
- Decode (D): higher tensor parallelism, lower replica count

For non-PD (colocated) baseline sizing:
- Keep TP as low as possible while fitting model memory
- Scale throughput with replicas first, then increase TP if latency or memory requires

| Model | PVC xPyD | PVC non-PD | BMG xPyD | BMG non-PD | Feasibility Note |
|---|---|---|---|---|---|
| GLM-4-9B | Prefill: TP=1, replicas=1; Decode: TP=1, replicas=1 | TP=1, replicas=2 | Prefill: TP=1, replicas=2 Decode: TP=1, replicas=1 | TP=1, replicas=2 | Strongly feasible on both platforms |
| Qwen3-30B-A3B-Instruct-2507 | Prefill: TP=2, replicas=1; Decode: TP=2, replicas=1 | TP=2, replicas=2 | Prefill: TP=4, replicas=1; Decode: TP=4, replicas=1 | TP=4, replicas=2 | Feasible on both platforms |
| DeepSeek-R1-Distill-Llama-70B | Prefill: TP=4, replicas=1; Decode: TP=4, replicas=1 | TP=4, replicas=2| Prefill: TP=8, replicas=1; Decode: TP=8, replicas=1 | TP=8, replicas=2 | PVC feasible for production trial; BMG is high-risk and capacity-limited |
| Qwen3-235B-A22B-Instruct-2507 | Prefill: TP=8, replicas=1; Decode: TP=8, replicas=1 | TP=8, replicas=2 | Not recommended as first target | Not recommended as first target | Start with PVC only; BMG should be considered experimental |

### 3.3 Tuning PD Split Ratio

Use above matrix as initial ratios, then tune by TTFT/TPOT bottleneck:

- GLM-4-9B: prefill:decode = 2:1
- Qwen3-30B-A3B-Instruct-2507: prefill:decode = 2:1, 3:1
- DeepSeek-R1-Distill-Llama-70B: prefill:decode = 2:1 (PVC), 3:1 (PVC)
- Qwen3-235B-A22B-Instruct-2507: prefill:decode = 2:1 (PVC)

Tuning rule:

- TTFT high and TPOT low: add prefill replicas first (keep prefill TP low)
- TPOT high and TTFT acceptable: increase decode TP first, then add decode replicas only if needed


## 4. Test Case ID Rule and PD-search-sweep

**Goal**: Find break-even point vs colocated serving across the full ISL/OSL/concurrency parameter space.

Case IDs are generated in nested-loop order:
1. ISL group
2. OSL list under that ISL
3. Concurrency list under that ISL

ISL-adaptive profiles:

| ISL | OSL | Concurrency | Case Count |
|---:|---|---|---:|
| 1024 | 128, 1024 | 16, 32, 64 | 6 |
| 2048 | 128, 1024 | 4, 8, 16 | 6 |
| 4096 | 128, 1024 | 1, 4, 8 | 6 |
| 8192 | 128 | 1, 4 | 2 |
| 10240 | 128 | 1, 4 | 2 |
| 12288 | 128 | 1, 4 | 2 |
| 16384 | 128 | 1, 4 | 2 |
| 32768 | 128 | 1, 4 | 2 |

Total logical cases: 28.

Each case runs in 3 modes:
- baseline (non-pd)
- baseline-pd
- llm-d-pd

Total runs: 28 x 3 = 84.

### **Test Matrix**:

- Each row defines one logical case.
- Execute three modes for each row.

| CaseID | ISL | OSL | Concurrency | WarmupReq | MeasuredReq | Modes |
|---|---:|---:|---:|---:|---:|---|
| TC-001 | 1024 | 128 | 16 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-002 | 1024 | 128 | 32 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-003 | 1024 | 128 | 64 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-004 | 1024 | 1024 | 16 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-005 | 1024 | 1024 | 32 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-006 | 1024 | 1024 | 64 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-007 | 2048 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-008 | 2048 | 128 | 8 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-009 | 2048 | 128 | 16 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-010 | 2048 | 1024 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-011 | 2048 | 1024 | 8 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-012 | 2048 | 1024 | 16 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-013 | 4096 | 128 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-014 | 4096 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-015 | 4096 | 128 | 8 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-016 | 4096 | 1024 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-017 | 4096 | 1024 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-018 | 4096 | 1024 | 8 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-019 | 8192 | 128 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-020 | 8192 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-021 | 10240 | 128 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-022 | 10240 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-023 | 12288 | 128 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-024 | 12288 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-025 | 16384 | 128 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-026 | 16384 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-027 | 32768 | 128 | 1 | 30 | 300 | baseline,baseline-pd,llm-d-pd |
| TC-028 | 32768 | 128 | 4 | 30 | 300 | baseline,baseline-pd,llm-d-pd |

## 5. Per-Case Request Template

```json
{
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "messages": [
    {"role": "user", "content": "<PROMPT_WITH_EXACT_ISL_TOKENS>"}
  ],
  "max_tokens": <OSL>,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": false
}
```

## 6. Per-Case Benchmark Command Template

```bash
# baseline: --mode baseline
# PD without llm-d: --mode baseline-pd
# PD with llm-d: --mode llm-d-pd
python benchmark_runner.py \
  --mode llm-d-pd \
  --isl <ISL> \
  --osl <OSL> \
  --concurrency <CONCURRENCY> \
  --warmup-requests 30 \
  --measured-requests 300
```

## 7. Result Record Schema (Per Case Per Mode)

```json
{
  "case_id": "TC-001",
  "mode": "baseline",
  "isl": 512,
  "osl": 512,
  "concurrency": 16,
  "ttft_ms": {"avg": 0.0, "p50": 0.0, "p99": 0.0},
  "tpot_ms": {"avg": 0.0, "p50": 0.0, "p99": 0.0},
  "throughput": {"req_per_s": 0.0, "token_per_s": 0.0},
  "success_rate": 1.0,
  "errors": {"timeout": 0, "http": 0, "other": 0}
}
```

## 8. Break-Even Decision Rule

For the same CaseID, compare pd vs baseline:

- TTFT improvement (%):
  - `ttft_gain = (ttft_baseline_p50 - ttft_pd_p50) / ttft_baseline_p50 * 100`
- Throughput improvement (%):
  - `tps_gain = (tps_pd - tps_baseline) / tps_baseline * 100`

Case-level PD win definition:
- `ttft_gain >= 5` OR `tps_gain >= 10`
- and both modes have `success_rate >= 0.99`

Crossover region definition:
- Minimal (ISL, OSL, concurrency) frontier where PD win starts to hold consistently.


## 9. Acceptance Checklist

- [ ] All 28 CaseID completed in baseline.
- [ ] All 28 CaseID completed in baseline-pd.
- [ ] All 28 CaseID completed in llm-d-pd.
- [ ] No missing required metric fields.
- [ ] Success rate >= 99% for included crossover analysis.
- [ ] Crossover summary generated with case IDs.

