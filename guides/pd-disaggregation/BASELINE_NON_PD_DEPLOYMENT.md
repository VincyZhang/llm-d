# Baseline Non-PD Deployment Guide

This guide keeps only the shortest validated deployment path for baseline non-PD.
Use the existing manifest in `guides/pd-disaggregation/manifest/baseline_non_pd.yaml` directly.

Deployed component:

| Component | Role | Port |
|---|---|---:|
| `vllm-baseline` | non-PD serving StatefulSet | 8000 |

Client endpoint:

- `http://localhost:8084`

Important note:

- The manifest file is currently pinned to namespace `llm-d-baseline`

## 1. Preconditions

- Kubernetes cluster with Intel XPU runtime available
- DRA enabled for GPU
- Intel GPU device class `gpu.intel.com` available
- `kubectl` access
- HuggingFace token available

## 2. Create Namespace and Secret

```bash
export NAMESPACE=llm-d-baseline
export HF_TOKEN=<your_hf_token>

kubectl create namespace ${NAMESPACE}

kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

## 3. Deploy Directly from Manifest

Run from `guides/pd-disaggregation/`:

```bash
kubectl apply -f manifest/baseline_non_pd.yaml

kubectl rollout status -n ${NAMESPACE} statefulset/vllm-baseline -w
```

## 4. Verify

```bash
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}
kubectl get resourceclaim -n ${NAMESPACE}
```

Expected state:

- `vllm-baseline-0` and other replicas are `Running`
- service `vllm-baseline` exists

## 5. Port-Forward

```bash
kubectl port-forward -n ${NAMESPACE} svc/vllm-baseline 8084:8000
```

## 6. Smoke Test

```bash
curl -s http://localhost:8084/v1/models | python -m json.tool
```

## 7. Run Benchmark

```bash
python benchmark_runner.py \
  --mode baseline \
  --case-id TC-001 \
  --model Qwen/Qwen3-0.6B \
  --host localhost \
  --port 8084 \
  --isl 512 \
  --osl 128 \
  --concurrency 16 \
  --warmup-requests 30 \
  --measured-requests 300 \
  --output-file results/TC-001_baseline_non_pd_direct.json
```

## 8. Related Files

- `manifest/baseline_non_pd.yaml`
- `BASELINE_PD_DEPLOYMENT.md`
- `TEST_CONFIGURATION_SPEC.md`
- `benchmark_runner.py`
