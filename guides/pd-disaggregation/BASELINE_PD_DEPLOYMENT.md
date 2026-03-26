# Baseline PD Deployment Guide
## Co-located vLLM with Prefill / Decode / Toy-Proxy on the Same Namespace

This document provides step-by-step instructions for deploying a minimal
**co-located PD disaggregation** setup in a single Kubernetes namespace:

| Pod | Role | API Port | NIXL KV Port |
|-----|------|----------|--------------|
| `vllm-prefill` | Prefill (P) — runs `kv_role: kv_both` | 8000 | 5600 |
| `vllm-decode`  | Decode  (D) — runs `kv_role: kv_both` | 8200 | 5500 |
| `toy-proxy`    | Proxy — local script routes requests P→D via NIXL | 8085 | — |

Clients call **`http://localhost:8085`** (toy-proxy), which is the `endpoint_pd`
defined in `TEST_CONFIGURATION_SPEC.md` section 2.

---

## 1. Prerequisites

- Kubernetes v1.29+ cluster with Intel GPU plugin deployed
- `kubectl` access with namespace create / deploy privileges
- HuggingFace token secret created in target namespace
- GPU DRA available with `deviceClassName: gpu.intel.com`
- RDMA DRA enabled, and claim template file available at `ms-pd/rdma-resource-claims.yaml`
- `toy_proxy_server.py` available in container image or mounted

### Create namespace and secret
# Baseline PD Deployment Guide
## Validated GPU Claim + RDMA Claim Deployment

This document keeps only the currently validated PD deployment path in
`llm-d-pd`: prefill and decode both use explicit Intel GPU DRA claims and RDMA
DRA claims, and requests enter through `toy-proxy`.

| Component | Role | Port |
|---|---|---:|
| `vllm-prefill` | Prefill service | 8000 |
| `vllm-decode` | Decode service | 8200 |
| `toy-proxy` | PD entrypoint | 8085 |

Client endpoint:

- `http://localhost:8085`

Important notes:

- The manifest files are currently pinned to namespace `llm-d-pd`
- RDMA claims must be created before applying the PD manifests

## 1. Preconditions

- Kubernetes cluster with Intel XPU runtime available
- DRA enabled for GPU and RDMA
- Intel GPU device class `gpu.intel.com` available
- `kubectl` access
- HuggingFace token available

## 2. Create Namespace, Secret, and RDMA Claim Template

```bash
export NAMESPACE=llm-d-pd
export HF_TOKEN=<your_hf_token>

kubectl create namespace ${NAMESPACE}

kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}

kubectl apply -n ${NAMESPACE} -f ms-pd/rdma-resource-claims.yaml
```

## 3. Deploy Directly from Manifest

Run from `guides/pd-disaggregation/`:

```bash
kubectl apply -f manifest/prefill-deployment.yaml
kubectl apply -f manifest/decode-deployment.yaml
kubectl apply -f manifest/proxy-deployment.yaml

kubectl rollout status deployment/vllm-prefill -n ${NAMESPACE} -w
kubectl rollout status deployment/vllm-decode -n ${NAMESPACE} -w
kubectl rollout status deployment/toy-proxy -n ${NAMESPACE} -w
```

## 4. Verify

```bash
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}
kubectl get resourceclaim -n ${NAMESPACE}
```

Expected state:

- `vllm-prefill` is `Running`
- `vllm-decode` is `Running`
- `toy-proxy` is `Running`

## 5. Port-Forward

```bash
kubectl port-forward -n ${NAMESPACE} svc/toy-proxy 8085:8085
```

## 6. Smoke Test

```bash
curl -s http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
    "temperature": 0.0
  }' | python -m json.tool
```

## 7. Run Benchmark

```bash
python benchmark_runner.py \
  --mode baseline-pd \
  --case-id TC-001 \
  --model Qwen/Qwen3-0.6B \
  --host localhost \
  --port 8085 \
  --isl 512 \
  --osl 512 \
  --concurrency 16 \
  --warmup-requests 30 \
  --measured-requests 300 \
  --output-file results/TC-001_pd_toyproxy_qwen.json
```

## 8. Related Files

- `manifest/prefill-deployment.yaml`
- `manifest/decode-deployment.yaml`
- `manifest/proxy-deployment.yaml`
- `ms-pd/rdma-resource-claims.yaml`
- `TEST_CONFIGURATION_SPEC.md`
