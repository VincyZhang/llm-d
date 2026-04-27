# Intel XPU PD Disaggregation Deployment Guide

## Overview

This document provides complete steps for deploying Intel XPU PD (Prefill-Decode) disaggregation service on Kubernetes using the Qwen3-0.6B model. PD disaggregation separates the prefill and decode phases of inference, allowing for more efficient resource utilization and improved throughput.

Two deployment variants are available:

| Variant | Directory | Transport | Use Case |
|---|---|---|---|
| **Standard XPU** | `manifests/modelserver/xpu` | UCX over TCP | Default — works on any cluster with Intel GPUs |
| **XPU + RDMA** | `manifests/modelserver/xpu-rdma` | UCX over InfiniBand (`ib,rc,ze_copy`) | High-performance — requires RDMA-capable NICs and DRA RDMA driver |

## Prerequisites

### Hardware Requirements

* Intel Data Center GPU Max 1550 or compatible Intel XPU device
* At least 8GB system memory
* Sufficient disk space (recommended at least 50GB available)
* **RDMA variant only**: RDMA-capable network interface (InfiniBand or RoCE)

### Software Requirements

* Kubernetes cluster (v1.29.0+)
* [Gateway API Inference Extension CRDs](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/v1.4.0/config/crd) installed
* Intel GPU DRA driver deployed (provides `gpu.intel.com` device class)
* **RDMA variant only**: RDMA DRA network driver (`rdma-dranet` device class) deployed
* kubectl and helm installed — see [client setup](../../helpers/client-setup/README.md)

### Client Setup

* Create a namespace for installation.

  ```bash
  export NAMESPACE=llm-d-pd # or any other namespace (shorter names recommended)
  kubectl create namespace ${NAMESPACE}
  ```

* [Create the `llm-d-hf-token` secret in your target namespace with the key `HF_TOKEN` matching a valid HuggingFace token](../../helpers/hf-token.md) to pull models.

## Step 0: Build Intel XPU Docker Image (Optional)

If you need to customize the vLLM version or build the image from source:

```shell
# Build with the default vLLM version from docker/common-versions
make image-build DEVICE=xpu VERSION=v0.6.0
```

> **Note**: If you're using the pre-built `ghcr.io/llm-d/llm-d-xpu:v0.6.0` image, skip this step.

## Step 1: Install Gateway API Dependencies

```shell
cd guides/prereq/gateway-provider
./install-gateway-provider-dependencies.sh
```

## Step 2: Deploy Gateway Control Plane

```shell
cd guides/prereq/gateway-provider
helmfile apply -f istio.helmfile.yaml
```

## Step 3: Deploy the InferencePool (Helm)

Navigate to the `guides/pd-disaggregation` directory:

```bash
cd guides/pd-disaggregation
```

Deploy the InferencePool using the shared values file:

```bash
helm install llm-d-infpool \
  -n ${NAMESPACE} \
  -f ./manifests/inferencepool.values.yaml \
  --set "provider.name=istio" \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --version v1.4.0
```

> **Note**: Change `--set "provider.name=istio"` to match your gateway provider (e.g., `gke`, `none`).

## Step 4: Deploy the Model Server (Kustomize)

### Standard XPU (TCP transport)

```bash
kubectl apply -n ${NAMESPACE} -k ./manifests/modelserver/xpu
```

### XPU + RDMA (InfiniBand transport)

```bash
kubectl apply -n ${NAMESPACE} -k ./manifests/modelserver/xpu-rdma
```

> **Note**: The RDMA overlay inherits the `xpu/` base and adds RDMA-specific DRA claims
> and UCX transport configuration. No privileged mode is needed — DRA handles device allocation.

### Deployment Architecture

* **Decode Service**: 1 replica with 1 Intel GPU (port 8200, routing sidecar on port 8000)
* **Prefill Service**: 1 replica with 1 Intel GPU (port 8000)
* **RDMA variant**: Both decode and prefill use `UCX_TLS=ib,rc,ze_copy` for RDMA transport

## Step 5: Verify Deployment

### Check Pod Status

```shell
kubectl get pods -n ${NAMESPACE}

# Monitor decode pod startup (real-time)
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode -w

# Monitor prefill pods startup (real-time)
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -w
```

### Check InferencePool

```shell
helm list -n ${NAMESPACE}
```

### View vLLM Startup Logs

```shell
# Decode pod logs
DECODE_POD=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n ${NAMESPACE} ${DECODE_POD} -c vllm -f

# Prefill pod logs
PREFILL_POD=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n ${NAMESPACE} ${PREFILL_POD} -c vllm -f
```

## Step 6: Create HTTPRoute for Gateway Access

```shell
kubectl apply -f httproute.yaml -n ${NAMESPACE}

# Verify
kubectl get httproute -n ${NAMESPACE}
```

## Step 7: Test PD Disaggregation Inference Service

### Port Forwarding

```shell
kubectl port-forward -n ${NAMESPACE} service/llm-d-infpool 8086:80 &
```

### Perform Inference Request

```shell
curl -X POST "http://localhost:8086/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Explain the benefits of prefill-decode disaggregation in LLM inference"
      }
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

## Cleanup

```bash
# Remove model server
kubectl delete -n ${NAMESPACE} -k ./manifests/modelserver/xpu-rdma
# Or for standard XPU:
# kubectl delete -n ${NAMESPACE} -k ./manifests/modelserver/xpu

# Remove InferencePool
helm uninstall llm-d-infpool -n ${NAMESPACE}

# Remove namespace
kubectl delete namespace ${NAMESPACE}
```
