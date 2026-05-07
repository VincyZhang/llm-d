# Intel XPU PD Disaggregation Deployment Guide

## Overview

Intel XPU support in the current P/D guide follows the latest `main` layout under `guides/pd-disaggregation/modelserver/`.
Use the generic [P/D disaggregation guide](./README.md) for the Router, Gateway, and InferencePool setup, then choose one of these XPU modelserver overlays:

| Variant | Directory | Transport | Use Case |
|---|---|---|---|
| **Standard XPU** | `modelserver/xpu/vllm` | UCX over TCP | Default deployment for Intel GPUs |
| **XPU + RDMA** | `modelserver/xpu/vllm-rdma` | UCX over InfiniBand (`ib,rc,ze_copy`) | Clusters with RDMA-capable NICs and an RDMA DRA driver |

## Prerequisites

* Follow the shared [P/D disaggregation guide](./README.md) through Router and InferencePool deployment.
* Intel GPU DRA driver deployed and exposing the `gpu.intel.com` device class.
* For the RDMA variant, an RDMA DRA driver exposing the `rdma-dranet` device class.
* A valid Hugging Face token stored in the `llm-d-hf-token` secret.

## Deploy the Model Server

### Standard XPU

```bash
kubectl apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/xpu/vllm
```

### XPU + RDMA

```bash
kubectl apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/xpu/vllm-rdma
```

The RDMA overlay reuses the standard XPU vLLM base and adds one RDMA DRA claim per pod plus RDMA-specific UCX transport settings.

## Verify Deployment

```bash
kubectl get pods -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode -w
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -w
```

```bash
DECODE_POD=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n ${NAMESPACE} ${DECODE_POD} -c modelserver -f
```

```bash
PREFILL_POD=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n ${NAMESPACE} ${PREFILL_POD} -c modelserver -f
```

## Cleanup

```bash
kubectl delete -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/xpu/vllm-rdma
# Or for standard XPU:
# kubectl delete -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/xpu/vllm
```