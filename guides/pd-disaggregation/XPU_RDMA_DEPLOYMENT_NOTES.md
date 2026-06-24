# XPU RDMA PD Disaggregation — Deployment Notes & Security Configuration Analysis

**JIRA**: [OTCK8-3032](https://jira.devtools.intel.com/browse/OTCK8-3032)  
**Upstream**: https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation/modelserver/xpu/vllm-rdma  
**Validated**: 2026-05-18  
**Environment**: Intel Data Center GPU Max 1550 (22.71 GiB), single-node K8s, DRA drivers (`gpu.intel.com` + `rdma-dranet`)

---

## 1. Current Solution

### Deployment Architecture

```
guides/pd-disaggregation/modelserver/xpu/
├── vllm/                  ← XPU TCP (base)
│   ├── kustomization.yaml
│   ├── patch-decode.yaml
│   ├── patch-prefill.yaml
│   ├── resource-claim-templates.yaml
│   └── namereference.yaml
└── vllm-rdma/             ← XPU RDMA (overlay on vllm/)
    └── kustomization.yaml   ← Adds RDMA NIC claim + PCIe alignment + UCX transport switch
```

### Core Design

1. **DRA ResourceClaimTemplate** requests both GPU + RDMA NIC, using `constraints.matchAttribute: "resource.kubernetes.io/pcieRoot"` to enforce GPU and NIC allocation under the same PCIe root complex
2. **UCX Transport**: `UCX_TLS=ib,rc,ze_copy` (InfiniBand RC + Level-Zero GPU memory copy)
3. **KV Buffer**: `kv_buffer_device=xpu` (KV cache stored in GPU VRAM, RDMA reads directly from GPU)
4. **Zero Privilege**: No privileged mode, IPC_LOCK, NET_RAW, or ZE_AFFINITY_MASK required

### Upstream Links

| Resource | Link |
|----------|------|
| Repository | https://github.com/llm-d/llm-d |
| XPU RDMA overlay | https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation/modelserver/xpu/vllm-rdma |
| XPU base | https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation/modelserver/xpu/vllm |
| E2E CI workflow | https://github.com/llm-d/llm-d/blob/main/.github/workflows/e2e-pd-xpu.yaml |
| PD Guide README | https://github.com/llm-d/llm-d/blob/main/guides/pd-disaggregation/README.md |

---

## 2. GPU-NIC PCIe Alignment Verification

### Problem

The current solution declares GPU-NIC co-location via DRA constraint `matchAttribute: "resource.kubernetes.io/pcieRoot"`. However, there is **no way to confirm from inside the Pod** whether the scheduler actually satisfied this constraint.

Key dependency: DRA drivers (`rdma-dranet` and `gpu.intel.com`) must publish `resource.kubernetes.io/pcieRoot` for each device. If missing, the constraint is vacuously satisfied (all devices match) and alignment is not guaranteed.

### Verification Tool

See [pcie-topology-check.sh](./pcie-topology-check.sh), which provides:
- Node-level GPU and RDMA NIC PCIe root topology enumeration
- Automatic check for shared PCIe root GPU-NIC pairs
- ResourceSlice attribute inspection commands

### Quick Verification Commands

```bash
# 1. Check whether DRA drivers publish the pcieRoot attribute
kubectl get resourceslice -o json | jq '.items[] |
  select(.spec.driver == "gpu.intel.com" or .spec.driver == "dra.net") |
  {driver: .spec.driver, devices: [.spec.devices[] |
    {name: .name, pcieRoot: .attributes["resource.kubernetes.io/pcieRoot"]}]}'

# 2. Inspect the allocated ResourceClaim (most reliable method)
CLAIM=$(kubectl get pod <pod-name> -n <ns> -o jsonpath='{.spec.resourceClaims[0].resourceClaimName}')
kubectl get resourceclaim $CLAIM -n <ns> -o yaml
```

> **Note**: `/sys/class/drm` and `/sys/class/infiniband` inside a Pod show all host devices (sysfs is not namespace-isolated) and are unsuitable for alignment verification. Use ResourceClaim allocation results instead.

---

## 3. Security Configuration: Why No Privilege/Capability/Affinity Mask Is Needed

The current solution achieves **zero-privilege deployment** via DRA. Pod securityContext only requires `fsGroup: 107`. Breakdown:

| Setting | Required? | Reason |
|---------|-----------|--------|
| `privileged: true` | **No** | DRA driver automatically exposes GPU (`/dev/dri`) and RDMA (`/dev/infiniband`) device nodes |
| `ZE_AFFINITY_MASK` | **No** | DRA exposes only the single allocated GPU to the container |
| `CAP_IPC_LOCK` | **No** | Bare-metal cluster memlock ulimit = unlimited; kernel permits `mlock()` unconditionally |
| `CAP_NET_RAW` | **No** | UCX uses IB verbs (`ib,rc`), no raw sockets |

### IPC_LOCK Platform Dependency

RDMA memory registration (`ibv_reg_mr`) requires `mlock()`. Linux kernel logic: if `RLIMIT_MEMLOCK == RLIM_INFINITY`, `mlock()` is permitted without checking `CAP_IPC_LOCK`. Platform comparison:

| Environment | Default memlock | IPC_LOCK needed? |
|-------------|-----------------|------------------|
| Bare-metal cluster (current) | unlimited | **No** |
| AKS (Azure) | 64 KiB → raised by NRI ulimit-adjuster | **No** |
| OCI (Oracle) | restricted | **Yes** |

### Comparison With Other Platforms

- **HPU (Gaudi)**: Still requires `privileged: true` (habana driver limitation)
- **AMD on OCI**: Requires `CAP_IPC_LOCK` (restricted memlock)
- **Intel XPU (DRA)**: Zero privilege

References: [llm-d AKS docs](https://github.com/llm-d/llm-d/blob/main/docs/infra-providers/aks/README.md) | [OCI AMD config](https://github.com/llm-d/llm-d/blob/main/guides/pd-disaggregation/modelserver/amd/vllm/oci/kustomization.yaml)

---

## 4. Improvements Needed

### 4.1 `max-model-len` Default Too Large

**Symptom**: `--max-model-len=32000` allocates 20.16 GiB / 22.71 GiB for KV cache. After NIXL buffer registration, warmup inference OOMs (`UR_RESULT_ERROR_OUT_OF_RESOURCES`).

**Recommendation**: Reduce to `8192` or add `--gpu-memory-utilization=0.7`.

### 4.2 `VLLM_USE_V1` Environment Variable Deprecated

**Symptom**: `WARNING: Unknown vLLM environment variable detected: VLLM_USE_V1`

**Recommendation**: Remove; v0.7.0 uses V1 engine by default.

### 4.3 CPU Resource Requests Too High

**Symptom**: Decode requests 16 CPU; with EPP at 4 CPU, single-node easily overcommits.

**Recommendation**: Reduce vLLM container to 4 CPU, EPP to 1-2 CPU.

### 4.4 Prefill Defaults to 3 Replicas

**Recommendation**: For CI scenarios, 1 prefill + 1 decode is sufficient.

### 4.5 README Should Document Security Configuration Rationale

**Recommendation**: Add the following to the kustomization comment or README:

> This deployment requires no privileged mode, IPC_LOCK, or ZE_AFFINITY_MASK:
> - DRA handles device allocation and exposure (GPU + RDMA NIC)
> - Container runtime memlock ulimit is unlimited (bare-metal default)
> - DRA exposes only the allocated GPU, making affinity masks redundant
>
> On platforms with restricted memlock (e.g. AKS 64KiB default), configure
> the NRI ulimit-adjuster plugin or add CAP_IPC_LOCK to securityContext.

---

## Validation Summary

| Test | Result |
|------|--------|
| Deployment method | `kubectl apply -k modelserver/xpu/vllm-rdma` + helm standalone router |
| Model | Qwen/Qwen3-0.6B |
| Transport | UCX `ib,rc,ze_copy` |
| KV buffer | `xpu` (GPU memory) |
| GPU memory utilization | 0.7 (adjusted) |
| max-model-len | 8192 (adjusted) |
| E2E test | 10/10 iterations passed (`e2e-validate.sh`) |
| Security context | `fsGroup: 107` only — zero privilege |
