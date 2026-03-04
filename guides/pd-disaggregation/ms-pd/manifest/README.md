# PD Disaggregation Manifests

This directory contains curated manifests for PD disaggregation and RDMA claims.

## RDMA overlay (kustomize)

Location: guides/pd-disaggregation/manifest/rdma

Files:
- pd_rdma_deploy.yaml: Base rendered manifest for PD disaggregation (xpu_rdma) used as the kustomize input.
- rdma-resource-claims.yaml: RDMA ResourceClaimTemplate definitions for decode/prefill.
- pd_patch.yaml: Patch that injects RDMA claim references into decode/prefill pods and containers.
- kustomization.yaml: Kustomize entry that wires the base manifest, RDMA templates, and patch together.

How to render/apply:
- Render: kubectl kustomize guides/pd-disaggregation/manifest/rdma
- Apply:  kubectl apply -k guides/pd-disaggregation/manifest/rdma
