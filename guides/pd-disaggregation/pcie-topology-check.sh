#!/bin/bash
# pcie-topology-check.sh — 检查 GPU 和 RDMA NIC 的 PCIe root 对齐情况
# 用法: ./pcie-topology-check.sh
#
# 验证 DRA constraint `matchAttribute: "resource.kubernetes.io/pcieRoot"` 是否生效。
# 可在节点上直接运行，也可通过 kubectl debug node 执行。

set -euo pipefail

echo "=== Intel GPU Devices ==="
for gpu in /sys/class/drm/card*/device; do
    [[ -e "$gpu" ]] || continue
    pci_addr=$(basename "$(readlink -f "$gpu")")
    root=$(readlink -f "$gpu" | grep -oP '\d{4}:\d{2}' | head -1)
    echo "  GPU: $pci_addr (PCIe root: $root)"
done

echo ""
echo "=== RDMA Devices ==="
for rdma in /sys/class/infiniband/*/device; do
    [[ -e "$rdma" ]] || continue
    pci_addr=$(basename "$(readlink -f "$rdma")")
    root=$(readlink -f "$rdma" | grep -oP '\d{4}:\d{2}' | head -1)
    ib_dev=$(basename "$(dirname "$rdma")")
    echo "  RDMA: $ib_dev @ $pci_addr (PCIe root: $root)"
done

echo ""
echo "=== Alignment Check ==="
declare -A gpu_roots nic_roots
for gpu in /sys/class/drm/card*/device; do
    [[ -e "$gpu" ]] || continue
    root=$(readlink -f "$gpu" | grep -oP '\d{4}:\d{2}' | head -1)
    gpu_roots[$root]=1
done
for rdma in /sys/class/infiniband/*/device; do
    [[ -e "$rdma" ]] || continue
    root=$(readlink -f "$rdma" | grep -oP '\d{4}:\d{2}' | head -1)
    nic_roots[$root]=1
done

aligned=0
for root in "${!gpu_roots[@]}"; do
    if [[ -n "${nic_roots[$root]:-}" ]]; then
        echo "  ✅ PCIe root $root has both GPU and RDMA NIC — aligned"
        aligned=1
    fi
done
if [[ $aligned -eq 0 ]]; then
    echo "  ⚠️  No shared PCIe root between GPU and RDMA NIC found"
    exit 1
fi

echo ""
echo "=== ResourceSlice Check (run on a node with kubectl access) ==="
echo "Run the following to verify DRA drivers publish pcieRoot attribute:"
echo ""
echo '  kubectl get resourceslice -o json | jq '"'"'.items[] |'
echo '    select(.spec.driver == "gpu.intel.com" or .spec.driver == "dra.net") |'
echo '    {driver: .spec.driver, devices: [.spec.devices[] |'
echo '      {name: .name, pcieRoot: .attributes["resource.kubernetes.io/pcieRoot"]}]}'"'"''
echo ""
echo "If pcieRoot is null for any device, the DRA constraint has no effect."
