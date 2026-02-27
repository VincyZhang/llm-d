# DRA Resources Helm Chart

这是一个用于部署 Kubernetes Dynamic Resource Allocation (DRA) 资源的 Helm chart，特别是用于支持 RDMA 网络设备的 DeviceClass 和 ResourceClaimTemplate。

## 概述

DRA Resources chart 自动化部署以下 Kubernetes 资源：

- **DeviceClass**: 定义可用的设备类型和选择器
- **ResourceClaimTemplate**: 为工作负载提供设备资源声明的模板

## 自动部署流程

通过 helmfile 进行部署时，DRA 资源会自动创建，无需手动执行 `kubectl apply`。

### 部署步骤

编辑后直接运行：

```bash
cd guides/pd-disaggregation

# 部署 XPU 配置（包括 DRA 资源）
helmfile apply -e xpu -n ${NAMESPACE}
```

**部署顺序**：

1. **dra-pd** release：部署 DeviceClass 和 ResourceClaimTemplate
2. **infra-pd** release：部署网关基础设施
3. **gaie-pd** release：部署推理池
4. **ms-pd** release：部署模型服务（自动使用 DRA 资源）

## 配置说明

### 基础 RDMA 配置（values.yaml）

```yaml
rdmaDeviceClass:
  enabled: true
  name: rdma-dranet  # 基础 RDMA DeviceClass 名称

rdmaResourceClaimTemplate:
  enabled: true
  name: rdma-net-template  # 基础 RDMA 模板名称
```

### Intel XPU 特定配置

对于 Intel XPU 加速器，chart 会自动创建以下资源：

```yaml
intelXpuRdmaDeviceClass:
  enabled: true  # 当 xpu 环境启用时自动设置
  name: intel-xpu-rdma

intelXpuResourceClaimTemplate:
  enabled: true  # 当 xpu 环境启用时自动设置
  decodeName: intel-decode-claim-template  # decode 阶段的模板
  prefillName: intel-prefill-claim-template  # prefill 阶段的模板
```

## 与 values_xpu.yaml 的集成

helmfile 在部署 XPU 环境时应用以下文件顺序：

```
1. ms-pd/values_xpu.yaml           # 基础 XPU 配置
2. ms-pd/values_xpu_dra.yaml       # DRA override（覆盖 resourceClaimTemplateName）
```

这样可以自动使用正确的 Intel XPU DRA 模板名称：

**values_xpu.yaml 中的内容**：
```yaml
resourceClaims:
  - name: rdma-net-interface
    resourceClaimTemplateName: rdma-net-template
```

**values_xpu_dra.yaml 中的内容**（覆写）：
```yaml
resourceClaims:
  - name: intel-decode-rdma
    resourceClaimTemplateName: intel-decode-claim-template  # Intel XPU 特定
```

## 验证部署

部署完成后，验证 DRA 资源是否正确创建：

```bash
# 查看 DeviceClass
kubectl get deviceclass -n ${NAMESPACE}

# 查看 ResourceClaimTemplate
kubectl get resourceclaimtemplate -n ${NAMESPACE}

# 验证模板内容
kubectl describe resourceclaimtemplate intel-decode-claim-template -n ${NAMESPACE}

# 检查 Pod 是否正确引用 ResourceClaim
kubectl get pod -n ${NAMESPACE} -o jsonpath='{.items[0].spec.containers[0].resourceClaims}'
```

## 故障排查

### 问题：DeviceClass 未创建
- 检查 values.yaml 中的 `enabled` 设置
- 验证 Kubernetes 版本支持 DRA（v1.29.0+）

### 问题：ResourceClaimTemplate 名称不匹配
- 确认正在使用的 values_xpu_dra.yaml override 文件
- 检查 templateName 字段是否与 DRA resources chart 中定义的名称一致

### 问题：Pod 启动失败（ResourceClaim 相关错误）
- 验证 DRA resource driver 已部署
- 检查 DeviceClass 选择器是否匹配集群中可用的设备

## 参考

- [Kubernetes DRA 文档](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
- [Intel Resource Drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes)
- [llm-d PD Disaggregation 部署指南](../README.xpu.md)
