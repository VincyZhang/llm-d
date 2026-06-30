#!/usr/bin/env bash
set -euo pipefail

# Known bad UCX debug path in this environment can break XPU runtime detection.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | sed -E 's#(^|:)/tmp/ucx_install/lib(:|$)#\1#g; s#::#:#g; s#^:##; s#:$##')"
  export LD_LIBRARY_PATH
fi

# oneAPI environment setup is required for XPU runtime libraries.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  # Prefer force mode because some base images source setvars early with stale env.
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-xpu}"

exec python -m vllm.entrypoints.openai.api_server "$@"
