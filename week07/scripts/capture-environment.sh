#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab
require_command amd-smi
require_command python3

mkdir -p "$LAB_DIR/results"
node_gpu=$(kube get node "$INSTANCE" -o jsonpath='{.status.allocatable.amd\.com/gpu}')
vllm_version=$(kube exec -n "$NAMESPACE" deployment/vllm -- \
  /runtime/vllm/bin/python -c 'import vllm; print(vllm.__version__)')
epp_image=$(kube get deployment "${RELEASE}-epp" -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec.containers[?(@.name=="epp")].image}')
envoy_image=$(kube get deployment "${RELEASE}-epp" -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec.containers[?(@.name=="envoy-proxy")].image}')
k3s_version=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["items"][0]["status"]["nodeInfo"]["kubeletVersion"])' \
  <<<"$(kube get nodes -o json)")
gpu_name=$(amd-smi static --asic 2>/dev/null | awk -F': ' '/MARKET_NAME:/ {print $2; exit}')
[[ -n $gpu_name ]] || {
  echo "failed to detect GPU market name with amd-smi" >&2
  exit 1
}

python3 - "$LAB_DIR/results/environment.json" <<PY
import json
import sys
from datetime import datetime, timezone

document = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "hardware": {
        "gpu": ${gpu_name@Q},
        "gpu_count": 1,
    },
    "kubernetes": {
        "distribution": "k3s",
        "version": ${k3s_version@Q},
        "node_allocatable_amd_gpu": ${node_gpu@Q},
        "namespace": ${NAMESPACE@Q},
    },
    "model_server": {
        "engine": "vLLM",
        "version": ${vllm_version@Q},
        "model": "Qwen3-4B-Instruct-2507",
        "served_model_name": "qwen3-4b",
        "max_num_seqs": 16,
        "max_model_len": 4096,
    },
    "flow_control": {
        "llm_d_guide": "v0.9.0",
        "router_chart": "v0.10.0",
        "router_chart_digest": "sha256:72e2478ffe79d0310bf56aaabc3655ec2028ce6fae809674c826c07a5808936e",
        "gaie_crds": "v1.5.0",
        "router_crds": "v0.10.0",
        "epp_image": ${epp_image@Q},
        "envoy_image": ${envoy_image@Q},
        "max_concurrency": 16,
        "global_max_requests": 32,
        "default_request_ttl": "3s",
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2)
    handle.write("\n")
print(json.dumps(document, indent=2))
PY
