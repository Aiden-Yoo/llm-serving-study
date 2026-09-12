#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab
require_command jq

SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-4b}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}

apply_manifest "$LAB_DIR/manifests/namespace.yaml"
apply_manifest "$LAB_DIR/manifests/gpu-resource-quota.yaml"
apply_manifest "$LAB_DIR/manifests/vllm-service.yaml"

# 사용자 설정이 반영된 Pod template을 한 번만 적용해 Recreate rollout 중복을 피한다.
deployment="$LAB_DIR/manifests/vllm.yaml"
remote="/tmp/week6-$(basename "$deployment")"
lxc exec "$INSTANCE" -- rm -f "$remote"
lxc file push "$deployment" "$INSTANCE$remote"
patch=$(jq -cn \
  --arg served_model_name "$SERVED_MODEL_NAME" \
  --arg max_model_len "$MAX_MODEL_LEN" \
  --arg gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --arg max_num_seqs "$MAX_NUM_SEQS" \
  --arg max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
  '{spec:{template:{spec:{containers:[{name:"vllm",env:[
    {name:"SERVED_MODEL_NAME",value:$served_model_name},
    {name:"MAX_MODEL_LEN",value:$max_model_len},
    {name:"GPU_MEMORY_UTILIZATION",value:$gpu_memory_utilization},
    {name:"MAX_NUM_SEQS",value:$max_num_seqs},
    {name:"MAX_NUM_BATCHED_TOKENS",value:$max_num_batched_tokens}
  ]}]}}}}')
kube patch --local -f "$remote" --type=strategic -p "$patch" -o yaml | kube apply -f -

if ! kube rollout status deployment/vllm -n week6-llm --timeout=20m; then
  kube get pods -n week6-llm -o wide
  kube describe deployment/vllm -n week6-llm
  kube logs deployment/vllm -n week6-llm --tail=300 || true
  exit 1
fi
service_ip=$(kube get service vllm -n week6-llm -o jsonpath='{.spec.clusterIP}')
lxc exec "$INSTANCE" -- curl -fsS "http://${service_ip}:8000/v1/models"
