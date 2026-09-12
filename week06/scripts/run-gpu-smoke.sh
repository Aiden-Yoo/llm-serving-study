#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

apply_manifest "$LAB_DIR/manifests/amd-device-plugin.yaml"
kube rollout status daemonset/amdgpu-device-plugin-daemonset -n kube-system --timeout=180s
for _ in $(seq 1 90); do
  gpu=$(kube get node -o jsonpath='{.items[0].status.allocatable.amd\.com/gpu}' 2>/dev/null || true)
  [[ $gpu == 1 ]] && break
  sleep 2
done
[[ ${gpu:-} == 1 ]] || { echo "amd.com/gpu was not registered" >&2; exit 1; }

apply_manifest "$LAB_DIR/manifests/namespace.yaml"
apply_manifest "$LAB_DIR/manifests/gpu-resource-quota.yaml"

# 완료된 smoke 결과가 있으면 재사용한다. 실행 중인 vLLM을 건드리지 않고도
# 전체 실습 뒤 검증 명령을 다시 실행할 수 있게 하기 위함이다.
phase=$(kube get pod week6-rocm-smoke -n week6-llm \
  -o jsonpath='{.status.phase}' 2>/dev/null || true)
if [[ $phase == Succeeded ]] && \
  kube logs week6-rocm-smoke -n week6-llm | grep -q '^GPU_SMOKE_OK$'; then
  kube logs week6-rocm-smoke -n week6-llm | tee "$LAB_DIR/results/gpu-smoke.log"
  echo 'Reusing the completed GPU smoke result.'
  exit 0
fi

gpu_in_use=$(kube get resourcequota single-amd-gpu -n week6-llm \
  -o jsonpath='{.status.used.requests\.amd\.com/gpu}' 2>/dev/null || true)
if [[ ${gpu_in_use:-0} != 0 ]]; then
  echo 'GPU is already allocated in week6-llm; run the smoke test before deploying vLLM.' >&2
  exit 1
fi

kube delete pod week6-rocm-smoke -n week6-llm --ignore-not-found --wait=true
apply_manifest "$LAB_DIR/manifests/rocm-smoke-pod.yaml"
if ! kube wait --for=jsonpath='{.status.phase}'=Succeeded pod/week6-rocm-smoke \
  -n week6-llm --timeout=360s; then
  kube describe pod week6-rocm-smoke -n week6-llm
  kube logs week6-rocm-smoke -n week6-llm --tail=300 || true
  exit 1
fi
kube logs week6-rocm-smoke -n week6-llm | tee "$LAB_DIR/results/gpu-smoke.log"
