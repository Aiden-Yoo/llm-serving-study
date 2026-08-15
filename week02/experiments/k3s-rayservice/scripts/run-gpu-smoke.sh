#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-k3s}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

for manifest in amd-device-plugin.yaml rocm-smoke-pod.yaml; do
  lxc exec "$INSTANCE" -- rm -f "/tmp/$manifest"
  lxc file push "$LAB_DIR/manifests/$manifest" "$INSTANCE/tmp/$manifest"
done

lxc exec "$INSTANCE" -- bash -s <<'INNER'
set -euo pipefail
kubectl apply -f /tmp/amd-device-plugin.yaml
kubectl rollout status daemonset/amdgpu-device-plugin-daemonset \
  -n kube-system --timeout=180s
for _ in $(seq 1 90); do
  gpu=$(kubectl get node -o jsonpath='{.items[0].status.allocatable.amd\.com/gpu}' 2>/dev/null || true)
  [[ $gpu == 1 ]] && break
  sleep 2
done
[[ $gpu == 1 ]]

kubectl delete pod week2-rocm-smoke --ignore-not-found --wait=true
kubectl apply -f /tmp/rocm-smoke-pod.yaml
for _ in $(seq 1 180); do
  phase=$(kubectl get pod week2-rocm-smoke -o jsonpath='{.status.phase}' 2>/dev/null || true)
  [[ $phase == Succeeded || $phase == Failed ]] && break
  sleep 2
done
kubectl logs week2-rocm-smoke
[[ $phase == Succeeded ]]
INNER
