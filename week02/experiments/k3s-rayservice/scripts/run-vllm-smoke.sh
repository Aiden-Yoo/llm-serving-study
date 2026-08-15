#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-k3s}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MANIFEST=$LAB_DIR/manifests/vllm-smoke.yaml

lxc exec "$INSTANCE" -- test -d /lab-assets/runtime/vllm
lxc exec "$INSTANCE" -- test -d /lab-assets/models/Qwen3-0.6B
lxc exec "$INSTANCE" -- test -d /lab-assets/rocm
lxc exec "$INSTANCE" -- rm -f /tmp/vllm-smoke.yaml
lxc file push "$MANIFEST" "$INSTANCE/tmp/vllm-smoke.yaml"

lxc exec "$INSTANCE" -- bash -s <<'INNER'
set -euo pipefail
kubectl delete -f /tmp/vllm-smoke.yaml --ignore-not-found --wait=true
kubectl apply -f /tmp/vllm-smoke.yaml
for _ in $(seq 1 300); do
  phase=$(kubectl get pod week2-vllm -o jsonpath='{.status.phase}' 2>/dev/null || true)
  ready=$(kubectl get pod week2-vllm -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || true)
  [[ $ready == true || $phase == Failed ]] && break
  sleep 2
done
if [[ $ready != true ]]; then
  kubectl logs week2-vllm --tail=300
  exit 1
fi
kubectl delete pod week2-vllm-curl --ignore-not-found --wait=true >/dev/null
kubectl run week2-vllm-curl --restart=Never --image=curlimages/curl:8.16.0 \
  --command -- sleep 300
kubectl wait --for=condition=Ready pod/week2-vllm-curl --timeout=120s
kubectl exec week2-vllm-curl -- curl -fsS http://week2-vllm:8000/v1/models
INNER
