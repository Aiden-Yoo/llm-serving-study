#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-kind}
CLUSTER=${CLUSTER:-week2}
KUBERAY_VERSION=${KUBERAY_VERSION:-1.6.0}
HELM_VERSION=${HELM_VERSION:-v3.21.4}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MANIFEST=$LAB_DIR/manifests/rayservice-cpu.yaml

command -v lxc >/dev/null
lxc info "$INSTANCE" >/dev/null

lxc exec "$INSTANCE" -- bash -s -- "$HELM_VERSION" "$KUBERAY_VERSION" <<'INNER'
set -euo pipefail
HELM_VERSION=$1
KUBERAY_VERSION=$2

if ! command -v helm >/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 -o /tmp/get_helm.sh
  chmod 700 /tmp/get_helm.sh
  DESIRED_VERSION=$HELM_VERSION /tmp/get_helm.sh
fi

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --version "$KUBERAY_VERSION" \
  --namespace kuberay-system \
  --create-namespace \
  --wait --timeout 5m
kubectl wait --for=condition=Available deployment/kuberay-operator \
  -n kuberay-system --timeout=180s
INNER

lxc exec "$INSTANCE" -- rm -f /tmp/rayservice-cpu.yaml
lxc file push "$MANIFEST" "$INSTANCE/tmp/rayservice-cpu.yaml"
lxc exec "$INSTANCE" -- kubectl apply -f /tmp/rayservice-cpu.yaml

for _ in $(seq 1 120); do
  status=$(lxc exec "$INSTANCE" -- kubectl get rayservice week2-rayservice \
    -o jsonpath='{.status.serviceStatus}' 2>/dev/null || true)
  [[ $status == Running ]] && break
  sleep 5
done
[[ ${status:-} == Running ]]

lxc exec "$INSTANCE" -- bash -s <<'INNER'
set -euo pipefail
kubectl get rayservice,raycluster,pods,svc -o wide
kubectl delete pod week2-rayservice-curl --ignore-not-found --wait=true >/dev/null
kubectl run week2-rayservice-curl --restart=Never --image=curlimages/curl:8.16.0 \
  --command -- sleep 300
kubectl wait --for=condition=Ready pod/week2-rayservice-curl --timeout=120s
serve_service=week2-rayservice-serve-svc
fruit=$(kubectl exec week2-rayservice-curl -- curl -fsS \
  -X POST -H 'Content-Type: application/json' \
  --data '["MANGO", 2]' "http://${serve_service}:8000/fruit/")
calc=$(kubectl exec week2-rayservice-curl -- curl -fsS \
  -X POST -H 'Content-Type: application/json' \
  --data '["MUL", 3]' "http://${serve_service}:8000/calc/")
printf 'FRUIT_RESPONSE=%s\nCALC_RESPONSE=%s\n' "$fruit" "$calc"
[[ $fruit == 6 ]]
[[ $calc == '15 pizzas please!' ]]
INNER
