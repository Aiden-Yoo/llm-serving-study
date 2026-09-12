#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

HELM_VERSION=${HELM_VERSION:-v3.21.4}
PROMETHEUS_STACK_VERSION=${PROMETHEUS_STACK_VERSION:-90.1.1}
AMD_EXPORTER_VERSION=${AMD_EXPORTER_VERSION:-v1.5.1}

lxc exec "$INSTANCE" -- bash -s -- \
  "$HELM_VERSION" "$PROMETHEUS_STACK_VERSION" "$AMD_EXPORTER_VERSION" <<'INNER'
set -euo pipefail
HELM_VERSION=$1
PROMETHEUS_STACK_VERSION=$2
AMD_EXPORTER_VERSION=$3
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

if ! command -v helm >/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
    -o /tmp/get_helm.sh
  chmod 700 /tmp/get_helm.sh
  DESIRED_VERSION="$HELM_VERSION" /tmp/get_helm.sh
fi
[[ $(helm version --template '{{.Version}}') == "$HELM_VERSION" ]]
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts --force-update >/dev/null
helm repo add exporter https://rocm.github.io/device-metrics-exporter \
  --force-update >/dev/null
helm repo update >/dev/null

helm upgrade --install amd-device-metrics \
  exporter/device-metrics-exporter-charts \
  --version "$AMD_EXPORTER_VERSION" \
  --namespace kube-amd-gpu --create-namespace \
  --set image.pullPolicy=IfNotPresent \
  --set serviceMonitor.enabled=false \
  --wait --timeout 10m
INNER

values=/tmp/week6-monitoring-values.yaml
lxc exec "$INSTANCE" -- rm -f "$values"
lxc file push "$LAB_DIR/manifests/monitoring-values.yaml" "$INSTANCE$values"

lxc exec "$INSTANCE" -- bash -s -- \
  "$PROMETHEUS_STACK_VERSION" "$AMD_EXPORTER_VERSION" <<'INNER'
set -euo pipefail
PROMETHEUS_STACK_VERSION=$1
AMD_EXPORTER_VERSION=$2
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

helm upgrade --install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack \
  --version "$PROMETHEUS_STACK_VERSION" \
  --namespace monitoring --create-namespace \
  --values /tmp/week6-monitoring-values.yaml \
  --wait --timeout 15m

helm upgrade --install amd-device-metrics \
  exporter/device-metrics-exporter-charts \
  --version "$AMD_EXPORTER_VERSION" \
  --namespace kube-amd-gpu --create-namespace \
  --set image.pullPolicy=IfNotPresent \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.honorLabels=true \
  --wait --timeout 10m
INNER

apply_manifest "$LAB_DIR/manifests/vllm-servicemonitor.yaml"
kube get pods -n monitoring
kube get pods -n kube-amd-gpu
