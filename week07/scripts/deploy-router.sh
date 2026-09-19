#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

GAIE_VERSION=${GAIE_VERSION:-v1.5.0}
ROUTER_VERSION=${ROUTER_VERSION:-v0.10.0}
CHART_VERSION=${CHART_VERSION:-v0.10.0}
CHART=${CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-standalone}
REMOTE_VALUES=/tmp/week7-router-values.yaml
REMOTE_OBJECTIVES=/tmp/week7-inference-objectives.yaml

kube wait --for=condition=available deployment/vllm -n "$NAMESPACE" --timeout=300s
kube apply -f \
  "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml"
kube apply -f \
  "https://github.com/llm-d/llm-d-router/releases/download/${ROUTER_VERSION}/manifests.yaml"

push_file "$LAB_DIR/manifests/router-values.yaml" "$REMOTE_VALUES"
guest_helm upgrade --install "$RELEASE" "$CHART" \
  --namespace "$NAMESPACE" \
  --version "$CHART_VERSION" \
  --values "$REMOTE_VALUES" \
  --wait --timeout 5m

# EndpointPickerConfig는 startup 때만 읽으므로 ConfigMap 변경 뒤 명시적으로 재시작한다.
kube rollout restart deployment/${RELEASE}-epp -n "$NAMESPACE"
kube rollout status deployment/${RELEASE}-epp -n "$NAMESPACE" --timeout=300s

push_file "$LAB_DIR/manifests/inference-objectives.yaml" "$REMOTE_OBJECTIVES"
kube apply -n "$NAMESPACE" -f "$REMOTE_OBJECTIVES"
echo "router_service=${RELEASE}-epp"
kube get inferencepool "$RELEASE" -n "$NAMESPACE"
kube get inferenceobjective -n "$NAMESPACE"
