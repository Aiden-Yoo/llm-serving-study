#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

if kube get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
  apply_manifest "$LAB_DIR/manifests/vllm-servicemonitor.yaml"
  kube get servicemonitor vllm -n week6-llm
else
  echo 'ServiceMonitor CRD not installed; validating the Prometheus endpoint directly.'
fi
kube delete pod metrics-curl -n week6-llm --ignore-not-found --wait=true >/dev/null
kube run metrics-curl -n week6-llm --restart=Never --image=curlimages/curl:8.16.0 \
  --rm -i --command -- sh -c \
  'curl -fsS http://vllm:8000/metrics | grep -m 1 "^vllm:"'
