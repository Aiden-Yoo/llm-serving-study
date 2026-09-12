#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

kube get crd servicemonitors.monitoring.coreos.com >/dev/null

vllm_ip=$(kube get service vllm -n week6-llm -o jsonpath='{.spec.clusterIP}')
amd_ip=$(kube get service amd-device-metrics-amd-metrics-exporter-svc \
  -n kube-amd-gpu -o jsonpath='{.spec.clusterIP}')
prometheus_ip=$(kube get service kube-prometheus-stack-prometheus \
  -n monitoring -o jsonpath='{.spec.clusterIP}')

vllm_metrics=$(mktemp)
amd_metrics=$(mktemp)
prometheus_targets=$(mktemp)
trap 'rm -f "$vllm_metrics" "$amd_metrics" "$prometheus_targets"' EXIT
lxc exec "$INSTANCE" -- curl -fsS "http://${vllm_ip}:8000/metrics" >"$vllm_metrics"
lxc exec "$INSTANCE" -- curl -fsS "http://${amd_ip}:5000/metrics" >"$amd_metrics"
grep -m 5 '^vllm:' "$vllm_metrics" | tee "$LAB_DIR/results/vllm-metrics-sample.txt"
grep -m 5 '^amd_gpu_' "$amd_metrics" | tee "$LAB_DIR/results/amd-metrics-sample.txt"

for _ in $(seq 1 30); do
  if lxc exec "$INSTANCE" -- curl -fsS \
    "http://${prometheus_ip}:9090/api/v1/targets" >"$prometheus_targets"; then
    if python3 - "$prometheus_targets" "$LAB_DIR/results/observability-targets.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
targets = data["data"]["activeTargets"]
selected = {}
for target in targets:
    service = target.get("discoveredLabels", {}).get("__meta_kubernetes_service_name")
    if service in {"vllm", "amd-device-metrics-amd-metrics-exporter-svc"}:
        selected[service] = {
            "health": target.get("health"),
            "last_error": target.get("lastError"),
            "scrape_url": target.get("scrapeUrl"),
        }
services = {service: target["health"] for service, target in selected.items()}
assert services.get("vllm") == "up", services
assert services.get("amd-device-metrics-amd-metrics-exporter-svc") == "up", services
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(selected, output, indent=2)
print("OBSERVABILITY_TARGETS_OK")
PY
    then
      exit 0
    fi
  fi
  sleep 10
done

echo 'Prometheus did not report both vLLM and AMD targets as up' >&2
exit 1
