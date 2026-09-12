#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

WINDOW=${WINDOW:-5m}
[[ $WINDOW =~ ^[1-9][0-9]*[smhd]$ ]] || {
  echo 'WINDOW must use Prometheus duration syntax such as 5m or 1h' >&2
  exit 1
}
prometheus_ip=$(kube get service kube-prometheus-stack-prometheus \
  -n monitoring -o jsonpath='{.spec.clusterIP}')
result="$LAB_DIR/results/prometheus-summary.json"

lxc exec "$INSTANCE" -- python3 - "$prometheus_ip" "$WINDOW" <<'PY' >"$result"
import json
import sys
import urllib.parse
import urllib.request

base = f"http://{sys.argv[1]}:9090/api/v1/query"
window = sys.argv[2]
queries = {
    "max_gpu_gfx_activity_percent": f"max(max_over_time(amd_gpu_gfx_activity[{window}]))",
    "max_gpu_used_vram_mb": f"max(max_over_time(amd_gpu_used_vram[{window}]))",
    "max_gpu_power_watts": f"max(max_over_time(amd_gpu_average_package_power[{window}]))",
    "max_vllm_running_requests": f"max(max_over_time(vllm:num_requests_running[{window}]))",
    "max_vllm_waiting_requests": f"max(max_over_time(vllm:num_requests_waiting[{window}]))",
    "max_vllm_kv_cache_usage_ratio": f"max(max_over_time(vllm:kv_cache_usage_perc[{window}]))",
}
values = {}
for name, query in queries.items():
    url = base + "?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    series = data["data"]["result"]
    values[name] = float(series[0]["value"][1]) if series else None
print(json.dumps({"window": window, "queries": queries, "values": values}, indent=2))
PY
cat "$result"
