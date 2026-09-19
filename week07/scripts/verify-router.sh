#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

MODEL=${MODEL:-qwen3-4b}
EXPECTED_GLOBAL_QUEUE=${EXPECTED_GLOBAL_QUEUE:-32}
EXPECTED_STANDARD_QUEUE=${EXPECTED_STANDARD_QUEUE:-16}
service_ip=$(kube get service "${RELEASE}-epp" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
prometheus_ip=$(kube get service kube-prometheus-stack-prometheus \
  -n monitoring -o jsonpath='{.spec.clusterIP}')

epp_logs=$(kube logs deployment/${RELEASE}-epp -c epp -n "$NAMESPACE")
grep -F 'Initializing Flow Control layer' <<<"$epp_logs" >/dev/null
config_line=$(grep -F '"msg":"Raw config after phase one"' <<<"$epp_logs")
CONFIG_LINE="$config_line" \
  EXPECTED_GLOBAL_QUEUE="$EXPECTED_GLOBAL_QUEUE" \
  EXPECTED_STANDARD_QUEUE="$EXPECTED_STANDARD_QUEUE" \
  python3 - <<'PY'
import json
import os

config = json.loads(os.environ["CONFIG_LINE"])["config"]["flowControl"]
assert int(config["maxRequests"]) == int(os.environ["EXPECTED_GLOBAL_QUEUE"])
standard = next(band for band in config["priorityBands"] if band["priority"] == 0)
assert int(standard["maxRequests"]) == int(os.environ["EXPECTED_STANDARD_QUEUE"])
print("FLOW_CONTROL_CONFIG_OK")
PY

lxc exec "$INSTANCE" -- env \
  ROUTER_IP="$service_ip" \
  PROMETHEUS_IP="$prometheus_ip" \
  MODEL="$MODEL" \
  NAMESPACE="$NAMESPACE" \
  python3 - <<'PY'
import json
import os
import time
import urllib.parse
import urllib.request

router = os.environ["ROUTER_IP"]
model = os.environ["MODEL"]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: WEEK7_OK"}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": False,
}).encode()
request = urllib.request.Request(
    f"http://{router}/v1/chat/completions",
    data=body,
    headers={
        "Content-Type": "application/json",
        "x-llm-d-inference-objective": "premium-traffic",
        "x-llm-d-inference-fairness-id": "verification",
    },
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    payload = json.load(response)
assert response.status == 200
assert payload["choices"][0]["message"]["content"]

with urllib.request.urlopen(f"http://{router}:9090/metrics", timeout=30) as response:
    metrics = response.read().decode()
assert "llm_d_epp_flow_control_" in metrics

query = f'up{{job="week7-router-epp",namespace="{os.environ["NAMESPACE"]}"}}'
prometheus_url = (
    f'http://{os.environ["PROMETHEUS_IP"]}:9090/api/v1/query?'
    + urllib.parse.urlencode({"query": query})
)
deadline = time.monotonic() + 60
while True:
    with urllib.request.urlopen(prometheus_url, timeout=10) as response:
        result = json.load(response)["data"]["result"]
    if result and all(float(series["value"][1]) == 1 for series in result):
        break
    if time.monotonic() >= deadline:
        raise RuntimeError(f"Prometheus target did not become up: {result}")
    time.sleep(2)
print("ROUTER_REQUEST_OK")
print("FLOW_CONTROL_METRICS_OK")
print("PROMETHEUS_EPP_TARGET_UP")
PY

echo "WEEK7_ROUTER_VERIFIED"
