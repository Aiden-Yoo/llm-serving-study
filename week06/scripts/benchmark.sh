#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

MODEL=${MODEL:-qwen3-4b}
CONCURRENCIES=${CONCURRENCIES:-1,2,4,8,16}
REQUESTS_PER_LEVEL=${REQUESTS_PER_LEVEL:-20}
MAX_TOKENS=${MAX_TOKENS:-128}
REMOTE_SCRIPT=/tmp/week6-benchmark.py
REMOTE_RESULTS=/tmp/week6-benchmark-results

lxc exec "$INSTANCE" -- rm -f "$REMOTE_SCRIPT"
lxc file push "$SCRIPT_DIR/benchmark.py" "$INSTANCE$REMOTE_SCRIPT"
lxc exec "$INSTANCE" -- rm -rf "$REMOTE_RESULTS"
service_ip=$(kube get service vllm -n week6-llm -o jsonpath='{.spec.clusterIP}')
lxc exec "$INSTANCE" -- python3 "$REMOTE_SCRIPT" \
  --endpoint "http://${service_ip}:8000" \
  --model "$MODEL" \
  --concurrencies "$CONCURRENCIES" \
  --requests-per-level "$REQUESTS_PER_LEVEL" \
  --max-tokens "$MAX_TOKENS" \
  --output-dir "$REMOTE_RESULTS"
while IFS= read -r remote_result; do
  lxc file pull "$INSTANCE$remote_result" "$LAB_DIR/results/"
done < <(lxc exec "$INSTANCE" -- find "$REMOTE_RESULTS" -maxdepth 1 -type f -print)
