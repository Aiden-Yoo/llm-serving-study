#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

MODEL=${MODEL:-qwen3-4b}
MIXED_RPS=${MIXED_RPS:-12}
MIXED_DURATION=${MIXED_DURATION:-20}
FAIRNESS_RPS=${FAIRNESS_RPS:-12}
FAIRNESS_DURATION=${FAIRNESS_DURATION:-45}
MAX_QUEUE_REQUESTS=${MAX_QUEUE_REQUESTS:-32}
REMOTE_SCRIPT=/tmp/week7-open-loop-load.py
REMOTE_RESULTS=/tmp/week7-challenge-results-$(date +%s)

mkdir -p "$LAB_DIR/results"
"$SCRIPT_DIR/verify-router.sh" | tee "$LAB_DIR/results/router-verification.log"
push_file "$SCRIPT_DIR/open_loop_load.py" "$REMOTE_SCRIPT"
lxc exec "$INSTANCE" -- mkdir -p "$REMOTE_RESULTS"

direct_ip=$(kube get service vllm -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
router_ip=$(kube get service "${RELEASE}-epp" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')

echo "[1/3] direct vLLM mixed overload"
lxc exec "$INSTANCE" -- python3 "$REMOTE_SCRIPT" \
  --endpoint "http://${direct_ip}:8000" \
  --endpoint-label direct-vllm \
  --model "$MODEL" \
  --scenario mixed \
  --rps "$MIXED_RPS" \
  --duration "$MIXED_DURATION" \
  --output-prefix "$REMOTE_RESULTS/direct-mixed"

echo "[2/3] llm-d priority and bounded-queue challenge"
lxc exec "$INSTANCE" -- python3 "$REMOTE_SCRIPT" \
  --endpoint "http://${router_ip}" \
  --endpoint-label llm-d-flow-control \
  --metrics-endpoint "http://${router_ip}:9090/metrics" \
  --model "$MODEL" \
  --scenario mixed \
  --rps "$MIXED_RPS" \
  --duration "$MIXED_DURATION" \
  --output-prefix "$REMOTE_RESULTS/flow-mixed"

echo "[3/3] llm-d noisy-neighbor fairness challenge"
lxc exec "$INSTANCE" -- python3 "$REMOTE_SCRIPT" \
  --endpoint "http://${router_ip}" \
  --endpoint-label llm-d-flow-control \
  --metrics-endpoint "http://${router_ip}:9090/metrics" \
  --model "$MODEL" \
  --scenario fairness \
  --rps "$FAIRNESS_RPS" \
  --duration "$FAIRNESS_DURATION" \
  --output-prefix "$REMOTE_RESULTS/flow-fairness"

while IFS= read -r remote_result; do
  lxc file pull "$INSTANCE$remote_result" "$LAB_DIR/results/"
done < <(lxc exec "$INSTANCE" -- find "$REMOTE_RESULTS" -maxdepth 1 -type f -print | sort)

python3 "$SCRIPT_DIR/verify-results.py" \
  --csv "$LAB_DIR/results/direct-mixed.csv" \
  --summary "$LAB_DIR/results/direct-mixed.json"
python3 "$SCRIPT_DIR/verify-results.py" \
  --csv "$LAB_DIR/results/flow-mixed.csv" \
  --summary "$LAB_DIR/results/flow-mixed.json"
python3 "$SCRIPT_DIR/verify-results.py" \
  --csv "$LAB_DIR/results/flow-fairness.csv" \
  --summary "$LAB_DIR/results/flow-fairness.json"

python3 "$SCRIPT_DIR/evaluate-challenge.py" \
  --direct "$LAB_DIR/results/direct-mixed.json" \
  --flow "$LAB_DIR/results/flow-mixed.json" \
  --flow-metrics "$LAB_DIR/results/flow-mixed-metrics.json" \
  --fairness "$LAB_DIR/results/flow-fairness.json" \
  --fairness-metrics "$LAB_DIR/results/flow-fairness-metrics.json" \
  --max-queue-requests "$MAX_QUEUE_REQUESTS" \
  --output "$LAB_DIR/results/challenge-verdict.json"

"$SCRIPT_DIR/capture-environment.sh" >/dev/null
(cd "$LAB_DIR/results" && sha256sum *.csv *.json *.log >SHA256SUMS)
echo "WEEK7_CHALLENGE_COMPLETE"
