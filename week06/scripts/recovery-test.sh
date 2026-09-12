#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

INTERVAL_SECONDS=${INTERVAL_SECONDS:-1}
RESULT="$LAB_DIR/results/recovery-$(date -u +%Y%m%d-%H%M%S).tsv"
service_ip=$(kube get service vllm -n week6-llm -o jsonpath='{.spec.clusterIP}')
old_pod=$(kube get pod -n week6-llm -l app.kubernetes.io/name=vllm -o jsonpath='{.items[0].metadata.name}')

printf 'timestamp_utc\thttp_code\ttotal_seconds\n' >"$RESULT"
probe_loop() {
  while true; do
    timestamp=$(date -u +%FT%T.%3NZ)
    probe=$(lxc exec "$INSTANCE" -- curl -sS -o /dev/null \
      -w '%{http_code}\t%{time_total}' --max-time 5 "http://${service_ip}:8000/health" \
      2>/dev/null || true)
    [[ $probe == *$'\t'* ]] || probe=$'000\t5.000000'
    printf '%s\t%s\n' "$timestamp" "$probe" >>"$RESULT"
    sleep "$INTERVAL_SECONDS"
  done
}
probe_loop &
probe_pid=$!
trap 'kill "$probe_pid" 2>/dev/null || true' EXIT

sleep 5
deleted_at=$(date -u +%FT%T.%3NZ)
kube delete pod "$old_pod" -n week6-llm --wait=true
new_pod=
ready=
for _ in $(seq 1 240); do
  new_pod=$(kube get pod -n week6-llm -l app.kubernetes.io/name=vllm \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  ready=$(kube get pod "$new_pod" -n week6-llm \
    -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || true)
  [[ -n $new_pod && $new_pod != "$old_pod" && $ready == true ]] && break
  sleep 5
done
if [[ -z $new_pod || $new_pod == "$old_pod" || $ready != true ]]; then
  kube get pods -n week6-llm -o wide
  echo 'replacement vLLM Pod did not become Ready within 20 minutes' >&2
  exit 1
fi
ready_at=$(date -u +%FT%T.%3NZ)
sleep 5
kill "$probe_pid" 2>/dev/null || true
wait "$probe_pid" 2>/dev/null || true
trap - EXIT

printf '# old_pod=%s deleted_at=%s new_pod=%s ready_at=%s\n' \
  "$old_pod" "$deleted_at" "$new_pod" "$ready_at" >>"$RESULT"
echo "result=$RESULT"
