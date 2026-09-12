#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

manifest="$LAB_DIR/manifests/gpu-quota-probe.yaml"
remote="/tmp/week6-$(basename "$manifest")"
lxc exec "$INSTANCE" -- rm -f "$remote"
lxc file push "$manifest" "$INSTANCE$remote"
kube delete pod second-gpu-probe -n week6-llm --ignore-not-found --wait=true >/dev/null

set +e
output=$(kube apply -f "$remote" 2>&1)
status=$?
set -e
printf '%s\n' "$output"

if [[ $status -eq 0 ]]; then
  kube delete pod second-gpu-probe -n week6-llm --ignore-not-found --wait=true >/dev/null
  echo 'expected the second GPU request to be rejected by ResourceQuota' >&2
  exit 1
fi
grep -q 'exceeded quota: single-amd-gpu' <<<"$output" || {
  echo 'the request failed, but not because of the expected GPU quota' >&2
  exit 1
}
echo GPU_QUOTA_GUARD_OK
