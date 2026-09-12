#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

echo '--- host AMD GPU ---'
if command -v amd-smi >/dev/null 2>&1; then amd-smi static; else lspci -nn | grep -i 'AMD.*VGA\|AMD.*Display'; fi
[[ -e /dev/kfd ]] || { echo '/dev/kfd is missing on host' >&2; exit 1; }

echo '--- LXD devices and assets ---'
lxc exec "$INSTANCE" -- bash -lc '
  set -e
  ls -l /dev/kfd /dev/dri/render*
  test -d /lab-assets/rocm
  test -x /lab-assets/runtime/vllm/bin/python
  test -d /lab-assets/models/Qwen3
  test -d /lab-assets/cache/week6-vllm
'

echo '--- k3s ---'
kube get nodes -o wide
kube get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tamd.com/gpu="}{.status.allocatable.amd\.com/gpu}{"\n"}{end}'
