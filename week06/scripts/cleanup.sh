#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

# Device Plugin은 다른 GPU workload가 사용할 수 있으므로 기본적으로 유지한다.
kube delete namespace week6-llm --ignore-not-found --wait=true
if [[ ${REMOVE_DEVICE_PLUGIN:-false} == true ]]; then
  kube delete daemonset amdgpu-device-plugin-daemonset -n kube-system --ignore-not-found
fi
if [[ ${REMOVE_OBSERVABILITY:-false} == true ]]; then
  if lxc exec "$INSTANCE" -- helm status kube-prometheus-stack -n monitoring >/dev/null 2>&1; then
    lxc exec "$INSTANCE" -- helm uninstall kube-prometheus-stack -n monitoring
  fi
  if lxc exec "$INSTANCE" -- helm status amd-device-metrics -n kube-amd-gpu >/dev/null 2>&1; then
    lxc exec "$INSTANCE" -- helm uninstall amd-device-metrics -n kube-amd-gpu
  fi
fi
if [[ ${STOP_INSTANCE:-false} == true ]]; then
  if lxc exec "$INSTANCE" -- systemctl is-active --quiet k3s; then
    lxc exec "$INSTANCE" -- systemctl stop k3s
  fi
  lxc stop "$INSTANCE" --timeout 30 || lxc stop "$INSTANCE" --force
fi
