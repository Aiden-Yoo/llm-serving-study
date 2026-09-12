#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

POOL=${POOL:-week6-k3s}
NETWORK=${NETWORK:-week6-k3s-net}
PROFILE=${PROFILE:-week6-k3s}
K3S_VERSION=${K3S_VERSION:-v1.34.9+k3s1}
GPU_PCI_ADDRESS=${GPU_PCI_ADDRESS:-}
: "${LXD_STORAGE_SOURCE:?Set LXD_STORAGE_SOURCE to an empty directory on local SSD}"
: "${ROCM_SOURCE:?Set ROCM_SOURCE to the host ROCm directory}"
: "${VLLM_RUNTIME_SOURCE:?Set VLLM_RUNTIME_SOURCE to the host vLLM virtualenv}"
: "${MODEL_SOURCE:?Set MODEL_SOURCE to the host model directory}"

require_command lxc
require_command jq

lxc storage show "$POOL" >/dev/null 2>&1 || lxc storage create "$POOL" dir source="$LXD_STORAGE_SOURCE"
lxc network show "$NETWORK" >/dev/null 2>&1 || lxc network create "$NETWORK" \
  ipv4.address=10.221.0.1/24 ipv4.nat=true ipv6.address=none dns.mode=managed
lxc profile show "$PROFILE" >/dev/null 2>&1 || lxc profile create "$PROFILE"
lxc profile set "$PROFILE" limits.cpu=12 limits.memory=32GiB \
  security.nesting=true \
  security.syscalls.intercept.mknod=true \
  security.syscalls.intercept.setxattr=true
lxc profile device show "$PROFILE" | grep -q '^eth0:' || \
  lxc profile device add "$PROFILE" eth0 nic network="$NETWORK" name=eth0

lxc info "$INSTANCE" >/dev/null 2>&1 || lxc init ubuntu:24.04 "$INSTANCE" -s "$POOL" -p "$PROFILE"
if [[ $(lxc list "$INSTANCE" --format csv -c s) != RUNNING ]]; then
  lxc start "$INSTANCE"
fi
for _ in $(seq 1 60); do
  lxc exec "$INSTANCE" -- systemctl is-system-running --wait >/dev/null 2>&1 && break
  sleep 2
done

if [[ -z $GPU_PCI_ADDRESS ]]; then
  GPU_PCI_ADDRESS=$(lxc query /1.0/resources | jq -r \
    '.gpu.cards[] | select(.vendor_id == "1002") | .pci_address' | head -n 1)
fi
[[ -n $GPU_PCI_ADDRESS ]] || { echo "AMD GPU not found" >&2; exit 1; }

lxc config device show "$INSTANCE" | grep -q '^amd-gpu:' || \
  lxc config device add "$INSTANCE" amd-gpu gpu gputype=physical pci="$GPU_PCI_ADDRESS"
lxc config device show "$INSTANCE" | grep -q '^kfd:' || \
  lxc config device add "$INSTANCE" kfd unix-char source=/dev/kfd path=/dev/kfd mode=0666

add_readonly_disk() {
  local name=$1 source=$2 target=$3
  lxc config device show "$INSTANCE" | grep -q "^${name}:" || \
    lxc config device add "$INSTANCE" "$name" disk source="$source" path="$target" readonly=true
}
add_readonly_disk rocm-userspace "$ROCM_SOURCE" /lab-assets/rocm
add_readonly_disk vllm-runtime "$VLLM_RUNTIME_SOURCE" /lab-assets/runtime/vllm
add_readonly_disk model "$MODEL_SOURCE" /lab-assets/models/Qwen3

lxc exec "$INSTANCE" -- mkdir -p /lab-assets/cache/week6-vllm
lxc exec "$INSTANCE" -- bash -s -- "$K3S_VERSION" <<'INNER'
set -euo pipefail
K3S_VERSION=$1
mkdir -p /etc/rancher/k3s
cat >/etc/rancher/k3s/config.yaml <<'YAML'
disable:
  - traefik
  - servicelb
write-kubeconfig-mode: "0644"
kubelet-arg:
  - feature-gates=KubeletInUserNamespace=true
YAML
if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -
else
  installed_version=$(k3s --version | awk 'NR == 1 {print $3}')
  if [[ $installed_version != "$K3S_VERSION" ]]; then
    echo "existing k3s version $installed_version does not match requested $K3S_VERSION" >&2
    exit 1
  fi
  systemctl restart k3s
fi
for _ in $(seq 1 90); do
  kubectl get node 2>/dev/null | grep -Eq '[[:space:]]Ready[[:space:]]' && exit 0
  sleep 2
done
echo "k3s node did not become Ready" >&2
exit 1
INNER

"$SCRIPT_DIR/preflight.sh"
