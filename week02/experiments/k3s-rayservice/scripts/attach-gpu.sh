#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-k3s}
GPU_PCI_ADDRESS=${GPU_PCI_ADDRESS:-}

command -v lxc >/dev/null
command -v jq >/dev/null
lxc info "$INSTANCE" >/dev/null

if [[ -z $GPU_PCI_ADDRESS ]]; then
  GPU_PCI_ADDRESS=$(lxc query /1.0/resources | \
    jq -r '.gpu.cards[] | select(.vendor_id == "1002") | .pci_address' | head -n 1)
fi
[[ -n $GPU_PCI_ADDRESS ]]

lxc config device show "$INSTANCE" | grep -q '^r9700:' || \
  lxc config device add "$INSTANCE" r9700 gpu \
    gputype=physical pci="$GPU_PCI_ADDRESS"
lxc config device show "$INSTANCE" | grep -q '^kfd:' || \
  lxc config device add "$INSTANCE" kfd unix-char \
    source=/dev/kfd path=/dev/kfd mode=0666

add_readonly_disk() {
  local name=$1 source=$2 target=$3
  [[ -n $source ]] || return 0
  lxc config device show "$INSTANCE" | grep -q "^${name}:" || \
    lxc config device add "$INSTANCE" "$name" disk \
      source="$source" path="$target" readonly=true
}

add_readonly_disk vllm-runtime "${VLLM_RUNTIME_SOURCE:-}" /lab-assets/runtime/vllm
add_readonly_disk qwen-small "${MODEL_SOURCE:-}" /lab-assets/models/Qwen3-0.6B
add_readonly_disk rocm-userspace "${ROCM_SOURCE:-}" /lab-assets/rocm

lxc exec "$INSTANCE" -- bash -lc \
  'ls -l /dev/kfd /dev/dri/card* /dev/dri/render*; grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties'
