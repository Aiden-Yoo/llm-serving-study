#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
INSTANCE=${INSTANCE:-llm-week6-k3s}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

require_lab() {
  require_command lxc
  lxc info "$INSTANCE" >/dev/null 2>&1 || {
    echo "LXD instance not found: $INSTANCE (run setup-lab.sh first)" >&2
    exit 1
  }
}

kube() {
  lxc exec "$INSTANCE" -- env KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl "$@"
}

apply_manifest() {
  local file=$1 remote="/tmp/week6-$(basename "$1")"
  lxc exec "$INSTANCE" -- rm -f "$remote"
  lxc file push "$file" "$INSTANCE$remote"
  kube apply -f "$remote"
}
