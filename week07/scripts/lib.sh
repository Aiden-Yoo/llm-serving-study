#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
INSTANCE=${INSTANCE:-llm-serving-k3s}
NAMESPACE=${NAMESPACE:-llm-serving}
RELEASE=${RELEASE:-week7-router}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

require_lab() {
  require_command lxc
  lxc info "$INSTANCE" >/dev/null 2>&1 || {
    echo "LXD instance not found: $INSTANCE" >&2
    exit 1
  }
}

kube() {
  lxc exec "$INSTANCE" -- env KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl "$@"
}

guest_helm() {
  lxc exec "$INSTANCE" -- env KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm "$@"
}

push_file() {
  local source=$1 destination=$2
  lxc exec "$INSTANCE" -- rm -f "$destination"
  lxc file push "$source" "$INSTANCE$destination"
}
