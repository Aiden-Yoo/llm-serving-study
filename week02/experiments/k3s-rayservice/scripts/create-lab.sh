#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-k3s}
POOL=${POOL:-week2-k3s}
NETWORK=${NETWORK:-week2-k3s-net}
PROFILE=${PROFILE:-week2-k3s}
K3S_VERSION=${K3S_VERSION:-v1.34.9+k3s1}
: "${LXD_STORAGE_SOURCE:?Set LXD_STORAGE_SOURCE to an empty directory on local SSD}"

command -v lxc >/dev/null

lxc storage show "$POOL" >/dev/null 2>&1 || \
  lxc storage create "$POOL" dir source="$LXD_STORAGE_SOURCE"
lxc network show "$NETWORK" >/dev/null 2>&1 || \
  lxc network create "$NETWORK" \
    ipv4.address=10.220.0.1/24 ipv4.nat=true ipv6.address=none dns.mode=managed
lxc profile show "$PROFILE" >/dev/null 2>&1 || lxc profile create "$PROFILE"

lxc profile set "$PROFILE" limits.cpu=8 limits.memory=24GiB \
  security.nesting=true \
  security.syscalls.intercept.mknod=true \
  security.syscalls.intercept.setxattr=true
lxc profile device show "$PROFILE" | grep -q '^eth0:' || \
  lxc profile device add "$PROFILE" eth0 nic network="$NETWORK" name=eth0

lxc info "$INSTANCE" >/dev/null 2>&1 || \
  lxc init ubuntu:24.04 "$INSTANCE" -s "$POOL" -p "$PROFILE"
lxc start "$INSTANCE" 2>/dev/null || true

for _ in $(seq 1 60); do
  lxc exec "$INSTANCE" -- systemctl is-system-running --wait >/dev/null 2>&1 && break
  sleep 2
done

lxc exec "$INSTANCE" -- bash -s -- "$K3S_VERSION" <<'INNER'
set -euo pipefail
K3S_VERSION=$1
mkdir -p /etc/rancher/k3s
cat > /etc/rancher/k3s/config.yaml <<'YAML'
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
  systemctl restart k3s
fi

for _ in $(seq 1 90); do
  kubectl get node 2>/dev/null | grep -Eq '[[:space:]]Ready[[:space:]]' && break
  sleep 2
done
kubectl get nodes -o wide
kubectl get pods -A -o wide
INNER
