#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-kind}
POOL=${POOL:-week2-kind}
NETWORK=${NETWORK:-week2-kind-net}
PROFILE=${PROFILE:-week2-kind}
CLUSTER=${CLUSTER:-week2}
KIND_VERSION=${KIND_VERSION:-v0.32.0}
KIND_NODE_IMAGE=${KIND_NODE_IMAGE:-kindest/node:v1.36.1@sha256:3489c7674813ba5d8b1a9977baea8a6e553784dab7b84759d1014dbd78f7ebd5}
KUBECTL_VERSION=${KUBECTL_VERSION:-v1.36.1}
ATTACH_GPU=${ATTACH_GPU:-1}
: "${LXD_STORAGE_SOURCE:?Set LXD_STORAGE_SOURCE to an empty directory on local SSD}"

command -v lxc >/dev/null
command -v jq >/dev/null

lxc storage show "$POOL" >/dev/null 2>&1 || \
  lxc storage create "$POOL" dir source="$LXD_STORAGE_SOURCE"
lxc network show "$NETWORK" >/dev/null 2>&1 || \
  lxc network create "$NETWORK" \
    ipv4.address=10.221.0.1/24 ipv4.nat=true ipv6.address=none dns.mode=managed
lxc profile show "$PROFILE" >/dev/null 2>&1 || lxc profile create "$PROFILE"

lxc profile set "$PROFILE" limits.cpu=8 limits.memory=24GiB \
  security.nesting=true \
  security.syscalls.intercept.mknod=true \
  security.syscalls.intercept.setxattr=true
lxc profile device show "$PROFILE" | grep -q '^eth0:' || \
  lxc profile device add "$PROFILE" eth0 nic network="$NETWORK" name=eth0

lxc info "$INSTANCE" >/dev/null 2>&1 || \
  lxc init ubuntu:24.04 "$INSTANCE" -s "$POOL" -p "$PROFILE"

if [[ $ATTACH_GPU == 1 ]]; then
  GPU_PCI_ADDRESS=${GPU_PCI_ADDRESS:-$(
    lxc query /1.0/resources | \
      jq -r '.gpu.cards[] | select(.vendor_id == "1002") | .pci_address' | head -n 1
  )}
  [[ -n $GPU_PCI_ADDRESS ]]
  lxc config device show "$INSTANCE" | grep -q '^r9700:' || \
    lxc config device add "$INSTANCE" r9700 gpu \
      gputype=physical pci="$GPU_PCI_ADDRESS"
  lxc config device show "$INSTANCE" | grep -q '^kfd:' || \
    lxc config device add "$INSTANCE" kfd unix-char \
      source=/dev/kfd path=/dev/kfd mode=0666
fi

lxc start "$INSTANCE" 2>/dev/null || \
  [[ $(lxc list "$INSTANCE" --format csv -c s) == RUNNING ]]
for _ in $(seq 1 60); do
  state=$(lxc exec "$INSTANCE" -- systemctl is-system-running 2>/dev/null || true)
  [[ $state == running || $state == degraded ]] && break
  sleep 2
done
[[ ${state:-} == running || ${state:-} == degraded ]]

lxc exec "$INSTANCE" -- bash -s -- \
  "$KIND_VERSION" "$KUBECTL_VERSION" "$CLUSTER" "$KIND_NODE_IMAGE" <<'INNER'
set -euo pipefail
KIND_VERSION=$1
KUBECTL_VERSION=$2
CLUSTER=$3
KIND_NODE_IMAGE=$4
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
systemctl enable --now docker
docker info >/dev/null

arch=$(dpkg --print-architecture)
case "$arch" in
  amd64) kind_arch=amd64 ;;
  arm64) kind_arch=arm64 ;;
  *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
esac
curl -fsSLo /usr/local/bin/kind \
  "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${kind_arch}"
curl -fsSLo /tmp/kind.sha256sum \
  "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${kind_arch}.sha256sum"
echo "$(awk '{print $1}' /tmp/kind.sha256sum)  /usr/local/bin/kind" | \
  sha256sum --check
chmod +x /usr/local/bin/kind

curl -fsSLo /usr/local/bin/kubectl \
  "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${kind_arch}/kubectl"
curl -fsSLo /tmp/kubectl.sha256 \
  "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${kind_arch}/kubectl.sha256"
echo "$(cat /tmp/kubectl.sha256)  /usr/local/bin/kubectl" | sha256sum --check
chmod +x /usr/local/bin/kubectl

cat > /root/kind-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${CLUSTER}
featureGates:
  KubeletInUserNamespace: true
nodes:
  - role: control-plane
EOF

# Pass the GPU device nodes through Docker into the Kind node.
mounts=()
[[ -e /dev/kfd ]] && mounts+=("/dev/kfd|/dev/kfd|false")
[[ -d /dev/dri ]] && mounts+=("/dev/dri|/dev/dri|false")
if ((${#mounts[@]})); then
  echo '    extraMounts:' >> /root/kind-config.yaml
  for entry in "${mounts[@]}"; do
    IFS='|' read -r source target readonly <<<"$entry"
    cat >> /root/kind-config.yaml <<EOF
      - hostPath: ${source}
        containerPath: ${target}
        readOnly: ${readonly}
EOF
  done
fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster \
    --name "$CLUSTER" \
    --image "$KIND_NODE_IMAGE" \
    --config /root/kind-config.yaml \
    --wait 5m
fi

kind version
docker version --format 'Docker {{.Server.Version}}'
kubectl version
kubectl get nodes -o wide
kubectl get pods -A -o wide
echo '--- devices visible in the Kind node ---'
docker exec "${CLUSTER}-control-plane" \
  sh -c 'ls -l /dev/kfd /dev/dri/render* 2>/dev/null || true'
INNER
