#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
require_lab

kube delete inferenceobjective \
  premium-traffic standard-traffic best-effort-traffic \
  -n "$NAMESPACE" --ignore-not-found

if guest_helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
  guest_helm uninstall "$RELEASE" -n "$NAMESPACE"
fi

echo "Router resources removed; the vLLM and monitoring resources were kept."
