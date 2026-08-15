#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-k3s}

if lxc info "$INSTANCE" >/dev/null 2>&1 && \
   [[ $(lxc list "$INSTANCE" --format csv -c s) == RUNNING ]]; then
  lxc exec "$INSTANCE" -- systemctl stop k3s || true
  lxc stop "$INSTANCE" --timeout 30 || lxc stop "$INSTANCE" --force
fi
