#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-kind}

if lxc info "$INSTANCE" >/dev/null 2>&1 && \
   [[ $(lxc list "$INSTANCE" --format csv -c s) == RUNNING ]]; then
  lxc exec "$INSTANCE" -- systemctl stop docker || true
  lxc stop "$INSTANCE" --timeout 30 || lxc stop "$INSTANCE" --force
fi
