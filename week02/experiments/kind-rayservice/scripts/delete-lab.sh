#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-kind}
POOL=${POOL:-week2-kind}
NETWORK=${NETWORK:-week2-kind-net}
PROFILE=${PROFILE:-week2-kind}

lxc delete -f "$INSTANCE" 2>/dev/null || true
lxc profile delete "$PROFILE" 2>/dev/null || true
lxc network delete "$NETWORK" 2>/dev/null || true
lxc storage delete "$POOL" 2>/dev/null || true
