#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUNTIME_ROOT="${VLLM_RUNTIME_ROOT:-${PROJECT_ROOT}/.runtime/vllm}"
VENV="${VLLM_VENV:-${RUNTIME_ROOT}/venvs/vllm-rocm723}"
MODEL_PATH="${VLLM_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3-4b-vllm}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"

if [[ ! -x "${VENV}/bin/vllm" ]]; then
  echo "vLLM executable not found: ${VENV}/bin/vllm" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

mkdir -p \
  "${RUNTIME_ROOT}/cache/huggingface" \
  "${RUNTIME_ROOT}/cache/vllm" \
  "${RUNTIME_ROOT}/cache/torchinductor" \
  "${RUNTIME_ROOT}/cache/triton" \
  "${RUNTIME_ROOT}/cache/torch-kernels" \
  "${RUNTIME_ROOT}/logs" \
  "${RUNTIME_ROOT}/tmp"

export HF_HOME="${RUNTIME_ROOT}/cache/huggingface"
export VLLM_CACHE_ROOT="${RUNTIME_ROOT}/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="${RUNTIME_ROOT}/cache/torch-kernels"
export TMPDIR="${RUNTIME_ROOT}/tmp"

echo "Starting ${SERVED_MODEL_NAME} from ${MODEL_PATH} on ${HOST}:${PORT}" >&2
exec "${VENV}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
