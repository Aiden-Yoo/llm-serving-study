#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEEK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${VLLM_RUNTIME_ROOT:-${WEEK_DIR}/.runtime/vllm}"
VENV="${VLLM_VENV:-${RUNTIME_ROOT}/venvs/vllm-rocm723}"
SMALL_MODEL_PATH="${SMALL_MODEL_PATH:-${WEEK_DIR}/models/Qwen3-0.6B}"
BASE_MODEL_PATH="${VLLM_MODEL_PATH:-${WEEK_DIR}/models/Qwen3-4B-Instruct-2507}"
RESULTS_DIR="${SCRIPT_DIR}/results"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Python executable not found: ${VENV}/bin/python" >&2
  exit 1
fi
for model in "${SMALL_MODEL_PATH}" "${BASE_MODEL_PATH}"; do
  if [[ ! -f "${model}/config.json" ]]; then
    echo "Model config not found: ${model}/config.json" >&2
    exit 1
  fi
done

mkdir -p "${RESULTS_DIR}" "${RUNTIME_ROOT}/tmp"
export TMPDIR="${RUNTIME_ROOT}/tmp"

"${VENV}/bin/python" "${SCRIPT_DIR}/multi_model_lru.py" \
  --model "small=./models/Qwen3-0.6B=${SMALL_MODEL_PATH}" \
  --model "base=./models/Qwen3-4B-Instruct-2507=${BASE_MODEL_PATH}" \
  --sequence small small base small \
  --capacity 1 \
  --max-new-tokens 16 \
  --output-json "${RESULTS_DIR}/multi-model-lru-latest.json" \
  --output-markdown "${RESULTS_DIR}/multi-model-lru-latest.md"
