#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEEK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${VLLM_RUNTIME_ROOT:-${WEEK_DIR}/.runtime/vllm}"
VENV="${VLLM_VENV:-${RUNTIME_ROOT}/venvs/vllm-rocm723}"
MODEL_PATH="${VLLM_MODEL_PATH:-${WEEK_DIR}/models/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3-4b-vllm}"
RESULTS_DIR="${SCRIPT_DIR}/results"
VLLM_PORT="${VLLM_PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-8001}"

if [[ ! -x "${VENV}/bin/vllm" ]]; then
  echo "vLLM executable not found: ${VENV}/bin/vllm" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

mkdir -p \
  "${RESULTS_DIR}" \
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

vllm_pid=""
gateway_pid=""

stop_process_group() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "${pid}" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_process_group "${gateway_pid}"
  stop_process_group "${vllm_pid}"
  gateway_pid=""
  vllm_pid=""
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local pid="$2"
  local name="$3"
  for _ in $(seq 1 600); do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before becoming ready" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name}: ${url}" >&2
  return 1
}

run_profile() {
  local profile="$1"
  local max_num_seqs="$2"
  local max_num_batched_tokens="$3"
  local vllm_log="${RUNTIME_ROOT}/logs/week2-vllm-${profile}.log"
  local gateway_log="${RUNTIME_ROOT}/logs/week2-gateway-${profile}.log"

  echo "Starting vLLM profile=${profile} max_num_seqs=${max_num_seqs} max_num_batched_tokens=${max_num_batched_tokens}"
  setsid "${VENV}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --enable-prefix-caching \
    >"${vllm_log}" 2>&1 &
  vllm_pid=$!
  wait_for_url "http://127.0.0.1:${VLLM_PORT}/v1/models" "${vllm_pid}" "vLLM"

  echo "Starting gateway profile=${profile}"
  VLLM_UPSTREAM_URL="http://127.0.0.1:${VLLM_PORT}" \
    setsid "${VENV}/bin/python" -m uvicorn gateway:app \
      --app-dir "${SCRIPT_DIR}" \
      --host 127.0.0.1 \
      --port "${GATEWAY_PORT}" \
      --workers 1 \
      >"${gateway_log}" 2>&1 &
  gateway_pid=$!
  wait_for_url "http://127.0.0.1:${GATEWAY_PORT}/health" "${gateway_pid}" "gateway"

  "${VENV}/bin/python" "${SCRIPT_DIR}/benchmark_serving_modes.py" \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}" \
    --gateway-url "http://127.0.0.1:${GATEWAY_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --profile "${profile}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --output-json "${RESULTS_DIR}/serving-${profile}.json" \
    --output-markdown "${RESULTS_DIR}/serving-${profile}.md"

  stop_process_group "${gateway_pid}"
  gateway_pid=""
  stop_process_group "${vllm_pid}"
  vllm_pid=""
  sleep 5
}

run_profile constrained 4 2048
run_profile balanced 16 8192

"${VENV}/bin/python" "${SCRIPT_DIR}/summarize_serving_modes.py" \
  --constrained "${RESULTS_DIR}/serving-constrained.json" \
  --balanced "${RESULTS_DIR}/serving-balanced.json" \
  --output-json "${RESULTS_DIR}/serving-modes-latest.json" \
  --output-markdown "${RESULTS_DIR}/serving-modes-latest.md"
