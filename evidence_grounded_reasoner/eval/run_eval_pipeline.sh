#!/usr/bin/env bash
set -euo pipefail

# End-to-end evaluation pipeline: deploy model -> run inference -> validate results -> cleanup.
# Usage: bash run_eval_pipeline.sh <checkpoint_path> [options] [-- <eval args>]
#
# Options:
#   --gpus GPU_IDS       Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#   --host HOST          Server host (default: 127.0.0.1)
#   --port PORT          Server port (default: 8000)
#   --run-dir RUN_DIR    Working directory for logs/outputs (default: script dir)
#   --benchmark-file FILE  Benchmark JSON file (required, forwarded to eval script)
#
# All other args before -- are forwarded to eval_parallel_http.py.
# Environment variables:
#   EVAL_TIMEOUT_SEC          Max evaluation time in seconds (default: 21600)
#   NO_PROGRESS_TIMEOUT_SEC   Stall timeout in seconds (default: 1800)
#   HEARTBEAT_INTERVAL_SEC    Heartbeat write interval (default: 60)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CKPT_PATH=${1:?usage: $0 <checkpoint_path> [options]}
shift || true

GPUS=0,1,2,3,4,5,6,7
HOST=127.0.0.1
PORT=8000
RUN_DIR=${SCRIPT_DIR}
BENCHMARK_FILE=''
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)           GPUS=$2;          shift 2 ;;
    --host)           HOST=$2;          shift 2 ;;
    --port)           PORT=$2;          shift 2 ;;
    --run-dir)        RUN_DIR=$2;       shift 2 ;;
    --benchmark-file) BENCHMARK_FILE=$2; shift 2 ;;
    *)                EVAL_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${BENCHMARK_FILE}" ]]; then
  echo "ERROR: --benchmark-file is required" >&2
  exit 1
fi

PID_FILE=${RUN_DIR}/deploy.pid
RUN_META_FILE=${RUN_DIR}/run.meta.json
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-21600}
NO_PROGRESS_TIMEOUT_SEC=${NO_PROGRESS_TIMEOUT_SEC:-1800}
HEARTBEAT_INTERVAL_SEC=${HEARTBEAT_INTERVAL_SEC:-60}
REQUEST_CONNECT_TIMEOUT_SEC=${REQUEST_CONNECT_TIMEOUT_SEC:-30}
REQUEST_READ_TIMEOUT_SEC=${REQUEST_READ_TIMEOUT_SEC:-600}
REQUEST_MAX_RETRIES=${REQUEST_MAX_RETRIES:-3}
REQUEST_RETRY_BACKOFF_SEC=${REQUEST_RETRY_BACKOFF_SEC:-2.0}
HEARTBEAT_FILE=${RUN_DIR}/heartbeat.json

write_run_meta() {
  local status=$1 message=${2:-}
  python - <<'PY' "${RUN_META_FILE}" "${CKPT_PATH}" "${HOST}" "${PORT}" "${status}" "${message}" "${HEARTBEAT_FILE}"
import json, os, sys
from datetime import datetime
path, ckpt_path, host, port, status, message, heartbeat_file = sys.argv[1:8]
existing = {}
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        existing = json.load(f)
existing.update({
    'ckpt_path': ckpt_path, 'host': host, 'port': int(port),
    'status': status, 'message': message,
    'updated_at': datetime.now().isoformat(timespec='seconds'),
    'heartbeat_file': heartbeat_file,
})
with open(path, 'w', encoding='utf-8') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
PY
}

validate_result() {
  python - <<'PY' "${RUN_DIR}/res.json"
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    payload = json.load(f)
summary = payload['summary']
results = payload['results']
total = int(summary['total_samples'])
if total <= 0:
    raise SystemExit('total_samples must be > 0')
if len(results) != total:
    raise SystemExit(f'results length {len(results)} != total_samples {total}')
PY
}

cleanup() {
  if [[ -f "${PID_FILE}" ]]; then
    DEPLOY_PID=$(<"${PID_FILE}")
    if [[ -n "${DEPLOY_PID}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
      PGID=$(ps -o pgid= -p "${DEPLOY_PID}" 2>/dev/null | tr -d ' ')
      if [[ -n "${PGID}" ]]; then
        kill -TERM -- -"${PGID}" 2>/dev/null || true
        sleep 2
        kill -KILL -- -"${PGID}" 2>/dev/null || true
      fi
      kill -TERM "${DEPLOY_PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "${RUN_DIR}"
write_run_meta running

bash "${SCRIPT_DIR}/vllm_server.sh" "${CKPT_PATH}" --gpus "${GPUS}" --host "${HOST}" --port "${PORT}" --run-dir "${RUN_DIR}"
write_run_meta evaluating

if ! timeout --signal=TERM --kill-after=30s "${EVAL_TIMEOUT_SEC}" \
  python "${SCRIPT_DIR}/eval_parallel_http.py" \
    --benchmark-file "${BENCHMARK_FILE}" \
    --ckpt-path "${CKPT_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --output-file "${RUN_DIR}/res.json" \
    --heartbeat-file "${HEARTBEAT_FILE}" \
    --heartbeat-interval "${HEARTBEAT_INTERVAL_SEC}" \
    --no-progress-timeout "${NO_PROGRESS_TIMEOUT_SEC}" \
    --request-connect-timeout "${REQUEST_CONNECT_TIMEOUT_SEC}" \
    --request-read-timeout "${REQUEST_READ_TIMEOUT_SEC}" \
    --max-retries "${REQUEST_MAX_RETRIES}" \
    --retry-backoff "${REQUEST_RETRY_BACKOFF_SEC}" \
    "${EVAL_ARGS[@]}"; then
  write_run_meta failed "evaluation command failed"
  exit 1
fi

validate_result
write_run_meta completed
