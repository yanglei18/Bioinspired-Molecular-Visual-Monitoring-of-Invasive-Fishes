#!/usr/bin/env bash
set -euo pipefail

# Deploy a vision-language model with SWIFT + vLLM backend and wait until healthy.
# Usage: bash vllm_server.sh <checkpoint_path> [--gpus GPU_IDS] [--host HOST] [--port PORT] [--run-dir RUN_DIR]

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CKPT_PATH=${1:?usage: $0 <checkpoint_path> [--gpus GPU_IDS] [--host HOST] [--port PORT] [--run-dir RUN_DIR]}
shift || true

GPUS=0,1,2,3,4,5,6,7
HOST=127.0.0.1
PORT=8000
RUN_DIR=${SCRIPT_DIR}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)   GPUS=$2;    shift 2 ;;
    --host)   HOST=$2;    shift 2 ;;
    --port)   PORT=$2;    shift 2 ;;
    --run-dir) RUN_DIR=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

PID_FILE=${RUN_DIR}/deploy.pid
LOG_FILE=${RUN_DIR}/deploy.log
STATUS_FILE=${RUN_DIR}/server.status.json
STARTUP_TIMEOUT_SEC=${STARTUP_TIMEOUT_SEC:-1800}
TP_SIZE=$(python -c "print(len([g for g in '${GPUS}'.split(',') if g]))")

write_status() {
  local state=$1 message=${2:-} pid=${3:-}
  python - <<'PY' "${STATUS_FILE}" "${CKPT_PATH}" "${HOST}" "${PORT}" "${state}" "${message}" "${pid}"
import json, os, sys
from datetime import datetime
path, ckpt_path, host, port, state, message, pid = sys.argv[1:8]
existing = {}
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        existing = json.load(f)
existing.update({
    'ckpt_path': ckpt_path, 'host': host, 'port': int(port),
    'state': state, 'message': message,
    'pid': int(pid) if pid else None,
    'updated_at': datetime.now().isoformat(timespec='seconds'),
})
with open(path, 'w', encoding='utf-8') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
PY
}

port_ready() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY \
    curl -fsS "http://${HOST}:${PORT}/health" >/dev/null
}

if port_ready; then
  write_status failed "port ${PORT} is already serving" ""
  echo "ERROR: port ${PORT} is already serving; aborting" >&2
  exit 1
fi

write_status starting "launching deploy" ""

CUDA_VISIBLE_DEVICES=${GPUS} \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
  --model "${CKPT_PATH}" \
  --infer_backend vllm \
  --host "${HOST}" \
  --port "${PORT}" \
  --vllm_tensor_parallel_size "${TP_SIZE}" \
  --vllm_data_parallel_size 1 \
  > "${LOG_FILE}" 2>&1 &
DEPLOY_PID=$!
printf '%s\n' "${DEPLOY_PID}" > "${PID_FILE}"
write_status starting "waiting for health check" "${DEPLOY_PID}"

start_ts=$(date +%s)
while true; do
  if port_ready; then
    write_status ready "deploy server is ready" "${DEPLOY_PID}"
    echo "deploy server is ready (pid=${DEPLOY_PID})"
    break
  fi

  if ! kill -0 "${DEPLOY_PID}" 2>/dev/null; then
    write_status failed "deploy exited before becoming ready" "${DEPLOY_PID}"
    tail -n 40 "${LOG_FILE}" >&2 || true
    exit 1
  fi

  now_ts=$(date +%s)
  if (( now_ts - start_ts >= STARTUP_TIMEOUT_SEC )); then
    write_status failed "startup timed out after ${STARTUP_TIMEOUT_SEC}s" "${DEPLOY_PID}"
    tail -n 40 "${LOG_FILE}" >&2 || true
    exit 1
  fi

  sleep 5
done
