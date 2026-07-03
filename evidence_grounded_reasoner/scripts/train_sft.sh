#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the base VLM path}"
: "${DATASET_PATH:?Set DATASET_PATH to the SFT training dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the directory for SFT outputs}"

LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RUN_TS="$(date '+%m%d_%H%M%S')"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Start time: $(date '+%F %T')"
PYTHON_BIN="${PYTHON_BIN:-python}"
which "${PYTHON_BIN}"
"${PYTHON_BIN}" --version

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

"${PYTHON_BIN}" -m swift.cli.main sft \
  --model "${MODEL_PATH}" \
  --train_type "${TRAIN_TYPE:-full}" \
  --dataset "${DATASET_PATH}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --eval_steps "${EVAL_STEPS:--1}" \
  --save_steps "${SAVE_STEPS:-50}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --max_length "${MAX_LENGTH:-16384}" \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --freeze_vit "${FREEZE_VIT:-true}" \
  --freeze_aligner "${FREEZE_ALIGNER:-true}" \
  --freeze_llm "${FREEZE_LLM:-false}" \
  --attn_impl "${ATTN_IMPL:-flash_attn}" \
  --deepspeed "${DEEPSPEED:-zero2}" \
  --report_to "${REPORT_TO:-tensorboard}"
