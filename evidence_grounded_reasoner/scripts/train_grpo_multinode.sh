#!/usr/bin/env bash
set -euo pipefail

# Multi-node GRPO training script.
# In addition to the single-node env vars (train_grpo.sh), set:
#   NNODES         - number of nodes (default: 1)
#   NODE_RANK      - rank of this node (default: 0)
#   MASTER_ADDR    - master node address (default: 127.0.0.1)
#   MASTER_PORT    - master node port (default: 29500)

: "${MODEL_PATH:?Set MODEL_PATH to the SFT checkpoint or base VLM path}"
: "${DATASET_PATH:?Set DATASET_PATH to the RL training dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the directory for RL outputs}"
: "${JUDGE_API_BASE:?Set JUDGE_API_BASE to an OpenAI-compatible judge endpoint}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY to the judge API key}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to the judge model name}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-401408}"
export VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-4816896}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"

"${PYTHON_BIN}" -m swift.cli.main rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --external_plugins "${RELEASE_DIR}/rewards/rl_rewards.py" \
  --reward_funcs fact_checking_judge answer_accuracy \
  --reward_weights "${REWARD_WEIGHT_JUDGE:-0.5}" "${REWARD_WEIGHT_ANSWER:-1.0}" \
  --dataset "${DATASET_PATH}" \
  --use_vllm "${USE_VLLM:-true}" \
  --vllm_mode "${VLLM_MODE:-colocate}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.4}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-16384}" \
  --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB:-0}" \
  --vllm_use_async_engine "${VLLM_USE_ASYNC_ENGINE:-true}" \
  --sleep_level "${SLEEP_LEVEL:-1}" \
  --tuner_type "${TUNER_TYPE:-full}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-2048}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --save_steps "${SAVE_STEPS:-5}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}" \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --deepspeed "${DEEPSPEED:-zero2}" \
  --log_completions "${LOG_COMPLETIONS:-true}" \
  --report_to "${REPORT_TO:-tensorboard}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --epsilon "${EPSILON:-0.2}" \
  --epsilon_high "${EPSILON_HIGH:-0.28}" \
  --scale_rewards "${SCALE_REWARDS:-none}" \
  --system "${RELEASE_DIR}/configs/prompt_fish.txt" \
  --output_dir "${OUTPUT_DIR}"
