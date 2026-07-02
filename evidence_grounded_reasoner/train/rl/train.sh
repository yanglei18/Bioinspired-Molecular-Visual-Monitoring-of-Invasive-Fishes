#!/bin/bash
# RL (GRPO) — further optimize the Think-SFT model with a final-answer accuracy
# reward, keeping the <think>/<rethink>/<answer> format.
#
# Default: trains from the Think-SFT checkpoint for 200 steps (recommended
# checkpoint is checkpoint-200). POLICY_CKPT must point to a Think-SFT checkpoint.
#
# Usage:
#   POLICY_CKPT=/path/to/think_sft/ckpts/<run>/checkpoint-200 bash train.sh
#
# Optional env vars: NUM_GPUS, MAX_STEPS, SAVE_STEPS.
# Requires: pip install ms-swift "transformers>=4.57" qwen_vl_utils
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${POLICY_CKPT:-}" ]]; then
    echo "POLICY_CKPT must be set to a Think-SFT checkpoint (e.g. think_sft/ckpts/<run>/checkpoint-200)." >&2
    exit 1
fi

RL_DATA_PATH="${RL_DATA_PATH:-$RELEASE_DIR/data/train_rl.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/ckpts}"
NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-8}}"
SWIFT_RLHF_ENTRY="$SCRIPT_DIR/run_rlhf_compat.py"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export QWENVL_BBOX_FORMAT="${QWENVL_BBOX_FORMAT:-new}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

SYSTEM_PROMPT="You are an expert ichthyologist. Analyze the provided video and identify the fish species from the options in the user prompt. You must present your full, step-by-step reasoning process using exactly the following sections:
- <think>: Initial appearance and behavior observations with confidence scores.
- <rethink>: Detailed analysis including supporting evidence, exclusion of alternatives, and uncertainty reasoning.
- <answer>: The final answer in the exact option format from the user prompt, such as '(A) black carp'."

# Throughput-tuned GRPO defaults: more generations per prompt packs the decode
# batch for higher GPU utilisation (see README).
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GENERATIONS))}"

mkdir -p "$OUTPUT_DIR"
MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); p=s.getsockname()[1]; s.close(); print(p)')}"

python3 -m torch.distributed.run \
        --nproc_per_node "$NUM_GPUS" \
        --standalone \
        --master_port "$MASTER_PORT" \
        "$SWIFT_RLHF_ENTRY" \
        --rlhf_type grpo \
        --model "$POLICY_CKPT" \
        --model_type qwen3_vl \
        --tuner_type full \
        --dataset "$RL_DATA_PATH" \
        --external_plugins "$SCRIPT_DIR/reward_plugin.py" \
        --reward_funcs fish_final_answer_accuracy \
        --torch_dtype bfloat16 \
        --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
        --num_iterations "${NUM_ITERATIONS:-1}" \
        --max_steps "${MAX_STEPS:-200}" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size 1 \
        --learning_rate "${LEARNING_RATE:-4e-7}" \
        --num_generations "$NUM_GENERATIONS" \
        --generation_batch_size "$GENERATION_BATCH_SIZE" \
        --logging_steps 1 \
        --save_steps "${SAVE_STEPS:-100}" \
        --save_total_limit 10 \
        --max_length 12288 \
        --max_completion_length "${MAX_COMPLETION_LENGTH:-1536}" \
        --response_prefix $'<think>\n' \
        --temperature 1.0 \
        --top_p 0.95 \
        --beta "${BETA:-0.03}" \
        --warmup_ratio 0.03 \
        --dataloader_num_workers 4 \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --attn_impl flash_attn \
        --truncation_strategy delete \
        --use_vllm false \
        --save_strategy steps \
        --eval_strategy no \
        --log_completions true \
        --log_entropy true \
        --system "$SYSTEM_PROMPT" \
        --output_dir "$OUTPUT_DIR" \
        --deepspeed zero2
