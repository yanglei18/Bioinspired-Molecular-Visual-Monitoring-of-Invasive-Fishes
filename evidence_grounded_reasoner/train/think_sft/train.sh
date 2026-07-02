#!/bin/bash
# Think SFT — fine-tune Qwen3-VL-4B-Thinking to classify fish species with an
# explicit <think>/<rethink>/<answer> reasoning format.
#
# Default: trains from the base model for 200 steps (the recommended checkpoint
# is checkpoint-200). Override MAX_STEPS / MODEL_PATH / DATASET_PATH as needed.
#
# Usage:
#   bash train.sh                       # base model, 200 steps
#   MAX_STEPS=1000 bash train.sh        # train longer
#
# Requires: pip install ms-swift "transformers>=4.57" qwen_vl_utils
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Thinking}"
DATASET_PATH="${DATASET_PATH:-$RELEASE_DIR/data/train_think.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/ckpts}"

SYSTEM_PROMPT="You are an expert ichthyologist. Analyze the provided video and identify the fish species from the options in the user prompt. You must present your full, step-by-step reasoning process using exactly the following sections:
- <think>: Initial appearance and behavior observations with confidence scores.
- <rethink>: Detailed analysis including supporting evidence, exclusion of alternatives, and uncertainty reasoning.
- <answer>: The final answer in the exact option format from the user prompt, such as '(A) black carp'."

mkdir -p "$OUTPUT_DIR"

# run_sft_compat.py patches a torch 2.5.x FSDPModule stub before delegating to
# `swift sft` (no-op on torch >= 2.6). torchrun launches one worker per GPU.
torchrun --nproc_per_node "${NPROC_PER_NODE:-8}" \
    "$SCRIPT_DIR/run_sft_compat.py" \
        --model "$MODEL_PATH" \
        --model_type qwen3_vl \
        --tuner_type full \
        --dataset "$DATASET_PATH" \
        --torch_dtype bfloat16 \
        --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
        --max_steps "${MAX_STEPS:-200}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
        --per_device_eval_batch_size 1 \
        --learning_rate "${LEARNING_RATE:-1e-5}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
        --eval_steps -1 \
        --save_steps "${SAVE_STEPS:-100}" \
        --save_total_limit 10 \
        --logging_steps 5 \
        --max_length "${MAX_LENGTH:-16384}" \
        --output_dir "$OUTPUT_DIR" \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --freeze_vit true \
        --freeze_aligner false \
        --freeze_llm false \
        --response_prefix $'<think>\n' \
        --attn_impl flash_attn \
        --deepspeed zero2 \
        --system "$SYSTEM_PROMPT"
