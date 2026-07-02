#!/bin/bash
# Evaluate a checkpoint on the validation set (macro accuracy, 10-class balanced),
# using the think (<think>/<rethink>/<answer>) inference prompt.
#
# Usage:
#   CHECKPOINT_PATH=/path/to/checkpoint bash eval.sh
#
# Optional env vars:
#   BENCH_FILE   benchmark json (default: ../data/val.json)
#   OUTPUT_DIR   where to write outputs (default: ./eval_outputs/<ckpt name>)
#   NUM_GPUS / GPU_IDS   GPU sharding (default 8 / 0-7)
#   EVAL_SCRIPT  override evaluator path (default: bundled eval_parallel_mc_local.py)
#
# Requires: pip install ms-swift "transformers>=4.57" qwen_vl_utils
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    echo "Usage: CHECKPOINT_PATH=/path/to/checkpoint bash eval.sh" >&2
    exit 1
fi

BENCH_FILE="${BENCH_FILE:-$RELEASE_DIR/data/val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/eval_outputs/$(basename "$CHECKPOINT_PATH")}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export QWENVL_BBOX_FORMAT="${QWENVL_BBOX_FORMAT:-new}"

SYSTEM_PROMPT="You are an expert ichthyologist. Analyze the provided video and identify the fish species from the options in the user prompt. You must present your full, step-by-step reasoning process using exactly the following sections:
- <think>: Initial appearance and behavior observations with confidence scores.
- <rethink>: Detailed analysis including supporting evidence, exclusion of alternatives, and uncertainty reasoning.
- <answer>: The final answer in the exact option format from the user prompt, such as '(A) black carp'."

mkdir -p "$OUTPUT_DIR"
EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRIPT_DIR/eval_parallel_mc_local.py}"

python3 "$EVAL_SCRIPT" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --benchmark-file "$BENCH_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --system-prompt "$SYSTEM_PROMPT" \
    --num-gpus "$NUM_GPUS" \
    --gpu-ids "$GPU_IDS"

echo ""
echo "Results (macro accuracy) saved to: $OUTPUT_DIR"
