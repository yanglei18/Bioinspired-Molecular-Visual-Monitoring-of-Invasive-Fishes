#!/bin/bash
# Fast vLLM eval of one checkpoint on a benchmark, sharded across given GPUs.
# Optional/faster alternative to eval.sh; requires a vLLM build that supports
# Qwen3-VL (e.g. vLLM >= 0.17). The canonical released numbers use eval.sh
# (PtEngine), not this script.
#
# Usage: run_vllm_eval.sh <ckpt> <benchmark.json> <run_dir> [gpu_ids_csv]
# Optional env: PYTHON (interpreter, default python3), HF_HOME.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/vllm_eval_mc.py"
CKPT="$1"; BENCH="$2"; RUNDIR="$3"; GPUS="${4:-0,1,2,3}"
mkdir -p "$RUNDIR"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
pids=()
for i in "${!GPU_ARR[@]}"; do
  g=${GPU_ARR[$i]}
  CUDA_VISIBLE_DEVICES=$g $PYTHON $SCRIPT \
    --mode worker --checkpoint-path "$CKPT" --benchmark-file "$BENCH" \
    --run-dir "$RUNDIR" --shard-id "$i" --num-shards "$N" \
    --max-tokens 2048 --max-model-len 16384 --gpu-mem-util 0.85 --enforce-eager \
    > "$RUNDIR/shard_${i}_gpu${g}.log" 2>&1 &
  pids+=($!)
done
echo "launched $N shards (pids ${pids[*]}) on GPUs $GPUS"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail -ne 0 ]; then echo "WARNING: a shard failed; see logs in $RUNDIR"; fi
$PYTHON $SCRIPT --mode reduce --checkpoint-path "$CKPT" --benchmark-file "$BENCH" --run-dir "$RUNDIR"
