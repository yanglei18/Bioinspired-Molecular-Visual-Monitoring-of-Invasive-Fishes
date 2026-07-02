# Evaluation

Evaluate any checkpoint on the validation set (`../data/val.json`,
1,099 samples) and report **macro accuracy** (class-averaged over the 10 species).
The think (`<think>/<rethink>/<answer>`) inference prompt is used.

## Quick start

```bash
CHECKPOINT_PATH=/path/to/checkpoint bash eval.sh
```

For the released model:

```bash
CHECKPOINT_PATH=FishVL-4B-Thinking bash eval.sh   # -> 98.10% macro
```

## Options (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `CHECKPOINT_PATH` | (required) | model checkpoint to evaluate |
| `BENCH_FILE` | `../data/val.json` | benchmark json |
| `OUTPUT_DIR` | `./eval_outputs/<ckpt name>` | where results are written |
| `NUM_GPUS` / `GPU_IDS` | `8` / `0,1,2,3,4,5,6,7` | GPU sharding |

Results are written to `OUTPUT_DIR/` including `eval_results.json` with the
overall `macro_accuracy`, `sample_accuracy`, and per-species breakdown.

## Files

- `eval.sh` — entry point (PtEngine evaluator, think prompt, macro accuracy).
- `eval_parallel_mc_local.py` — the evaluator (multi-GPU, transformers/swift).
- `vllm_eval_mc.py` + `run_vllm_eval.sh` — optional faster evaluator; requires a
  vLLM build that supports Qwen3-VL (e.g. vLLM ≥ 0.17). Usage:
  `bash run_vllm_eval.sh <ckpt> ../data/val.json <out_dir> 0,1,2,3`.

## Benchmark format

Each row of `val.json` is `{video_path, question, answer}` where
`answer` is the gold option, e.g. `(A) black carp`. Point `video_path` at your
local copy of the videos. The set has no overlap with the training data.

## Requirements

```bash
pip install ms-swift "transformers>=4.57" qwen_vl_utils
```
