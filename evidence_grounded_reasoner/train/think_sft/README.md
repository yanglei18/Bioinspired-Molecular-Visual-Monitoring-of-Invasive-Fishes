# Think SFT

Supervised fine-tuning of `Qwen/Qwen3-VL-4B-Thinking` to classify the fish
species while emitting an explicit `<think>/<rethink>/<answer>` reasoning trace.

## Data

`../../data/train_think.json` (3,519 rows, ms-swift `messages`+`videos` format).
The assistant target is the full reasoning trace ending in
`<answer>(X) species</answer>`.

## Run

```bash
# Default: from the base model, 200 steps (recommended checkpoint = checkpoint-200)
bash train.sh
```

Common overrides (environment variables):

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL_PATH` | `Qwen/Qwen3-VL-4B-Thinking` | starting model |
| `MAX_STEPS` | `200` | training steps |
| `SAVE_STEPS` | `100` | checkpoint interval (→ checkpoint-100, -200, …) |
| `NPROC_PER_NODE` | `8` | GPUs |
| `OUTPUT_DIR` | `./ckpts` | output directory |
| `PER_DEVICE_TRAIN_BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | `1` / `4` | effective batch = GPUs×bs×accum |

Output checkpoints land in `ckpts/<run>/checkpoint-*`. The **recommended
checkpoint is `checkpoint-200`**.

`run_sft_compat.py` is a thin wrapper around `swift sft` that patches a missing
`torch.distributed.fsdp.FSDPModule` symbol on torch < 2.6 (no-op otherwise).

## Next stage

Use a Think-SFT checkpoint as the policy for RL:

```bash
POLICY_CKPT=$(pwd)/ckpts/<run>/checkpoint-200 bash ../rl/train.sh
```
