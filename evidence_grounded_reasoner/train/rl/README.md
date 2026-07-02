# RL (GRPO)

Reinforcement learning (GRPO) from a Think-SFT checkpoint, optimizing a
final-answer accuracy reward while keeping the `<think>/<rethink>/<answer>`
format. `reward_plugin.py` implements the reward
(`fish_final_answer_accuracy`): +weight if the extracted `<answer>` option
matches the ground truth, with per-species weights emphasizing harder classes.

## Data

`../../data/train_rl.json` (5,223 RL prompts).

## Run

```bash
# Default: from a Think-SFT checkpoint, 200 steps (recommended = checkpoint-200)
POLICY_CKPT=/path/to/think_sft/ckpts/<run>/checkpoint-200 bash train.sh
```

Common overrides:

| Var | Default | Meaning |
|-----|---------|---------|
| `POLICY_CKPT` | (required) | Think-SFT checkpoint to start from |
| `MAX_STEPS` | `200` | training steps |
| `SAVE_STEPS` | `100` | checkpoint interval |
| `NUM_GPUS` | `8` | GPUs |
| `OUTPUT_DIR` | `./ckpts` | output directory |
| `NUM_GENERATIONS` | `16` | rollouts per prompt |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `2` | prompts per GPU per step |
| `MAX_COMPLETION_LENGTH` | `1536` | max generated tokens |

The **recommended checkpoint is `checkpoint-200`**.

### Throughput note

GRPO rollout generation (HF path) is the bottleneck. The defaults use
`num_generations=16` with `per_device_train_batch_size=2` (generation batch = 32):
more rollouts per prompt pack the decode batch and keep all GPUs near 100%
utilization, roughly halving wall-clock vs. a `bs=4, num_generations=8` setup.

`run_rlhf_compat.py` wraps `swift rlhf` and patches torch/trl compatibility on
torch < 2.6 (no-op otherwise).
