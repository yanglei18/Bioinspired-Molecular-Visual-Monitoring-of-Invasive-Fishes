# Fine-Grained Fish Species Classification (Qwen3-VL-4B)

Identify one of 10 freshwater fish species from a short video. The model reasons
in an explicit `<think>/<rethink>/<answer>` format. The pipeline has two stages:

1. **Think SFT** — supervised fine-tuning of `Qwen/Qwen3-VL-4B-Thinking` on
   reasoning-annotated data.
2. **RL (GRPO)** — reinforcement learning from the Think-SFT model with a
   final-answer accuracy reward.

## Released model

We release a single model — **FishVL-4B-Thinking** (in `FishVL-4B-Thinking/`).
It is evaluated on the validation set (`data/val.json`). See [MODEL.md](MODEL.md).

## Species

| Option | Species | Option | Species |
|--------|---------|--------|---------|
| (A) | black carp | (F) | common carp |
| (B) | chinese labeo | (G) | chinese paddlefish |
| (C) | chinese sucker | (H) | mud carp |
| (D) | redeye barbel | (I) | schizothorax fish |
| (E) | serrated barb | (J) | wuchang bream |

## Repository layout

```
.
├── README.md                  # this file
├── MODEL.md                   # the released model
├── LICENSE
├── FishVL-4B-Thinking/        # the released model weights
├── data/
│   ├── train_think.json       # Think SFT data (3,519)
│   ├── train_rl.json          # RL prompt data (5,223)
│   ├── val.json      # validation set (1,099)
│   └── videos/                # all referenced videos (9,841 clips)
├── train/
│   ├── think_sft/             # Think SFT — train.sh + README
│   └── rl/                    # RL (GRPO) — train.sh + reward_plugin + README
└── eval/                      # evaluation script + README
```

> Run all scripts from the repository root — the dataset JSONs reference videos
> by repo-relative paths (`data/videos/<file>.mp4`).

## Environment

```bash
pip install ms-swift "transformers>=4.57" qwen_vl_utils
```

- Base model: `Qwen/Qwen3-VL-4B-Thinking`.
- Scripts assume 8-GPU training with DeepSpeed ZeRO-2; set `NPROC_PER_NODE`,
  `NUM_GPUS`, `GPU_IDS` as needed.
- The `run_*_compat.py` shims patch a missing `torch.distributed.fsdp.FSDPModule`
  on torch < 2.6 and are no-ops otherwise.

## Data

| File | Rows | Description |
|------|------|-------------|
| `data/train_think.json` | 3,519 | Think SFT — `<think>/<rethink>/<answer>` targets |
| `data/train_rl.json` | 5,223 | RL (GRPO) prompts |
| `data/val.json` | 1,099 | Validation set (no overlap with training data) |
| `data/videos/` | 9,841 | all video clips referenced by the JSONs (~5.7 GB) |

The videos are bundled under `data/videos/`. Training rows are in ms-swift format
(`messages` + `videos`); validation rows are `{video_path, question, answer}`.
All paths are repo-relative (`data/videos/<file>.mp4`), so run scripts from the
repository root. The validation set has zero video overlap with the training
splits.

## How to train

Two stages, run in order. Defaults reproduce the recommended checkpoints
(**Think SFT step 200**, **RL step 200**).

```bash
# 1) Think SFT (from the base model) -> recommended: checkpoint-200
bash train/think_sft/train.sh

# 2) RL (GRPO) from the Think-SFT checkpoint -> recommended: checkpoint-200
POLICY_CKPT=train/think_sft/ckpts/<run>/checkpoint-200 bash train/rl/train.sh
```

See `train/think_sft/README.md` and `train/rl/README.md` for details and options.

## How to evaluate

```bash
CHECKPOINT_PATH=/path/to/checkpoint bash eval/eval.sh
```

Evaluates on `data/val.json` (macro accuracy). See `eval/README.md`.
