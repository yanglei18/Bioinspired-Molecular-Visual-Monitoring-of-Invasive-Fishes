# Data Pipeline — Expert-Grounded CoT Annotation Construction

This directory builds expert-grounded Chain-of-Thought (CoT) training data for FG-VLM from fish
videos. A two-stage pipeline extracts and validates diagnostic attributes, generates logic-gated
reasoning, and converts the result into SFT / RL training JSON.

> Run all commands from the `evidence_grounded_reasoner/` module root so that the relative
> `data_pipeline/…` and `configs/…` paths resolve correctly.

## Pipeline Overview

```text
Videos  ──►  Stage 1 (VQA)  ──►  Stage 2 (Validate + CoT)  ──►  Convert  ──►  SFT / RL JSON
               │                        │                       │
        Extract appearance &       Validate attributes via    Format into
        behavior attributes        LLM checker, then generate  training schema
        from video with VLM        logic-gated CoT reasoning
```

## Setup

1. Copy the config template and fill in your API credentials:

```bash
cp configs/data_pipeline.yaml configs/my_pipeline.yaml
# Edit my_pipeline.yaml: set api_key, endpoint, model_name, video_dir, fish_code, fish_real_name
```

Or use environment variables (recommended for API keys):

```bash
export FGVLM_API_KEY="your-api-key"
export FGVLM_ENDPOINT="https://your-endpoint/v1"
export FGVLM_MODEL_NAME="your-model"
```

2. Install dependencies:

```bash
pip install opencv-python openai tqdm pyyaml
```

## Stage 1: VQA Attribute Extraction

Send each video + expert knowledge rules to a VLM to extract appearance and behavior attributes.

```bash
python data_pipeline/stage1_vqa.py \
  --config configs/my_pipeline.yaml \
  --fish-code crucian_carp \
  --fish-name "Crucian carp" \
  --video-dir /path/to/crucian_carp_videos \
  --output-dir stage1_vqa_results
```

Output: per-video JSON files in `stage1_vqa_results/<fish_code>/<video_id>.json`

## Stage 2: Attribute Validation + CoT Reasoning

Validate Stage 1 results against expert definitions, then generate logic-gated CoT reasoning.

```bash
python data_pipeline/stage2_cot.py \
  --config configs/my_pipeline.yaml \
  --fish-code crucian_carp \
  --fish-name "Crucian carp" \
  --vqa-dir stage1_vqa_results \
  --output-dir stage2_cot_results
```

Output: per-video JSON files in `stage2_cot_results/<fish_code>/<video_id>.json`

## Convert to Training Data

After Stage 2, convert the per-video CoT results into SFT and RL training JSON:

```bash
python data_pipeline/convert_to_training.py \
    --stage2-dir stage2_cot_results \
    --video-root /path/to/videos \
    --sft-output data/sft_train.json \
    --rl-output data/rl_train.json \
    --options "Common carp" "Crucian carp" "Grass carp"
```

- `--stage2-dir`: root directory of Stage 2 output (contains species subdirectories)
- `--video-root`: root directory of original video files (same subdirectory structure as stage2-dir)
- `--options`: candidate species for multiple-choice; omit for open-ended identification
- `--sft-output` / `--rl-output`: paths for the resulting SFT / RL training JSON

## Expert Knowledge Base

`expert_kd.py` defines appearance attributes, behavior attributes, negatives, and confusion rules
for 9 fish species:

- Common carp, Crucian carp, Mosquitofish, Guppy, Grass carp, Largemouth bass, Mozambique tilapia,
  Rainbow trout, Brown trout

To add a new species, append an entry to `expert_knowledge_data["species"]` in `expert_kd.py`.

## Training Data Format

SFT data is a JSON list in this format:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are an expert ichthyologist..."},
      {"role": "user", "content": "<video>\nDescribe what kind of fish is in this video.\nOptions:\n(A) black carp\n(B) common carp"},
      {"role": "assistant", "content": "<think>...</think><rethink>...</rethink><answer>(A) black carp</answer>"}
    ],
    "videos": ["/path/to/video.mp4"]
  }
]
```

RL data is a JSON list in this format:

```json
[
  {
    "messages": [{"role": "user", "content": "<video>\nDescribe what kind of fish is in this video."}],
    "videos": ["/path/to/video.mp4"],
    "solution": ["(A) brown trout"],
    "reasoning_content": "<think>evidence observations...</think><rethink>evidence-based exclusion...</rethink>"
  }
]
```
