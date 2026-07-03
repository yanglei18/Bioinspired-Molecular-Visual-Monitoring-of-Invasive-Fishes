# Evidence-Grounded Fine-Grained Confirmation

This module (`evidence_grounded_reasoner/`) implements the **evidence-grounded fine-grained
confirmation** stage of the paper *"Bioinspired molecular–visual monitoring of invasive fishes"* — an
evidence-grounded vision–language reasoning model that confirms fish species from a short video,
producing an explicit `<think>` / `<rethink>` / `<answer>` reasoning trace. *(Referred to in code and
scripts by the short handle **FG-VLM**.)*

- **Base model:** `Qwen/Qwen3-VL-4B-Instruct`
- **Training:** supervised fine-tuning (SFT) on expert-grounded CoT data, then GRPO reinforcement
  learning with a fact-checking LLM-judge reward
- **Species (9):** common carp, crucian carp, mosquitofish, guppy, grass carp, largemouth bass,
  mozambique tilapia, rainbow trout, brown trout

## Repository Structure

```text
evidence_grounded_reasoner/
├── configs/                 # prompts & data pipeline config
├── data_pipeline/           # CoT data construction pipeline (see data_pipeline/README.md)
│   ├── expert_kd.py         # expert knowledge base (9 species)
│   ├── llm_check_prompt.py  # LLM attribute validation prompts
│   ├── llm_reason_prompt.py # CoT reasoning generation prompts
│   ├── stage1_vqa.py        # Stage 1: VQA attribute extraction
│   ├── stage2_cot.py        # Stage 2: validation + CoT reasoning
│   └── convert_to_training.py  # convert Stage 2 results → SFT/RL JSON
├── rewards/                 # GRPO reward plugin (judge-based evidence grounding + answer accuracy)
├── scripts/                 # training & utility scripts
│   ├── train_sft.sh         # single-node SFT training
│   ├── train_grpo.sh        # single-node GRPO training
│   ├── train_grpo_multinode.sh  # multi-node GRPO training
│   └── prepare_smoke_data.py    # create small RL smoke-test dataset
├── eval/                    # benchmark generation, vLLM serving, HTTP inference, scoring
│   ├── generate_benchmark.py   # generate benchmark JSON from a video directory
│   ├── vllm_server.sh          # deploy model with SWIFT + vLLM backend
│   ├── eval_parallel_http.py   # parallel HTTP inference against the vLLM server
│   ├── run_eval_pipeline.sh    # end-to-end: deploy → infer → validate
│   └── compute_accuracy.py     # compute MC accuracy (micro/macro + per-species)
├── examples/                # data schema example
├── environment.yml
├── requirements.txt
└── README.md
```

## Environment Setup

SFT and GRPO use the same environment.

```bash
conda env create -f environment.yml
conda activate FG-VLM
pip install "ms-swift==4.0.2"
```

If conda CUDA packages are unavailable:

```bash
conda create -n FG-VLM python=3.11 pip -y
conda activate FG-VLM
pip install -r requirements.txt
pip install "ms-swift==4.0.2"
```

## Prepare Model and Data

### Download the Base Model

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir ./checkpoints/Qwen3-VL-4B-Instruct
```

### Prepare Training Data

Either download the pre-processed SFT/RL data, or build it yourself from videos with the two-stage
CoT pipeline.

- **Option A — download pre-processed data:**

  ```bash
  # TODO: upload data and update the link below
  wget https://xxxx/FG-VLM-training-data.tar.gz
  tar -xzf FG-VLM-training-data.tar.gz -C ./data/
  ```

  This provides `data/sft_train.json` and `data/rl_train.json` directly.

- **Option B — build it yourself:** run the VQA → validate → CoT → convert pipeline and the expert
  knowledge base described in [`data_pipeline/README.md`](data_pipeline/README.md). The expected
  SFT / RL JSON schema is documented there as well.

Optionally create a small RL smoke-test dataset for a quick training run:

```bash
python scripts/prepare_smoke_data.py \
  --source /path/to/rl_train.json \
  --output ./data/rl_smoke.json \
  --num-samples 8
```

## Training

### Stage 1: SFT

```bash
export MODEL_PATH=$(pwd)/checkpoints/Qwen3-VL-4B-Instruct
export DATASET_PATH=/path/to/sft_train.json
export OUTPUT_DIR=$(pwd)/outputs/sft

bash scripts/train_sft.sh
```

Use the resulting SFT checkpoint as the initialization checkpoint for GRPO.

### Stage 2: GRPO

```bash
export MODEL_PATH=/path/to/sft_checkpoint
export DATASET_PATH=$(pwd)/data/rl_smoke.json
export OUTPUT_DIR=$(pwd)/outputs/grpo

export JUDGE_API_BASE=https://your-openai-compatible-endpoint/v1
export JUDGE_API_KEY=your-api-key
export JUDGE_MODEL=your-judge-model

bash scripts/train_grpo.sh
```

Training logs and checkpoints are written to `${OUTPUT_DIR}`.

## Evaluation

Evaluate a trained checkpoint on a fish identification benchmark.

### Step 1: Prepare a benchmark file

You can either download the pre-built benchmark or generate it from your own video directory.

- **Option A — download the pre-built benchmark:**

  ```bash
  # TODO: upload benchmark and update the link below
  wget https://xxxx/FG-VLM-benchmark.json
  ```

- **Option B — generate from your own videos:**

  ```bash
  python eval/generate_benchmark.py \
    --video-dir /path/to/test_videos \
    --output benchmark.json \
    --options "Common carp" "Crucian carp" "Grass carp" \
              "Mosquitofish" "Guppy" "Largemouth bass" \
              "Mozambique tilapia" "Rainbow trout" "Brown trout"
  ```

  Each benchmark entry has `video_path` and `question` fields. Add `--answer "species name"` to
  include a `ground_truth` field for accuracy scoring.

### Step 2: Run inference

Either the full pipeline (deploy + infer + cleanup) or manual step-by-step.

- **Full pipeline (recommended):**

  ```bash
  bash eval/run_eval_pipeline.sh /path/to/checkpoint \
    --benchmark-file benchmark.json \
    --gpus 0,1,2,3,4,5,6,7 \
    --port 8000 \
    --run-dir ./eval_output
  ```

  This deploys the model via SWIFT + vLLM, runs parallel HTTP inference, validates results, and
  cleans up the server on exit. Output: `eval_output/res.json`.

- **Step-by-step:**

  ```bash
  # Terminal 1: start the model server
  bash eval/vllm_server.sh /path/to/checkpoint --gpus 0,1,2,3,4,5,6,7 --port 8000

  # Terminal 2: run inference
  python eval/eval_parallel_http.py \
    --benchmark-file benchmark.json \
    --ckpt-path /path/to/checkpoint \
    --host 127.0.0.1 --port 8000 \
    --num-workers 8 \
    --output-file res.json
  ```

### Step 3: Compute accuracy

If the benchmark includes ground truth:

```bash
python eval/compute_accuracy.py \
  --result-file res.json \
  --benchmark-file benchmark.json \
  --output accuracy.json
```

This computes micro/macro accuracy and a per-species breakdown. A prediction is correct if the
option letter or species name matches the ground truth.
