# Bioinspired molecular–visual monitoring of invasive fishes — Visual Perception Pipeline

![Paper: under review](https://img.shields.io/badge/Paper-under%20review-b31b1b.svg)

**Paper:** *Bioinspired molecular–visual monitoring of invasive fishes* — Lei Li<sup>†</sup>,
Yanyu Li<sup>†</sup>, Lei Yang<sup>†</sup>, *et al.*, Chen Lv<sup>\*</sup>, Dekui He<sup>\*</sup>,
Junzhi Yu<sup>\*</sup> (under review).

<sup>†</sup> Equal contribution (co-first authors). &nbsp; <sup>\*</sup> Co-corresponding authors.

Vision code for the paper above. It turns raw underwater video into auditable, species-level
invasive-fish confirmations through three modules.

> **Scope of this repository.** This repository releases the **visual perception pipeline** only:
> (1) `object_centric_extractor` — object-centric sequence generation (detection → segmentation →
> tracking → instance export); (2) `open_world_filter` — open-world candidate filtering; and
> (3) `evidence_grounded_reasoner` — evidence-grounded fine-grained confirmation (a reasoning VLM,
> SFT + GRPO on Qwen3-VL). The **molecular / eDNA acquisition and bioinformatics pipeline** is
> **not included**. Large assets (the reasoner's training videos and model weights) are hosted
> externally — see the module Tutorials for download links.

## Modules

Each module Tutorial is the authoritative source for that module's commands, configs, and outputs.

| Module | Role |
|--------|------|
| [`object_centric_extractor`](object_centric_extractor/README.md) | Object-centric sequence generation: Grounded-SAM2 detection → segmentation → mask tracking → instance crop/video export, plus detection/tracking evaluation. |
| [`open_world_filter`](open_world_filter/README.md) | Open-world candidate filtering: momentum-based dual-branch representation training + reference-set matching to retain in-scope invasive candidates and reject unknowns. |
| [`evidence_grounded_reasoner`](evidence_grounded_reasoner/README.md) | Evidence-grounded fine-grained confirmation: a reasoning VLM (Qwen3-VL, Think-SFT + GRPO) producing `<think>/<rethink>/<answer>` species decisions. |

## Repository Structure

```text
Bioinspired-Invasive-Fish-Monitoring/
├── object_centric_extractor/     # module 1 — detection / tracking / export / eval
├── open_world_filter/            # module 2 — open-world candidate filtering
├── evidence_grounded_reasoner/   # module 3 — reasoning VLM (SFT + GRPO)
├── checkpoints/                  # local, not tracked — weights (SAM2, GroundingDINO, Qwen3-VL, FG-VLM)
├── data/                         # local, not tracked — datasets (download from Hugging Face)
│   ├── robotfish-data/               # object_centric_extractor: raw videos + ground truth
│   │   ├── video/
│   │   └── label_rectified.json
│   ├── object-centric-sequence-data/ # extracted object-centric sequences + prediction JSON
│   │   └── invasive-fishes-sequence-data/   # → evidence_grounded_reasoner (train / eval)
│   │       ├── videos/               #   train/ and val/
│   │       ├── sft_train.json
│   │       ├── rl_train.json
│   │       └── val.json
│   └── fish-recognition-dataset/     # open_world_filter
│       ├── invasive-dataset/
│       └── SuppExps/                 # generated open-world splits (Invasive-S2/S3/S4-*)
├── work_dirs/                    # local, not tracked — run outputs
├── requirements.txt              # shared Python deps for the two vision modules
└── LICENSE                       # MIT
```

`checkpoints/`, `data/`, and `work_dirs/` are local runtime directories and are not tracked in git.

## Data

Datasets and released model weights are hosted on Hugging Face (not included in this repository):
**[yanglei18/Bioinspired-Molecular-Visual-Monitoring-of-Invasive-Fishes](https://huggingface.co/datasets/yanglei18/Bioinspired-Molecular-Visual-Monitoring-of-Invasive-Fishes)**.

| Hosted item | Module | Purpose |
|---|---|---|
| `robotfish-data.zip` | `object_centric_extractor` | raw robot-fish videos + ground truth — unzip to `data/robotfish-data/` (`video/` + `label_rectified.json`) |
| `object-centric-sequence-data/` | `object_centric_extractor` | object-centric sequences and prediction JSON extracted from `robotfish-data`, scored with `evaluate_pipeline.py` |
| `object-centric-sequence-data/invasive-fishes-sequence-data/` | `evidence_grounded_reasoner` | reasoner training/eval data: object-centric videos + `sft_train.json` / `rl_train.json` / `val.json` |
| `fish-recognition-dataset/` | `open_world_filter` | `data/fish-recognition-dataset/` (then build the SuppExps splits) |
| `checkpoints/` | closed loop (Quick Start) | released **FG-VLM-4B-Thinking** — a Yanghu-pond (10-species) model that classifies `object_centric_extractor` clips for species-level evaluation |

The released **FG-VLM-4B-Thinking** checkpoint (a Yanghu-pond, 10-species model) drives the end-to-end
loop in [Quick Start](#quick-start-end-to-end); download it with:

```bash
huggingface-cli download yanglei18/Bioinspired-Molecular-Visual-Monitoring-of-Invasive-Fishes \
  checkpoints/FG-VLM-4B-Thinking.zip --repo-type dataset --local-dir ./
unzip checkpoints/FG-VLM-4B-Thinking.zip -d checkpoints/   # -> checkpoints/FG-VLM-4B-Thinking/
```

**Data flow.** `robotfish-data` (Yanghu, 10 species) → `object_centric_extractor` extracts
object-centric sequences + prediction JSON (scored by `evaluate_pipeline.py`); the released
`FG-VLM-4B-Thinking` then classifies each clip and its answers are fed back for species-level
evaluation. Separately, `object-centric-sequence-data` (9 invasive species) is used to train and
evaluate a from-scratch FG-VLM in `evidence_grounded_reasoner`. See each module's Tutorial for details.

## Installation

One shared Python 3.10 environment covers the two vision modules
(`object_centric_extractor`, `open_world_filter`):

```bash
conda create -n invasive-fish python=3.10 && conda activate invasive-fish

# Install PyTorch matching your CUDA runtime (example targets CUDA 12.4):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Module-specific setup lives in each module Tutorial:

- **object_centric_extractor** — Grounded-SAM2 / GroundingDINO / TrackEval, plus SAM2 and
  GroundingDINO checkpoints. See its [Tutorial](object_centric_extractor/README.md).
- **open_world_filter** — no extra dependencies beyond the shared environment above; build the
  open-world dataset splits before training/inference. See its [Tutorial](open_world_filter/README.md).
- **evidence_grounded_reasoner** — a separate `ms-swift` / Qwen3-VL environment, plus dataset and
  model-weight downloads (hosted externally). See its [Tutorial](evidence_grounded_reasoner/README.md).

Run all commands from the repository root so config-relative paths resolve correctly.

## Quick Start (end-to-end)

```bash
# 1) Extract object-centric sequences from raw robot-fish videos
python3 object_centric_extractor/run_extract_object_sequences.py
#    -> work_dirs/extraction_output/instance_video/<video>_<instance>.mp4  (object-centric clips)
#    -> work_dirs/extraction_output/annotation_det/prediction.json         (coarse "fish" predictions)

# 2) Coarse detection / tracking evaluation
python3 object_centric_extractor/evaluate_pipeline.py

# 3) Open-world filtering — retain in-scope invasive candidates, reject unknowns
bash open_world_filter/scripts/train_script.sh     open_world_filter/configs/invasive-s4-a.yaml 1
bash open_world_filter/scripts/inference_script.sh open_world_filter/configs/invasive-s4-a.yaml 1

# 4) Fine-grained species confirmation with the released FG-VLM-4B-Thinking checkpoint.
#    Use the 10 Yanghu-pond species (they match the extractor's evaluation classes, so the
#    answers can be fed back in step 5).
python3 evidence_grounded_reasoner/eval/generate_benchmark.py \
  --video-dir work_dirs/extraction_output/instance_video --output benchmark.json \
  --options "black carp" "chinese labeo" "chinese sucker" "redeye barbel" "serrated barb" \
            "common carp" "chinese paddlefish" "mud carp" "schizothorax fish" "wuchang bream"
bash evidence_grounded_reasoner/eval/run_eval_pipeline.sh checkpoints/FG-VLM-4B-Thinking \
  --benchmark-file benchmark.json --run-dir ./eval_output
#    -> eval_output/res.json   (per-clip answers: <answer>(X) species ...</answer>)

# 5) Feed the VLM predictions back for fine-grained (species-level) detection evaluation
python3 object_centric_extractor/evaluate_pipeline.py --inference_json eval_output/res.json
```

**Closed loop (Yanghu-pond evaluation).** Steps 1–2 and 4–5 form an automatic loop on the
robot-fish-collected Yanghu data: the extractor exports object-centric clips named
`<video>_<instance>.mp4`; the released **FG-VLM-4B-Thinking** classifies each clip into a `(A)…(J)`
species answer; and `evaluate_pipeline.py --inference_json` maps those answers back onto the detections
(by instance id, through the shared 10-species vocabulary) and runs species-level evaluation — no
manual conversion needed.

**Separate workflows.** Training FG-VLM from scratch on the **nine invasive species**
(`object-centric-sequence-data`) and scoring it on its own benchmark (`compute_accuracy.py`), and the
`open_world_filter` screening benchmark (step 3), are independent per-module workflows on their own
data — they are not part of the Yanghu feedback loop above. See each module's Tutorial for details.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 the authors of *"Bioinspired
molecular–visual monitoring of invasive fishes"*. The MIT license covers the **code** in this
repository; released **model weights** are derived from Qwen3-VL and additionally inherit the
upstream Qwen3-VL model license. Please also follow the licenses of the upstream projects listed
under Acknowledgements.

## Citation

If you use this code in academic work, please cite the associated paper:

```bibtex
@article{invasive_fish_monitoring_2026,
  title   = {Bioinspired molecular--visual monitoring of invasive fishes},
  author  = {Li, Lei and Li, Yanyu and Yang, Lei and Yu, Junzhi and He, Dekui and Lv, Chen and others},
  year    = {2026},
  note    = {Under review. Lei Li, Yanyu Li and Lei Yang contributed equally (co-first authors); Chen Lv, Dekui He and Junzhi Yu are co-corresponding authors. Venue and DOI to be added upon publication}
}
```

## Acknowledgements

This project builds on SAM2, GroundingDINO, TrackEval, PyTorch, Transformers, ms-swift, Qwen3-VL,
and vLLM. Please follow the licenses and citation requirements of those upstream projects.