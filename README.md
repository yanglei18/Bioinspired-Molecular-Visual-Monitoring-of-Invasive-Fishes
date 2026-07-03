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
| [`open_world_filter`](open_world_filter/README.md) | Open-world candidate filtering: BYOL-style representation training + reference-set matching to retain in-scope invasive candidates and reject unknowns. |
| [`evidence_grounded_reasoner`](evidence_grounded_reasoner/README.md) | Evidence-grounded fine-grained confirmation: a reasoning VLM (Qwen3-VL, Think-SFT + GRPO) producing `<think>/<rethink>/<answer>` species decisions. |

## Repository Structure

```text
Bioinspired-Invasive-Fish-Monitoring/
├── object_centric_extractor/     # module 1 — detection / tracking / export / eval
├── open_world_filter/            # module 2 — open-world candidate filtering
├── evidence_grounded_reasoner/   # module 3 — reasoning VLM (Think-SFT + GRPO)
├── checkpoints/                  # local: SAM2 / GroundingDINO weights (not tracked)
├── data/                         # local: datasets (not tracked)
├── work_dirs/                    # local: run outputs (not tracked)
├── requirements.txt              # shared Python deps for the two vision modules
└── LICENSE                       # MIT
```

`checkpoints/`, `data/`, and `work_dirs/` are local runtime directories and are not tracked in git.

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
# 1) Extract object-centric sequences (detection + tracking + instance export)
python3 object_centric_extractor/run_extract_object_sequences.py

# 2) Detection / tracking evaluation
python3 object_centric_extractor/evaluate_pipeline.py

# 3) Train and evaluate the open-world filter
bash open_world_filter/scripts/train_script.sh     open_world_filter/configs/invasive-s4-a.yaml 1
bash open_world_filter/scripts/inference_script.sh open_world_filter/configs/invasive-s4-a.yaml 1

# 4) Fine-grained species confirmation with the reasoning VLM
#    (see evidence_grounded_reasoner/README.md — training, eval, and weights)
```

Full options, data preparation, configuration, and output layouts for each step are documented in
the corresponding module Tutorial linked in the [Modules](#modules) table above.

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