# open_world_filter

Open-world candidate filtering: this module learns a discriminative representation with a
momentum-based, dual-branch (self-distillation) objective and uses reference-set matching to retain
in-scope invasive-fish candidates and reject unknown / non-target observations. It covers
representation training, multi-GPU inference, dataset building, and threshold optimization.

## Core Entry Points

- `scripts/train_script.sh`: main training launcher
- `scripts/inference_script.sh`: main multi-GPU inference launcher
- `configs/default.yaml`: default paths and hyperparameters
- `tools/build_dataset_suppExps.py`: build predefined `Invasive-S2/S3/S4-A/B/C/D` SuppExps train/val splits

## Quick Start

Train from the repository root:

```bash
bash open_world_filter/scripts/train_script.sh
```

Run multi-GPU inference:

```bash
bash open_world_filter/scripts/inference_script.sh
```

## Repository Layout

```text
open_world_filter/
├── configs/
│   ├── default.yaml
│   ├── class_thresholds.json
│   ├── invasive-s2-a.yaml
│   ├── invasive-s2-b.yaml
│   ├── invasive-s2-c.yaml
│   ├── invasive-s2-d.yaml
│   ├── invasive-s3-a.yaml
│   ├── invasive-s3-b.yaml
│   ├── invasive-s3-c.yaml
│   ├── invasive-s3-d.yaml
│   ├── invasive-s4-a.yaml
│   ├── invasive-s4-b.yaml
│   ├── invasive-s4-c.yaml
│   └── invasive-s4-d.yaml
├── config.py
├── train.py
├── model.py
├── dataset.py
├── evaluate.py
├── matching.py
├── plots.py
├── scripts/
│   ├── train_script.sh
│   └── inference_script.sh
├── tools/
│   └── build_dataset_suppExps.py
```

## Training

Training paths and hyperparameters are centralized in:

```text
open_world_filter/configs/default.yaml
```

Main launcher:

```bash
bash open_world_filter/scripts/train_script.sh open_world_filter/configs/default.yaml 1
```

The launcher accepts exactly two optional inputs: config path and GPU count.

By default, training reads:

```text
data/fish-recognition-dataset/SuppExps/Invasive-S4-A/train
data/fish-recognition-dataset/SuppExps/Invasive-S4-A/val
```

Use another config to switch experiments:

```bash
bash open_world_filter/scripts/train_script.sh \
  open_world_filter/configs/invasive-s4-b.yaml \
  1

python3 open_world_filter/train.py \
  --config open_world_filter/configs/invasive-s4-b.yaml
```

Predefined experiment configs are available for `Invasive-S2-A/B/C/D`, `Invasive-S3-A/B/C/D`, and `Invasive-S4-A/B/C/D`. Each experiment YAML is self-contained and includes experiment paths, output paths, training parameters, and inference parameters. `default.yaml` remains the default `Invasive-S4-A` experiment.

Training depends on:

- `dataset.py`
- `model.py`

If you need to adapt common training or inference parameters across experiments, update each affected experiment YAML explicitly so every config remains runnable on its own.

## Multi-GPU Inference

Main launcher:

```bash
bash open_world_filter/scripts/inference_script.sh open_world_filter/configs/default.yaml 1
```

The launcher accepts exactly two optional inputs: config path and GPU count.

and depends on:

- `dataset.py`
- `matching.py`
- `plots.py`

By default, inference reads references, validation data, thresholds, and output paths from:

```text
open_world_filter/configs/default.yaml
```

Main outputs typically include:

- `evaluation_summary.json`
- `detailed_predictions.json`
- confusion matrix figures
- open-world screening figures

Edit `outputs.evaluation_dir` in the YAML config to change the evaluation output directory.

## Dataset Building

First download the fish-recognition dataset from the
[Hugging Face dataset](https://huggingface.co/datasets/yanglei18/Bioinspired-Molecular-Visual-Surveillance-of-Invasive-Fishes):

```bash
huggingface-cli download yanglei18/Bioinspired-Molecular-Visual-Surveillance-of-Invasive-Fishes \
  fish-recognition-dataset/invasive-dataset.tar.gz --repo-type dataset --local-dir ./data
tar -xzf data/fish-recognition-dataset/invasive-dataset.tar.gz -C data/fish-recognition-dataset/
```

This yields `data/fish-recognition-dataset/invasive-dataset/` (raw species-labeled crops). Then build
all predefined open-world splits under `SuppExps/`:

```bash
python3 open_world_filter/tools/build_dataset_suppExps.py \
  --source_root data/fish-recognition-dataset/invasive-dataset \
  --target_root data/fish-recognition-dataset/SuppExps
```

The generated datasets include `Invasive-S2-A/B/C/D`, `Invasive-S3-A/B/C/D`, and `Invasive-S4-A/B/C/D`. Each dataset contains `train/`, `val/`, and `val/unknown/`. Files are soft-linked by default; pass `--copy` to materialize independent image copies.

Predefined class splits:

```text
S2 known: brown_trout, crucian_carp
S3 known: brown_trout, crucian_carp, eastern_mosquitofish
S4 known: brown_trout, crucian_carp, eastern_mosquitofish, guppies

A unknown: largemouth_bass, mozambique_tilapia
B unknown: largemouth_bass, mozambique_tilapia, rainbow_trout
C unknown: largemouth_bass, mozambique_tilapia, rainbow_trout, grass_carp
D unknown: largemouth_bass, mozambique_tilapia, rainbow_trout, grass_carp, carp
```

The S4 split uses `guppies` as the canonical output class name. The builder also accepts a source folder named `guppy` for compatibility, but generated train/val folders use `guppies`.

Build only selected experiments:

```bash
python3 open_world_filter/tools/build_dataset_suppExps.py \
  --experiments Invasive-S4-A Invasive-S4-D
```

## Open-World Screening

Open-world screening metrics are implemented in:

- `evaluate.py`
- `plots.py`

These provide:

- in-scope recall
- out-of-scope rejection rate
- candidate purity

The watchlist is discovered automatically from `reference_dir`.
