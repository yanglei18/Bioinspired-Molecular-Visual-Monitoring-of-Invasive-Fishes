# object_centric_extractor

Object-centric sequence generation: this module turns raw underwater video into object-centric fish
sequences via Grounded-SAM2 detection, mask segmentation, cross-frame tracking, and instance
crop/video export, together with detection/tracking evaluation. The recommended user-facing
entrypoints are:

- `run_extract_object_sequences.py`: main inference / tracking / crop export pipeline
- `evaluate_pipeline.py`: unified prediction update, detection, and tracking evaluation

## Repository Layout

```text
object_centric_extractor/
├── configs/
│   └── default.yaml                    # default paths and hyperparameters
├── run_extract_object_sequences.py   # main inference entrypoint
├── evaluate_pipeline.py                # main evaluation entrypoint
├── pipeline/                           # inference / tracking / export services
├── evaluation/                         # detection, tracking, reporting, KITTI conversion
│   ├── to_kitti_format_gt.py           # GT annotation conversion
│   └── to_kitti_format_pred.py         # prediction annotation conversion
├── tools/
│   ├── common.py                       # shared helpers for visualization tools
│   ├── visualization.py
│   └── visualization_gt.py
├── utils/                              # shared IO / tracking / visualization helpers
└── README.md
```

## Environment Setup

1. Download pretrained checkpoints into the repository-level `checkpoints/` directory:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

pip install -U huggingface_hub
huggingface-cli download IDEA-Research/grounding-dino-base \
  --local-dir checkpoints/grounding-dino-base
```

2. Install PyTorch first. Our local environment uses `python=3.10`, `torch>=2.3.1`, `torchvision>=0.18.1`, and CUDA 12.4.

```bash
conda create -n object_centric_extractor python=3.10
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python pillow numpy tqdm matplotlib transformers pyyaml
```

3. Install Grounded-SAM2:

```bash
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
```

4. Install GroundingDINO:

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

## Data

Download the raw robot-fish videos and ground-truth labels from the
[Hugging Face dataset](https://huggingface.co/datasets/yanglei18/Bioinspired-Molecular-Visual-Surveillance-of-Invasive-Fishes):

```bash
huggingface-cli download yanglei18/Bioinspired-Molecular-Visual-Surveillance-of-Invasive-Fishes \
  robotfish-data.zip --repo-type dataset --local-dir ./data
unzip data/robotfish-data.zip -d data/
```

This yields `data/robotfish-data/video/` (input videos/frames) and
`data/robotfish-data/label_rectified.json` (evaluation ground truth) — matching the default paths in
`configs/default.yaml`.

## Inference

Default paths, checkpoints, thresholds, crop settings, and evaluation options are defined in:

```text
object_centric_extractor/configs/default.yaml
```

Run from the repository root:

```bash
python3 object_centric_extractor/run_extract_object_sequences.py
```

Use `--config <yaml>` to run another experiment config. Runtime paths, thresholds, crop settings, and pipeline switches are read from the YAML file.

The configured `data.input_dir` supports both:

- frame directories: `video/<video_name>/` or `video/<fish_type>/<video_name>/`
- video files: `video/*.mp4` or `video/<fish_type>/*.mp4`

When the input is a video file, the script first extracts temporary frames and then reuses the same tracking, visualization, and instance export pipeline. Temporary extracted frames are removed automatically after each video finishes.

The temporary extraction root is fixed to:

- `<input_dir>_image`

Nested layouts keep their relative subpaths under that directory.

### Main Inference Outputs

- `work_dirs/extraction_output/annotation_det/prediction.json`
- `work_dirs/extraction_output/instance_image/*.webp`
- `work_dirs/extraction_output/instance_video/*.mp4`

Legacy `annotation_det/annotations.json` is still readable for backward compatibility during evaluation.

### Common Configuration Fields

- `detection.text_prompt`: GroundingDINO prompt.
- `detection.step`: sample interval for detection prompts.
- `detection.box_threshold`: box confidence threshold.
- `detection.iou_threshold`: mask update IoU threshold.
- `detection.min_edge_threshold`: minimum detection-box edge length.
- `pipeline.do_tracking`: enable or skip tracking.
- `pipeline.do_cropping`: enable or skip instance crop and video export.
- `pipeline.enable_visualization`: enable or skip masked visualization outputs.
- `pipeline.cleanup_temp`: remove or keep intermediate outputs.

### Output Naming

- `webp`: `<video_name>_<instance_id>_<frame_id>.webp`
- `mp4`: `<video_name>_<instance_id>.mp4`

Examples:

- `work_dirs/extraction_output/instance_image/PRO_VID_20250409_135930_00_014_012_3_006600.webp`
- `work_dirs/extraction_output/instance_video/PRO_VID_20250409_135930_00_014_012_3.mp4`

## Prediction Update During Evaluation

If you classify exported instance videos externally and want to evaluate the updated fine-grained labels, set `evaluation.inference_json` in `object_centric_extractor/configs/default.yaml` or pass `--inference_json` for a one-off run:

```yaml
evaluation:
  inference_json: "work_dirs/extraction_output/instance_video_results.json"
```

```bash
python3 object_centric_extractor/evaluate_pipeline.py \
  --inference_json work_dirs/extraction_output/instance_video_results.json
```

When `evaluation.inference_json` is set, the pipeline updates `class_name` fields in `evaluation.pred_json_dir` and then runs fine-grained detection evaluation. When it is `null`, the pipeline leaves prediction annotations unchanged and runs coarse detection evaluation.

Internally, this uses `utils/annotation_update.py`, which exposes `UpdateConfig` and `run_update()` for tests or future tools.

## Evaluation

Install evaluation dependencies:

```bash
conda create -n eval python=3.8
pip install git+https://github.com/JonathonLuiten/TrackEval
pip install numpy==1.21.0
```

The evaluation entrypoint is fish-oriented at the user level. Detection is reported as fish classes, while generated KITTI files still store `Car` for compatibility with KITTI / TrackEval tooling.

Internally, `evaluate_pipeline.py` is a compatibility wrapper around:

- `evaluation/pipeline.py`: top-level orchestration
- `evaluation/detection.py`: coarse/fine detection AP orchestration
- `evaluation/tracking.py`: TrackEval orchestration
- `evaluation/reporting.py`: JSON/text report helpers
- `evaluation/to_kitti_format_gt.py`: GT annotation conversion
- `evaluation/to_kitti_format_pred.py`: prediction annotation conversion

Run:

```bash
python3 object_centric_extractor/evaluate_pipeline.py
```

### Evaluation Inputs

- `evaluation.pred_json_dir`: prediction directory containing `prediction.json`, default `work_dirs/extraction_output/annotation_det`
- `evaluation.gt_label_dir`: GT annotation-det style JSON, default `data/robotfish-data/label_rectified.json`
- `evaluation.inference_json` or `--inference_json`: optional instance-video inference JSON, default `null`

`evaluate_pipeline.py` now assumes JSON GT and no longer accepts raw `label_rectified/*.txt`.

### Optional Evaluation Fields

- `evaluation.skip_tracking`: only run detection AP / mAP
- `evaluation.skip_detection`: only run tracking metrics
- `evaluation.no_plot`: disable PR curve generation
- `evaluation.work_dir`: choose a custom output directory

### Main Evaluation Outputs

- `detection_frame_map.csv`
- `detection_summary.json`
- `evaluation_report.json`
- `evaluation_report.txt`
- `tracking_results/`

## Visualization Tools

- `tools/visualization.py`: render prediction annotations and masks back onto original frames
- `tools/visualization_gt.py`: render GT-driven visualization for qualitative inspection

## Notes for Release

- New experiments and user-facing workflows should rely on the core entrypoints:
  - `run_extract_object_sequences.py`
  - `evaluate_pipeline.py`
