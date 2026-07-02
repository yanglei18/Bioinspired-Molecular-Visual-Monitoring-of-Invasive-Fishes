"""Configuration dataclasses for the SAM2 detector pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class RuntimeConfig:
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    grounding_model_id: str = "checkpoints/grounding-dino-base"


@dataclass(frozen=True)
class DataConfig:
    input_dir: str = "data/robotfish-data/video"
    gt_label_json: str = "data/robotfish-data/label_rectified.json"


@dataclass(frozen=True)
class OutputConfig:
    det_dir: str
    mask_dir: str
    masked_image_dir: str | None
    masked_video_dir: str | None
    instance_image_dir: str
    instance_video_dir: str


@dataclass(frozen=True)
class CropConfig:
    min_tracking_frames: int = 10
    padding: int = 50
    min_size: int = 32
    fixed_window: bool = True
    scale_factor: float = 1.0
    percentile: int = 80
    class_filter: tuple[str, ...] = ("fish", "carp")


@dataclass(frozen=True)
class DetectionConfig:
    text_prompt: str = "fish."
    box_threshold: float = 0.30
    iou_threshold: float = 0.65
    min_edge_threshold: int = 100
    step: int = 10


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: str
    outputs: OutputConfig
    crop: CropConfig
    detection: DetectionConfig
    do_tracking: bool
    do_cropping: bool
    cleanup_temp: bool
    enable_visualization: bool


@dataclass(frozen=True)
class EvaluationConfig:
    pred_json_dir: str = "work_dirs/extraction_output/annotation_det"
    gt_label_dir: str = "data/robotfish-data/label_rectified.json"
    inference_json: str | None = None
    work_dir: str = "work_dirs/extraction_output/kitti_eval_run"
    tracker_name: str = "SAM2Detector"
    skip_detection: bool = False
    skip_tracking: bool = False
    no_plot: bool = False
    verbose: bool = False
    frame_align_mode: str = "auto"


@dataclass(frozen=True)
class Sam2DetectorConfig:
    runtime: RuntimeConfig
    data: DataConfig
    pipeline: PipelineConfig
    evaluation: EvaluationConfig


def load_detector_config(config_path: str | Path | None = None) -> Sam2DetectorConfig:
    """Load SAM2 detector defaults from a YAML config file."""
    config_file = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    if not config_file.exists():
        raise FileNotFoundError(f"SAM2 detector config not found: {config_file}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load object_centric_extractor YAML configs. "
            "Install it with: pip install pyyaml"
        ) from exc

    with config_file.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Expected a mapping config in {config_file}")

    return build_detector_config(raw_config)


def build_detector_config(raw_config: Mapping[str, Any]) -> Sam2DetectorConfig:
    runtime_section = _section(raw_config, "runtime")
    data_section = _section(raw_config, "data")
    outputs_section = _section(raw_config, "outputs")
    detection_section = _section(raw_config, "detection")
    crop_section = _section(raw_config, "crop")
    pipeline_section = _section(raw_config, "pipeline")
    evaluation_section = _section(raw_config, "evaluation")

    data = DataConfig(
        input_dir=str(data_section.get("input_dir", DataConfig.input_dir)),
        gt_label_json=str(data_section.get("gt_label_json", DataConfig.gt_label_json)),
    )
    outputs = OutputConfig(
        det_dir=str(outputs_section.get("det_dir", "work_dirs/extraction_output/annotation_det")),
        mask_dir=str(outputs_section.get("mask_dir", "work_dirs/extraction_output/annotation_mask")),
        masked_image_dir=_optional_str(outputs_section.get("masked_image_dir", "work_dirs/extraction_output/masked_image")),
        masked_video_dir=_optional_str(outputs_section.get("masked_video_dir", "work_dirs/extraction_output/masked_video")),
        instance_image_dir=str(outputs_section.get("instance_image_dir", "work_dirs/extraction_output/instance_image")),
        instance_video_dir=str(outputs_section.get("instance_video_dir", "work_dirs/extraction_output/instance_video")),
    )
    crop = CropConfig(
        min_tracking_frames=int(crop_section.get("min_tracking_frames", CropConfig.min_tracking_frames)),
        padding=int(crop_section.get("padding", CropConfig.padding)),
        min_size=int(crop_section.get("min_size", CropConfig.min_size)),
        fixed_window=bool(crop_section.get("fixed_window", CropConfig.fixed_window)),
        scale_factor=float(crop_section.get("scale_factor", CropConfig.scale_factor)),
        percentile=int(crop_section.get("percentile", CropConfig.percentile)),
        class_filter=tuple(crop_section.get("class_filter", CropConfig.class_filter)),
    )
    detection = DetectionConfig(
        text_prompt=str(detection_section.get("text_prompt", DetectionConfig.text_prompt)),
        box_threshold=float(detection_section.get("box_threshold", DetectionConfig.box_threshold)),
        iou_threshold=float(detection_section.get("iou_threshold", DetectionConfig.iou_threshold)),
        min_edge_threshold=int(detection_section.get("min_edge_threshold", DetectionConfig.min_edge_threshold)),
        step=int(detection_section.get("step", DetectionConfig.step)),
    )
    pipeline = PipelineConfig(
        input_dir=data.input_dir,
        outputs=outputs,
        crop=crop,
        detection=detection,
        do_tracking=bool(pipeline_section.get("do_tracking", True)),
        do_cropping=bool(pipeline_section.get("do_cropping", True)),
        cleanup_temp=bool(pipeline_section.get("cleanup_temp", True)),
        enable_visualization=bool(pipeline_section.get("enable_visualization", True)),
    )
    evaluation = EvaluationConfig(
        pred_json_dir=str(evaluation_section.get("pred_json_dir", outputs.det_dir)),
        gt_label_dir=str(evaluation_section.get("gt_label_dir", data.gt_label_json)),
        inference_json=_optional_str(evaluation_section.get("inference_json")),
        work_dir=str(evaluation_section.get("work_dir", "work_dirs/extraction_output/kitti_eval_run")),
        tracker_name=str(evaluation_section.get("tracker_name", "SAM2Detector")),
        skip_detection=bool(evaluation_section.get("skip_detection", False)),
        skip_tracking=bool(evaluation_section.get("skip_tracking", False)),
        no_plot=bool(evaluation_section.get("no_plot", False)),
        verbose=bool(evaluation_section.get("verbose", False)),
        frame_align_mode=str(evaluation_section.get("frame_align_mode", "auto")),
    )
    runtime = RuntimeConfig(
        sam2_checkpoint=str(runtime_section.get("sam2_checkpoint", RuntimeConfig.sam2_checkpoint)),
        sam2_model_cfg=str(runtime_section.get("sam2_model_cfg", RuntimeConfig.sam2_model_cfg)),
        grounding_model_id=str(runtime_section.get("grounding_model_id", RuntimeConfig.grounding_model_id)),
    )
    return Sam2DetectorConfig(
        runtime=runtime,
        data=data,
        pipeline=pipeline,
        evaluation=evaluation,
    )


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return section


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
