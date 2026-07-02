"""YAML-backed configuration for the coarse classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CLASSIFIER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLASSIFIER_ROOT.parent
DEFAULT_CONFIG_PATH = CLASSIFIER_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "Invasive-S4-A"


@dataclass(frozen=True)
class DataConfig:
    train_dir: str = "data/fish-recognition-dataset/SuppExps/Invasive-S4-A/train"
    val_dir: str = "data/fish-recognition-dataset/SuppExps/Invasive-S4-A/val"
    reference_dir: str = "data/fish-recognition-dataset/SuppExps/Invasive-S4-A/train"


@dataclass(frozen=True)
class OutputConfig:
    save_path: str = "work_dirs/open-world-filter-outputs/Invasive-S4-A"
    evaluation_dir: str = "work_dirs/open-world-filter-outputs/Invasive-S4-A/evaluation"
    model_path: str = "work_dirs/open-world-filter-outputs/Invasive-S4-A/fish_classifier_epoch_80.pth"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1.0e-5
    weight_decay: float = 1.0e-6
    warmup_epochs: int = 10
    gradient_clip: float = 0.5
    backbone: str = "resnet50"
    image_size: int = 224
    num_workers: int = 2
    use_class_pairs: bool = True
    profile_batches: int = 0
    samples_per_class: int = 10000000
    save_frequency: int = 10
    val_frequency: int = 100
    drop_last: bool = True
    model_path: str | None = None
    radius_threshold_l2: float = 0.005
    radius_weight_l2: float = 10.0
    center_distance_threshold_l2: float = 20.0
    center_distance_weight_l2: float = 10.0


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int = 32
    image_size: int = 224
    num_workers: int = 4
    similarity_threshold: float = 0.99
    median_threshold: float = 0.99
    variance_threshold: float = 0.01
    l2_distance_threshold: float = 0.0065
    max_samples_per_class: int = 5000
    max_reference_per_class: int = 200
    num_ensembles: int = 1


@dataclass(frozen=True)
class CoarseClassifierConfig:
    experiment: ExperimentConfig
    data: DataConfig
    outputs: OutputConfig
    training: TrainingConfig
    inference: InferenceConfig


def load_classifier_config(config_path: str | Path | None = None) -> CoarseClassifierConfig:
    config_file = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    if not config_file.exists():
        raise FileNotFoundError(f"Coarse classifier config not found: {config_file}")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load open_world_filter YAML configs. "
            "Install it with: pip install pyyaml"
        ) from exc

    with config_file.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"Expected a mapping config in {config_file}")
    return build_classifier_config(raw_config)


def build_classifier_config(raw_config: Mapping[str, Any]) -> CoarseClassifierConfig:
    experiment_section = _section(raw_config, "experiment")
    data_section = _section(raw_config, "data")
    outputs_section = _section(raw_config, "outputs")
    training_section = _section(raw_config, "training")
    inference_section = _section(raw_config, "inference")

    return CoarseClassifierConfig(
        experiment=ExperimentConfig(
            name=str(experiment_section.get("name", ExperimentConfig.name)),
        ),
        data=DataConfig(
            train_dir=str(data_section.get("train_dir", DataConfig.train_dir)),
            val_dir=str(data_section.get("val_dir", DataConfig.val_dir)),
            reference_dir=str(data_section.get("reference_dir", DataConfig.reference_dir)),
        ),
        outputs=OutputConfig(
            save_path=str(outputs_section.get("save_path", OutputConfig.save_path)),
            evaluation_dir=str(outputs_section.get("evaluation_dir", OutputConfig.evaluation_dir)),
            model_path=str(outputs_section.get("model_path", OutputConfig.model_path)),
        ),
        training=TrainingConfig(
            batch_size=int(training_section.get("batch_size", TrainingConfig.batch_size)),
            epochs=int(training_section.get("epochs", TrainingConfig.epochs)),
            lr=float(training_section.get("lr", TrainingConfig.lr)),
            weight_decay=float(training_section.get("weight_decay", TrainingConfig.weight_decay)),
            warmup_epochs=int(training_section.get("warmup_epochs", TrainingConfig.warmup_epochs)),
            gradient_clip=float(training_section.get("gradient_clip", TrainingConfig.gradient_clip)),
            backbone=str(training_section.get("backbone", TrainingConfig.backbone)),
            image_size=int(training_section.get("image_size", TrainingConfig.image_size)),
            num_workers=int(training_section.get("num_workers", TrainingConfig.num_workers)),
            use_class_pairs=bool(training_section.get("use_class_pairs", TrainingConfig.use_class_pairs)),
            profile_batches=int(training_section.get("profile_batches", TrainingConfig.profile_batches)),
            samples_per_class=int(training_section.get("samples_per_class", TrainingConfig.samples_per_class)),
            save_frequency=int(training_section.get("save_frequency", TrainingConfig.save_frequency)),
            val_frequency=int(training_section.get("val_frequency", TrainingConfig.val_frequency)),
            drop_last=bool(training_section.get("drop_last", TrainingConfig.drop_last)),
            model_path=_optional_str(training_section.get("model_path")),
            radius_threshold_l2=float(training_section.get("radius_threshold_l2", TrainingConfig.radius_threshold_l2)),
            radius_weight_l2=float(training_section.get("radius_weight_l2", TrainingConfig.radius_weight_l2)),
            center_distance_threshold_l2=float(
                training_section.get(
                    "center_distance_threshold_l2",
                    TrainingConfig.center_distance_threshold_l2,
                )
            ),
            center_distance_weight_l2=float(
                training_section.get(
                    "center_distance_weight_l2",
                    TrainingConfig.center_distance_weight_l2,
                )
            ),
        ),
        inference=InferenceConfig(
            batch_size=int(inference_section.get("batch_size", InferenceConfig.batch_size)),
            image_size=int(inference_section.get("image_size", InferenceConfig.image_size)),
            num_workers=int(inference_section.get("num_workers", InferenceConfig.num_workers)),
            similarity_threshold=float(
                inference_section.get("similarity_threshold", InferenceConfig.similarity_threshold)
            ),
            median_threshold=float(inference_section.get("median_threshold", InferenceConfig.median_threshold)),
            variance_threshold=float(inference_section.get("variance_threshold", InferenceConfig.variance_threshold)),
            l2_distance_threshold=float(
                inference_section.get("l2_distance_threshold", InferenceConfig.l2_distance_threshold)
            ),
            max_samples_per_class=int(
                inference_section.get("max_samples_per_class", InferenceConfig.max_samples_per_class)
            ),
            max_reference_per_class=int(
                inference_section.get("max_reference_per_class", InferenceConfig.max_reference_per_class)
            ),
            num_ensembles=int(inference_section.get("num_ensembles", InferenceConfig.num_ensembles)),
        ),
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
