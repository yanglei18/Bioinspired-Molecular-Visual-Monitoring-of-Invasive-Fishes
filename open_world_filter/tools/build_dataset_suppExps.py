#!/usr/bin/env python3
"""Build SuppExps open-world datasets from unsplit invasive-fish class folders."""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the dataset helper dependency-light.
    tqdm = None


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.9

S2_KNOWN_CLASSES = ("brown_trout", "crucian_carp")
S3_KNOWN_CLASSES = (*S2_KNOWN_CLASSES, "eastern_mosquitofish")
S4_KNOWN_CLASSES = (*S3_KNOWN_CLASSES, "guppies")

UNKNOWN_SET_A = ("largemouth_bass", "mozambique_tilapia")
UNKNOWN_SET_B = (*UNKNOWN_SET_A, "rainbow_trout")
UNKNOWN_SET_C = (*UNKNOWN_SET_B, "grass_carp")
UNKNOWN_SET_D = (*UNKNOWN_SET_C, "carp")
SOURCE_CLASS_ALIASES = {
    "guppies": ("guppies", "guppy"),
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    known_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("Invasive-S2-A", S2_KNOWN_CLASSES, UNKNOWN_SET_A),
    ExperimentSpec("Invasive-S2-B", S2_KNOWN_CLASSES, UNKNOWN_SET_B),
    ExperimentSpec("Invasive-S2-C", S2_KNOWN_CLASSES, UNKNOWN_SET_C),
    ExperimentSpec("Invasive-S2-D", S2_KNOWN_CLASSES, UNKNOWN_SET_D),
    ExperimentSpec("Invasive-S3-A", S3_KNOWN_CLASSES, UNKNOWN_SET_A),
    ExperimentSpec("Invasive-S3-B", S3_KNOWN_CLASSES, UNKNOWN_SET_B),
    ExperimentSpec("Invasive-S3-C", S3_KNOWN_CLASSES, UNKNOWN_SET_C),
    ExperimentSpec("Invasive-S3-D", S3_KNOWN_CLASSES, UNKNOWN_SET_D),
    ExperimentSpec("Invasive-S4-A", S4_KNOWN_CLASSES, UNKNOWN_SET_A),
    ExperimentSpec("Invasive-S4-B", S4_KNOWN_CLASSES, UNKNOWN_SET_B),
    ExperimentSpec("Invasive-S4-C", S4_KNOWN_CLASSES, UNKNOWN_SET_C),
    ExperimentSpec("Invasive-S4-D", S4_KNOWN_CLASSES, UNKNOWN_SET_D),
)
EXPERIMENT_SPEC_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
FileMode = Literal["symlink", "copy"]


@dataclass(frozen=True)
class BuildConfig:
    source_root: Path
    target_root: Path
    spec: ExperimentSpec
    train_ratio: float
    file_mode: FileMode

    @property
    def experiment_root(self) -> Path:
        return self.target_root / self.spec.name

    @property
    def target_train_dir(self) -> Path:
        return self.experiment_root / "train"

    @property
    def target_val_dir(self) -> Path:
        return self.experiment_root / "val"

    @property
    def target_unknown_dir(self) -> Path:
        return self.target_val_dir / "unknown"


@dataclass(frozen=True)
class FileTask:
    source: Path
    target: Path
    split_name: str


@dataclass(frozen=True)
class BuildPlan:
    tasks: tuple[FileTask, ...]
    train_total: int
    val_total: int
    unknown_total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one or more SuppExps open-world datasets from unsplit class folders. "
            "By default, all Invasive-S2/S3/S4-A/B/C/D datasets are generated with "
            "soft links under data/fish-recognition-dataset/SuppExps."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=PROJECT_ROOT / "data/fish-recognition-dataset/invasive-dataset",
        help="Source dataset root that directly contains class subdirectories.",
    )
    parser.add_argument(
        "--target_root",
        type=Path,
        default=PROJECT_ROOT / "data/fish-recognition-dataset/SuppExps",
        help="Target SuppExps root. Each experiment is written under this directory.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=tuple(spec.name for spec in EXPERIMENT_SPECS),
        choices=tuple(spec.name for spec in EXPERIMENT_SPECS),
        help="Experiment names to build. Omit to build all predefined experiments.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating soft links. Soft links are the default.",
    )
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        help="Do not clear existing experiment directories before building.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def dedupe_items(items: Iterable[str]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for item in items:
        normalized = item.strip()
        if normalized:
            ordered[normalized] = None
    return tuple(ordered.keys())


def ensure_source_structure(source_root: Path) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")


def resolve_source_class_dir(source_root: Path, class_name: str) -> Path:
    for candidate_name in SOURCE_CLASS_ALIASES.get(class_name, (class_name,)):
        candidate_dir = source_root / candidate_name
        if candidate_dir.is_dir():
            return candidate_dir
    return source_root / class_name


def validate_experiment_specs(source_root: Path, specs: tuple[ExperimentSpec, ...]) -> None:
    missing_by_experiment: dict[str, list[str]] = {}

    for spec in specs:
        required_classes = set(spec.known_classes) | set(spec.unknown_classes)
        missing_classes = sorted(
            class_name
            for class_name in required_classes
            if not resolve_source_class_dir(source_root, class_name).is_dir()
        )
        overlapping_classes = sorted(set(spec.known_classes) & set(spec.unknown_classes))
        if overlapping_classes:
            raise ValueError(
                f"{spec.name} has classes marked as both known and unknown: "
                + ", ".join(overlapping_classes)
            )
        if missing_classes:
            missing_by_experiment[spec.name] = missing_classes

    if missing_by_experiment:
        details = "; ".join(
            f"{name}: {', '.join(classes)}"
            for name, classes in missing_by_experiment.items()
        )
        raise FileNotFoundError(
            "Missing source class directories under source_root. " + details
        )


def resolve_specs(args: argparse.Namespace) -> tuple[ExperimentSpec, ...]:
    experiment_names = dedupe_items(args.experiments)
    return tuple(EXPERIMENT_SPEC_BY_NAME[name] for name in experiment_names)


def build_config(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    source_root: Path,
    target_root: Path,
) -> BuildConfig:
    return BuildConfig(
        source_root=source_root,
        target_root=target_root,
        spec=spec,
        train_ratio=DEFAULT_TRAIN_RATIO,
        file_mode="copy" if args.copy else "symlink",
    )


def prepare_experiment_root(config: BuildConfig, keep_existing: bool) -> None:
    if config.experiment_root.exists() and not keep_existing:
        shutil.rmtree(config.experiment_root)
    config.target_train_dir.mkdir(parents=True, exist_ok=True)
    config.target_val_dir.mkdir(parents=True, exist_ok=True)
    config.target_unknown_dir.mkdir(parents=True, exist_ok=True)


def relative_symlink_source(source: Path, target: Path) -> Path:
    return Path(os.path.relpath(source, start=target.parent))


def replace_existing_target_if_needed(target: Path, source: Path) -> bool:
    if not target.exists() and not target.is_symlink():
        return False

    if target.is_symlink():
        resolved_target = target.resolve(strict=False)
        resolved_source = source.resolve(strict=False)
        if resolved_target == resolved_source:
            LOGGER.debug("Skip existing symlink: %s", target)
            return True
        target.unlink()
        LOGGER.debug("Replaced symlink: %s", target)
        return False

    if target.is_file():
        same_file = target.stat().st_size == source.stat().st_size
        if same_file:
            LOGGER.debug("Skip existing file: %s", target)
            return True
        target.unlink()
        return False

    raise FileExistsError(
        f"Target exists and is not a file or symlink. Please remove it manually: {target}"
    )


def create_symlink(source: Path, target: Path) -> bool:
    skipped = replace_existing_target_if_needed(target, source)
    if skipped:
        return False

    link_source = relative_symlink_source(source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(link_source, target_is_directory=source.is_dir())
    LOGGER.debug("Linked %s -> %s", target, link_source)
    return True


def copy_file(source: Path, target: Path) -> bool:
    skipped = replace_existing_target_if_needed(target, source)
    if skipped:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    LOGGER.debug("Copied %s -> %s", source, target)
    return True


def materialize_file(source: Path, target: Path, file_mode: FileMode) -> bool:
    if file_mode == "copy":
        return copy_file(source, target)
    return create_symlink(source, target)


def progress_items(items: Iterable, desc: str, unit: str) -> Iterable:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)


def iter_image_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_class_files(
    image_files: list[Path],
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not image_files:
        return [], []

    shuffled_files = list(image_files)
    random.Random(seed).shuffle(shuffled_files)
    train_count = int(len(shuffled_files) * train_ratio)
    train_count = max(1, train_count)
    if len(shuffled_files) > 1:
        train_count = min(train_count, len(shuffled_files) - 1)
    return shuffled_files[:train_count], shuffled_files[train_count:]


def collect_split_class_tasks(config: BuildConfig) -> tuple[list[FileTask], int, int]:
    tasks: list[FileTask] = []
    train_total = 0
    val_total = 0

    for class_name in config.spec.known_classes:
        source_class_dir = resolve_source_class_dir(config.source_root, class_name)
        image_files = iter_image_files(source_class_dir)
        train_files, val_files = split_class_files(
            image_files=image_files,
            train_ratio=config.train_ratio,
            seed=DEFAULT_SPLIT_SEED,
        )

        LOGGER.info(
            "  Known %-24s total=%d train=%d val=%d",
            class_name,
            len(image_files),
            len(train_files),
            len(val_files),
        )

        train_total += len(train_files)
        val_total += len(val_files)
        tasks.extend(
            FileTask(
                source=source_file,
                target=config.target_train_dir / class_name / source_file.name,
                split_name="train",
            )
            for source_file in train_files
        )
        tasks.extend(
            FileTask(
                source=source_file,
                target=config.target_val_dir / class_name / source_file.name,
                split_name="val",
            )
            for source_file in val_files
        )

    return tasks, train_total, val_total


def build_unknown_target_path(
    source_file: Path,
    source_subdir_name: str,
    target_unknown_dir: Path,
    used_names: set[str],
) -> Path:
    candidate_name = source_file.name
    if candidate_name in used_names or (target_unknown_dir / candidate_name).exists():
        candidate_name = f"{source_subdir_name}_{source_file.name}"

    if candidate_name in used_names or (target_unknown_dir / candidate_name).exists():
        stem = source_file.stem
        suffix = source_file.suffix
        index = 1
        while True:
            candidate_name = f"{source_subdir_name}_{stem}_{index}{suffix}"
            if candidate_name not in used_names and not (target_unknown_dir / candidate_name).exists():
                break
            index += 1

    used_names.add(candidate_name)
    return target_unknown_dir / candidate_name


def collect_unknown_tasks(config: BuildConfig) -> tuple[list[FileTask], int]:
    tasks: list[FileTask] = []
    used_names: set[str] = set()

    for class_name in config.spec.unknown_classes:
        source_dir = resolve_source_class_dir(config.source_root, class_name)
        image_files = iter_image_files(source_dir)
        LOGGER.info("  Unknown %-22s total=%d", class_name, len(image_files))

        for source_file in image_files:
            target_path = build_unknown_target_path(
                source_file=source_file,
                source_subdir_name=class_name,
                target_unknown_dir=config.target_unknown_dir,
                used_names=used_names,
            )
            tasks.append(
                FileTask(source=source_file, target=target_path, split_name="unknown")
            )

    return tasks, len(tasks)


def build_plan(config: BuildConfig) -> BuildPlan:
    split_tasks, train_total, val_total = collect_split_class_tasks(config)
    unknown_tasks, unknown_total = collect_unknown_tasks(config)
    return BuildPlan(
        tasks=tuple([*split_tasks, *unknown_tasks]),
        train_total=train_total,
        val_total=val_total,
        unknown_total=unknown_total,
    )


def execute_plan(plan: BuildPlan, file_mode: FileMode, desc: str) -> dict[str, int]:
    counts = {
        "train_created": 0,
        "train_skipped": 0,
        "val_created": 0,
        "val_skipped": 0,
        "unknown_created": 0,
        "unknown_skipped": 0,
    }
    for task in progress_items(plan.tasks, desc=desc, unit="file"):
        action = (
            "created"
            if materialize_file(task.source, task.target, file_mode=file_mode)
            else "skipped"
        )
        counts[f"{task.split_name}_{action}"] += 1
    return counts


def build_one_experiment(config: BuildConfig, keep_existing: bool) -> dict[str, int]:
    LOGGER.info("")
    LOGGER.info("Building %s", config.spec.name)
    LOGGER.info("  Source root: %s", config.source_root)
    LOGGER.info("  Target root: %s", config.experiment_root)
    LOGGER.info("  Mode: %s", config.file_mode)
    LOGGER.info("  Known classes: %s", ", ".join(config.spec.known_classes))
    LOGGER.info("  Unknown classes: %s", ", ".join(config.spec.unknown_classes))

    prepare_experiment_root(config, keep_existing=keep_existing)
    plan = build_plan(config)
    counts = execute_plan(
        plan=plan,
        file_mode=config.file_mode,
        desc=config.spec.name,
    )

    LOGGER.info("  Summary")
    LOGGER.info("    Train files: %d", plan.train_total)
    LOGGER.info("    Val known files: %d", plan.val_total)
    LOGGER.info("    Val unknown files: %d", plan.unknown_total)
    LOGGER.info("    Created: %d", sum(value for key, value in counts.items() if key.endswith("_created")))
    LOGGER.info("    Skipped: %d", sum(value for key, value in counts.items() if key.endswith("_skipped")))
    return counts


def main() -> int:
    configure_logging()
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    specs = resolve_specs(args)

    ensure_source_structure(source_root)
    validate_experiment_specs(source_root, specs)
    target_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building SuppExps datasets")
    LOGGER.info("  Source root: %s", source_root)
    LOGGER.info("  Target root: %s", target_root)
    LOGGER.info("  Experiments: %s", ", ".join(spec.name for spec in specs))
    LOGGER.info("  Default file mode: %s", "copy" if args.copy else "symlink")

    total_created = 0
    total_skipped = 0
    for spec in progress_items(specs, desc="Build experiments", unit="experiment"):
        config = build_config(
            args=args,
            spec=spec,
            source_root=source_root,
            target_root=target_root,
        )
        counts = build_one_experiment(config, keep_existing=args.keep_existing)
        total_created += sum(value for key, value in counts.items() if key.endswith("_created"))
        total_skipped += sum(value for key, value in counts.items() if key.endswith("_skipped"))

    LOGGER.info("")
    LOGGER.info("All requested SuppExps datasets finished.")
    LOGGER.info("  Experiments built: %d", len(specs))
    LOGGER.info("  Files created: %d", total_created)
    LOGGER.info("  Files skipped: %d", total_skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
