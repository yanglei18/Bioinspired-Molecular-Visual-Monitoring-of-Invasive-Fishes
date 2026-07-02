"""Shared helper functions for SAM2 detector tools."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from pipeline.input_discovery import is_frame_directory
    from utils.annotation_io import (
        AGGREGATED_ANNOTATION_FILENAME,
        LEGACY_AGGREGATED_ANNOTATION_FILENAMES,
        load_prediction_sequences,
        load_sequence_annotations,
    )
except ModuleNotFoundError:
    from object_centric_extractor.pipeline.input_discovery import is_frame_directory
    from object_centric_extractor.utils.annotation_io import (
        AGGREGATED_ANNOTATION_FILENAME,
        LEGACY_AGGREGATED_ANNOTATION_FILENAMES,
        load_prediction_sequences,
        load_sequence_annotations,
    )

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class VideoTask:
    video_dir: Path
    video_name: str
    sequence_key: str


@dataclass(frozen=True)
class AnnotationIndex:
    sequences: dict[str, dict[str, dict]]
    basename_to_keys: dict[str, list[str]]
    legacy_sequence_dirs: dict[str, Path]
    legacy_basename_to_keys: dict[str, list[str]]


@dataclass(frozen=True)
class ResolvedVideoTask:
    task: VideoTask
    resolved_sequence_key: str
    mask_sequence_dir: Path
    frame_annotations: dict[str, dict]


def discover_video_tasks(input_root: Path) -> list[VideoTask]:
    tasks: list[VideoTask] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue

        if is_frame_directory(child):
            tasks.append(VideoTask(video_dir=child, video_name=child.name, sequence_key=child.name))
            continue

        for nested_child in sorted(grandchild for grandchild in child.iterdir() if grandchild.is_dir()):
            if is_frame_directory(nested_child):
                tasks.append(
                    VideoTask(
                        video_dir=nested_child,
                        video_name=nested_child.name,
                        sequence_key=str(nested_child.relative_to(input_root)),
                    )
                )
    return tasks


def discover_legacy_annotation_dirs(annotation_det_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for root, _, files in os.walk(annotation_det_dir):
        root_path = Path(root)
        json_files = [
            filename
            for filename in files
            if filename.lower().endswith(".json")
            and filename not in (AGGREGATED_ANNOTATION_FILENAME, *LEGACY_AGGREGATED_ANNOTATION_FILENAMES)
        ]
        if json_files:
            discovered[str(root_path.relative_to(annotation_det_dir))] = root_path
    return discovered


def build_annotation_index(annotation_det_dir: Path) -> AnnotationIndex:
    sequences = load_prediction_sequences(annotation_det_dir)
    basename_to_keys: dict[str, list[str]] = defaultdict(list)
    for sequence_key in sequences:
        basename_to_keys[Path(sequence_key).name].append(sequence_key)

    legacy_sequence_dirs = discover_legacy_annotation_dirs(annotation_det_dir)
    legacy_basename_to_keys: dict[str, list[str]] = defaultdict(list)
    for sequence_key in legacy_sequence_dirs:
        legacy_basename_to_keys[Path(sequence_key).name].append(sequence_key)

    return AnnotationIndex(
        sequences=sequences,
        basename_to_keys={key: sorted(value) for key, value in basename_to_keys.items()},
        legacy_sequence_dirs=legacy_sequence_dirs,
        legacy_basename_to_keys={key: sorted(value) for key, value in legacy_basename_to_keys.items()},
    )


def resolve_sequence_key(task: VideoTask, annotation_index: AnnotationIndex) -> str | None:
    for candidate in [task.sequence_key, task.video_name]:
        if candidate in annotation_index.sequences:
            return candidate
        if candidate in annotation_index.legacy_sequence_dirs:
            return candidate

    basename_matches = annotation_index.basename_to_keys.get(task.video_name, [])
    if len(basename_matches) == 1:
        LOGGER.info("Resolved %s by basename match: %s", task.video_name, basename_matches[0])
        return basename_matches[0]
    if len(basename_matches) > 1:
        LOGGER.warning("Multiple annotation sequence keys match basename %s: %s", task.video_name, basename_matches)
        return None

    legacy_basename_matches = annotation_index.legacy_basename_to_keys.get(task.video_name, [])
    if len(legacy_basename_matches) == 1:
        LOGGER.info("Resolved %s by legacy basename match: %s", task.video_name, legacy_basename_matches[0])
        return legacy_basename_matches[0]
    if len(legacy_basename_matches) > 1:
        LOGGER.warning(
            "Multiple legacy annotation directories match basename %s: %s",
            task.video_name,
            legacy_basename_matches,
        )
    return None


def resolve_mask_sequence_dir(annotation_mask_dir: Path, sequence_key: str, video_name: str) -> Path | None:
    direct_candidates = [
        annotation_mask_dir / sequence_key,
        annotation_mask_dir / video_name,
    ]
    for candidate in direct_candidates:
        if candidate.is_dir() and any(
            path.is_file() and path.suffix.lower() == ".npy" for path in candidate.iterdir()
        ):
            return candidate

    basename_matches = [
        candidate
        for candidate in annotation_mask_dir.rglob("*")
        if candidate.is_dir()
        and candidate.name == video_name
        and any(path.is_file() and path.suffix.lower() == ".npy" for path in candidate.iterdir())
    ]
    if len(basename_matches) == 1:
        LOGGER.info("Resolved mask directory by basename match: %s -> %s", video_name, basename_matches[0])
        return basename_matches[0]
    if len(basename_matches) > 1:
        LOGGER.warning("Multiple mask directories match basename %s: %s", video_name, [str(path) for path in basename_matches])
    return None


def resolve_video_task(
    task: VideoTask,
    annotation_index: AnnotationIndex,
    annotation_mask_dir: Path,
) -> ResolvedVideoTask | None:
    resolved_sequence_key = resolve_sequence_key(task, annotation_index)
    if resolved_sequence_key is None:
        return None

    if resolved_sequence_key in annotation_index.sequences:
        frame_annotations = annotation_index.sequences.get(resolved_sequence_key, {})
    else:
        legacy_dir = annotation_index.legacy_sequence_dirs.get(resolved_sequence_key)
        frame_annotations = load_sequence_annotations(legacy_dir) if legacy_dir is not None else {}

    if not frame_annotations:
        return None

    mask_sequence_dir = resolve_mask_sequence_dir(annotation_mask_dir, resolved_sequence_key, task.video_name)
    if mask_sequence_dir is None:
        return None

    return ResolvedVideoTask(
        task=task,
        resolved_sequence_key=resolved_sequence_key,
        mask_sequence_dir=mask_sequence_dir,
        frame_annotations=frame_annotations,
    )


def sort_frame_names(frame_names: list[str]) -> list[str]:
    try:
        return sorted(frame_names, key=lambda name: int(Path(name).stem))
    except ValueError:
        return sorted(frame_names)


def build_frame_mappings(frame_names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    original_to_padded: dict[str, str] = {}
    padded_to_original: dict[str, str] = {}
    for frame_name in frame_names:
        frame_path = Path(frame_name)
        try:
            padded_name = f"{int(frame_path.stem):06d}{frame_path.suffix}"
        except ValueError:
            padded_name = frame_name
        original_to_padded[frame_name] = padded_name
        padded_to_original[padded_name] = frame_name
    return original_to_padded, padded_to_original
