"""Utilities for reading and writing aggregated annotation JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path


AGGREGATED_ANNOTATION_FILENAME = "prediction.json"
LEGACY_AGGREGATED_ANNOTATION_FILENAMES = (
    "annotations.json",
    # Compatibility with historical outputs that misspelled prediction.json.
    "prediction.json",
)
DATASET_SEQUENCE_KEY = "videos"


def frame_sort_key(frame_key: str) -> tuple[int, str]:
    try:
        return (0, f"{int(frame_key):06d}")
    except ValueError:
        return (1, frame_key)


def get_aggregated_annotation_path(annotation_path: str | os.PathLike[str]) -> Path:
    annotation_path = Path(annotation_path)
    if annotation_path.suffix == ".json":
        return annotation_path
    return annotation_path / AGGREGATED_ANNOTATION_FILENAME


def get_existing_aggregated_annotation_path(annotation_path: str | os.PathLike[str]) -> Path | None:
    annotation_path = Path(annotation_path)
    candidate_paths: list[Path] = []
    if annotation_path.suffix == ".json":
        candidate_paths.append(annotation_path)
    else:
        candidate_paths.append(annotation_path / AGGREGATED_ANNOTATION_FILENAME)
        candidate_paths.extend(annotation_path / name for name in LEGACY_AGGREGATED_ANNOTATION_FILENAMES)

    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path
    return None




def _load_json_payload(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_frame_annotation_payload(payload: dict) -> bool:
    return isinstance(payload.get("labels"), dict)


def _sorted_frames(frames: dict[str, dict]) -> dict[str, dict]:
    return {
        frame_key: frames[frame_key]
        for frame_key in sorted(frames.keys(), key=frame_sort_key)
    }


def _get_sequence_payload(payload: dict, sequence_key: str | None) -> dict | None:
    if DATASET_SEQUENCE_KEY in payload:
        if sequence_key is None:
            return None
        return payload.get(DATASET_SEQUENCE_KEY, {}).get(sequence_key)
    return payload


def _discover_legacy_sequence_dirs(annotation_dir: Path) -> dict[str, Path]:
    sequences: dict[str, Path] = {}
    for child in sorted(path for path in annotation_dir.rglob("*") if path.is_dir()):
        child_frames = load_sequence_annotations(child)
        if not child_frames:
            continue
        sequences[str(child.relative_to(annotation_dir))] = child
    return sequences


def load_sequence_annotations(
    annotation_path: str | os.PathLike[str],
    sequence_key: str | None = None,
) -> dict[str, dict]:
    aggregated_path = get_existing_aggregated_annotation_path(annotation_path)
    if aggregated_path is not None:
        payload = _load_json_payload(aggregated_path)
        sequence_payload = _get_sequence_payload(payload, sequence_key=sequence_key)
        if sequence_payload is None:
            return {}
        return sequence_payload.get("frames", {})

    annotation_dir = Path(annotation_path)
    if not annotation_dir.is_dir():
        return {}

    frames: dict[str, dict] = {}
    for json_file in sorted(annotation_dir.glob("*.json")):
        if json_file.name in (AGGREGATED_ANNOTATION_FILENAME, *LEGACY_AGGREGATED_ANNOTATION_FILENAMES):
            continue
        payload = _load_json_payload(json_file)
        if not _is_frame_annotation_payload(payload):
            continue
        frames[json_file.stem] = payload
    return frames


def load_prediction_sequences(annotation_path: str | os.PathLike[str]) -> dict[str, dict[str, dict]]:
    annotation_path = Path(annotation_path)
    aggregated_path = get_existing_aggregated_annotation_path(annotation_path)
    if aggregated_path is not None:
        payload = _load_json_payload(aggregated_path)
        if DATASET_SEQUENCE_KEY in payload:
            return {
                sequence_key: _sorted_frames(sequence_payload.get("frames", {}))
                for sequence_key, sequence_payload in payload.get(DATASET_SEQUENCE_KEY, {}).items()
            }

        sequence_name = payload.get("video_name") or aggregated_path.parent.name or aggregated_path.stem
        return {sequence_name: _sorted_frames(payload.get("frames", {}))}

    if annotation_path.is_file():
        payload = _load_json_payload(annotation_path)
        if DATASET_SEQUENCE_KEY in payload:
            return {
                sequence_key: _sorted_frames(sequence_payload.get("frames", {}))
                for sequence_key, sequence_payload in payload.get(DATASET_SEQUENCE_KEY, {}).items()
            }

        if not _is_frame_annotation_payload(payload):
            return {}
        sequence_name = payload.get("video_name") or annotation_path.stem
        return {sequence_name: _sorted_frames(payload.get("frames", {}))}

    if not annotation_path.is_dir():
        return {}

    direct_frames = load_sequence_annotations(annotation_path)
    if direct_frames:
        return {annotation_path.name: _sorted_frames(direct_frames)}

    sequences: dict[str, dict[str, dict]] = {}
    for child in sorted(path for path in annotation_path.iterdir() if path.is_dir()):
        child_frames = load_sequence_annotations(child)
        if child_frames:
            sequences[child.name] = _sorted_frames(child_frames)
    if sequences:
        return sequences

    for sequence_key, sequence_dir in _discover_legacy_sequence_dirs(annotation_path).items():
        child_frames = load_sequence_annotations(sequence_dir)
        if child_frames:
            sequences[sequence_key] = _sorted_frames(child_frames)
    return sequences


def count_sequence_annotation_frames(
    annotation_path: str | os.PathLike[str],
    sequence_key: str | None = None,
) -> int:
    aggregated_path = get_existing_aggregated_annotation_path(annotation_path)
    if aggregated_path is not None:
        payload = _load_json_payload(aggregated_path)
        sequence_payload = _get_sequence_payload(payload, sequence_key=sequence_key)
        if sequence_payload is None:
            return 0
        return len(sequence_payload.get("frames", {}))

    annotation_dir = Path(annotation_path)
    if not annotation_dir.is_dir():
        return 0

    return sum(
        1
        for json_file in annotation_dir.glob("*.json")
        if json_file.name not in (AGGREGATED_ANNOTATION_FILENAME, *LEGACY_AGGREGATED_ANNOTATION_FILENAMES)
        and _is_frame_annotation_payload(_load_json_payload(json_file))
    )


def write_sequence_annotations(
    annotation_path: str | os.PathLike[str],
    video_name: str,
    frame_annotations: dict[str, dict],
    sequence_key: str | None = None,
) -> Path:
    aggregated_path = get_aggregated_annotation_path(annotation_path)
    aggregated_path.parent.mkdir(parents=True, exist_ok=True)

    sequence_payload = {
        "video_name": video_name,
        "frame_count": len(frame_annotations),
        "frames": _sorted_frames(frame_annotations),
    }

    if sequence_key is None:
        payload = sequence_payload
    else:
        if aggregated_path.is_file():
            payload = _load_json_payload(aggregated_path)
            if DATASET_SEQUENCE_KEY not in payload:
                existing_sequence_key = payload.get("video_name") or aggregated_path.parent.name
                payload = {
                    DATASET_SEQUENCE_KEY: {
                        existing_sequence_key: payload,
                    }
                }
        else:
            payload = {
                DATASET_SEQUENCE_KEY: {},
            }
        payload.setdefault(DATASET_SEQUENCE_KEY, {})[sequence_key] = sequence_payload
        payload["sequence_count"] = len(payload[DATASET_SEQUENCE_KEY])

    with aggregated_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return aggregated_path
