"""Update annotation_det labels from instance-video classification predictions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from utils.annotation_io import (
        AGGREGATED_ANNOTATION_FILENAME,
        DATASET_SEQUENCE_KEY,
        LEGACY_AGGREGATED_ANNOTATION_FILENAMES,
        get_existing_aggregated_annotation_path,
    )
    from utils.fish_label_map import LABEL_ID_TO_NAME
except ModuleNotFoundError:
    from object_centric_extractor.utils.annotation_io import (
        AGGREGATED_ANNOTATION_FILENAME,
        DATASET_SEQUENCE_KEY,
        LEGACY_AGGREGATED_ANNOTATION_FILENAMES,
        get_existing_aggregated_annotation_path,
    )
    from object_centric_extractor.utils.fish_label_map import LABEL_ID_TO_NAME


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
OPTION_PATTERN = re.compile(r"\(([A-Ja-j])\)\s*([^\n]+)")
INSTANCE_VIDEO_PATTERN = re.compile(r"^(?P<video_name>.+)_(?P<instance_id>\d+)\.mp4$", re.IGNORECASE)
CANONICAL_CLASS_NAMES = set(LABEL_ID_TO_NAME.values())
CLASS_ALIASES = {
    "black_carp": "black_carp",
    "black carp": "black_carp",
    "chinese_labeo": "chinese_labeo",
    "chinese labeo": "chinese_labeo",
    "chinese_sucker": "chinese_sucker",
    "chinese sucker": "chinese_sucker",
    "redeye_barbel": "redeye_barbel",
    "redeye barbel": "redeye_barbel",
    "serrated_barb": "serrated_barb",
    "serrated barb": "serrated_barb",
    "common_carp": "carp",
    "common carp": "carp",
    "carp": "carp",
    "chinese_paddlefish": "chinese_paddlefish",
    "chinese paddlefish": "chinese_paddlefish",
    "mud_carp": "mud_carp",
    "mud carp": "mud_carp",
    "schizothorax_fish": "schizothorax_fish",
    "schizothorax fish": "schizothorax_fish",
    "wuchang_bream": "wuchang_bream",
    "wuchang bream": "wuchang_bream",
}


@dataclass(frozen=True)
class LoadStats:
    results_loaded: int = 0
    valid_predictions: int = 0
    conflicts: int = 0
    skipped_results: int = 0


@dataclass(frozen=True)
class UpdateStats:
    updated_sequences: int = 0
    updated_frames: int = 0
    updated_labels: int = 0
    unmatched_videos: int = 0


@dataclass(frozen=True)
class UpdateConfig:
    inference_json: Path
    annotation_det_dir: Path
    dry_run: bool = False
    summary_path: Path | None = None


def normalize_class_name(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip().lower().replace("-", "_").replace(" ", "_")
    if stripped in CANONICAL_CLASS_NAMES:
        return stripped
    if text.strip().lower() in CLASS_ALIASES:
        return CLASS_ALIASES[text.strip().lower()]
    return CLASS_ALIASES.get(text.strip().lower().replace("_", " "))


def parse_question_options(question: str | None) -> dict[str, str]:
    if not question:
        return {}
    option_map: dict[str, str] = {}
    for option_key, option_text in OPTION_PATTERN.findall(question):
        normalized = normalize_class_name(option_text)
        if normalized is not None:
            option_map[option_key.upper()] = normalized
    return option_map


def extract_answer_text(model_raw_output: str) -> str:
    match = ANSWER_PATTERN.search(model_raw_output)
    if match:
        return match.group(1).strip()
    return model_raw_output.strip()


def resolve_predicted_class(question: str | None, model_raw_output: str) -> str | None:
    option_map = parse_question_options(question)
    answer_text = extract_answer_text(model_raw_output)

    option_match = re.search(r"\(([A-Ja-j])\)", answer_text)
    if option_match:
        predicted_class = option_map.get(option_match.group(1).upper())
        if predicted_class is not None:
            return predicted_class

    normalized_answer = normalize_class_name(answer_text)
    if normalized_answer is not None:
        return normalized_answer

    for alias_text, canonical_name in CLASS_ALIASES.items():
        if alias_text in answer_text.strip().lower():
            return canonical_name
    return None


def parse_instance_video(video_path: str) -> tuple[str, int] | None:
    video_name = Path(video_path).name
    match = INSTANCE_VIDEO_PATTERN.match(video_name)
    if match is None:
        return None
    return match.group("video_name"), int(match.group("instance_id"))


def load_prediction_updates(inference_json: Path) -> tuple[dict[str, dict[int, str]], LoadStats]:
    payload = json.loads(inference_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    updates: dict[str, dict[int, str]] = {}
    conflicts = 0
    skipped = 0
    valid = 0

    for item in results:
        parsed_video = parse_instance_video(str(item.get("video_path", "")))
        predicted_class = resolve_predicted_class(item.get("question"), item.get("model_raw_output", ""))
        if parsed_video is None or predicted_class is None:
            skipped += 1
            continue

        video_name, instance_id = parsed_video
        video_updates = updates.setdefault(video_name, {})
        previous_value = video_updates.get(instance_id)
        if previous_value is not None and previous_value != predicted_class:
            conflicts += 1
        video_updates[instance_id] = predicted_class
        valid += 1

    return updates, LoadStats(
        results_loaded=len(results),
        valid_predictions=valid,
        conflicts=conflicts,
        skipped_results=skipped,
    )


def update_frame_payload(frame_payload: dict, instance_updates: dict[int, str]) -> tuple[bool, int]:
    labels = frame_payload.get("labels", {})
    changed = False
    updated_labels = 0
    for object_id, object_data in labels.items():
        instance_id = int(object_data.get("instance_id", object_id))
        predicted_class = instance_updates.get(instance_id)
        if predicted_class is None or object_data.get("class_name") == predicted_class:
            continue
        object_data["class_name"] = predicted_class
        changed = True
        updated_labels += 1
    return changed, updated_labels


def update_aggregated_annotation_file(
    annotation_path: Path,
    updates: dict[str, dict[int, str]],
    dry_run: bool,
) -> UpdateStats:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    sequences = payload.get(DATASET_SEQUENCE_KEY)
    if sequences is None:
        sequence_name = payload.get("video_name") or annotation_path.stem
        sequences_to_process = {sequence_name: payload}
    else:
        sequences_to_process = sequences

    stats = UpdateStats()
    unmatched_videos = set(updates.keys())
    changed = False

    for sequence_key, sequence_payload in sequences_to_process.items():
        video_name = Path(sequence_key).name
        instance_updates = updates.get(video_name)
        if not instance_updates:
            continue
        unmatched_videos.discard(video_name)

        frames = sequence_payload.get("frames", {})
        sequence_changed = False
        updated_frames = 0
        updated_labels = 0
        for frame_payload in frames.values():
            frame_changed, frame_updated_labels = update_frame_payload(frame_payload, instance_updates)
            if frame_changed:
                sequence_changed = True
                updated_frames += 1
                updated_labels += frame_updated_labels

        if sequence_changed:
            changed = True
            stats = UpdateStats(
                updated_sequences=stats.updated_sequences + 1,
                updated_frames=stats.updated_frames + updated_frames,
                updated_labels=stats.updated_labels + updated_labels,
            )

    if changed and not dry_run:
        annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return UpdateStats(
        updated_sequences=stats.updated_sequences,
        updated_frames=stats.updated_frames,
        updated_labels=stats.updated_labels,
        unmatched_videos=len(unmatched_videos),
    )


def discover_legacy_sequence_dirs(annotation_det_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for root in sorted(path for path in annotation_det_dir.rglob("*") if path.is_dir()):
        json_files = [
            path
            for path in root.glob("*.json")
            if path.name not in (AGGREGATED_ANNOTATION_FILENAME, *LEGACY_AGGREGATED_ANNOTATION_FILENAMES)
        ]
        if json_files:
            discovered[str(root.relative_to(annotation_det_dir))] = root
    return discovered


def update_legacy_sequence_dir(
    sequence_dir: Path,
    instance_updates: dict[int, str],
    dry_run: bool,
) -> tuple[int, int]:
    updated_frames = 0
    updated_labels = 0
    for json_file in sorted(sequence_dir.glob("*.json")):
        if json_file.name in (AGGREGATED_ANNOTATION_FILENAME, *LEGACY_AGGREGATED_ANNOTATION_FILENAMES):
            continue
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        changed, frame_updated_labels = update_frame_payload(payload, instance_updates)
        if not changed:
            continue
        updated_frames += 1
        updated_labels += frame_updated_labels
        if not dry_run:
            json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return updated_frames, updated_labels


def update_legacy_annotation_dirs(
    annotation_det_dir: Path,
    updates: dict[str, dict[int, str]],
    dry_run: bool,
) -> UpdateStats:
    legacy_dirs = discover_legacy_sequence_dirs(annotation_det_dir)
    unmatched_videos = set(updates.keys())
    stats = UpdateStats()

    for sequence_key, sequence_dir in legacy_dirs.items():
        video_name = Path(sequence_key).name
        instance_updates = updates.get(video_name)
        if not instance_updates:
            continue
        unmatched_videos.discard(video_name)
        updated_frames, updated_labels = update_legacy_sequence_dir(sequence_dir, instance_updates, dry_run)
        if updated_frames == 0:
            continue
        stats = UpdateStats(
            updated_sequences=stats.updated_sequences + 1,
            updated_frames=stats.updated_frames + updated_frames,
            updated_labels=stats.updated_labels + updated_labels,
        )

    return UpdateStats(
        updated_sequences=stats.updated_sequences,
        updated_frames=stats.updated_frames,
        updated_labels=stats.updated_labels,
        unmatched_videos=len(unmatched_videos),
    )


def resolve_summary_path(summary_path: Path | None, annotation_det_dir: Path) -> Path:
    if summary_path is not None:
        return summary_path.expanduser().resolve()
    return annotation_det_dir / "push_video_prediction_summary.json"


def build_update_summary(
    inference_json: Path,
    annotation_det_dir: Path,
    annotation_mode: str,
    target_path: Path,
    dry_run: bool,
    updates: dict[str, dict[int, str]],
    load_stats: LoadStats,
    update_stats: UpdateStats,
) -> dict[str, Any]:
    return {
        "inference_json": str(inference_json),
        "annotation_det_dir": str(annotation_det_dir),
        "annotation_mode": annotation_mode,
        "target_path": str(target_path),
        "dry_run": dry_run,
        "prediction_updates": {
            video_name: {str(instance_id): class_name for instance_id, class_name in sorted(instance_updates.items())}
            for video_name, instance_updates in sorted(updates.items())
        },
        "stats": {
            "load": asdict(load_stats),
            "update": asdict(update_stats),
        },
    }


def run_update(config: UpdateConfig) -> dict[str, Any]:
    inference_json = config.inference_json.expanduser().resolve()
    annotation_det_dir = config.annotation_det_dir.expanduser().resolve()

    if not inference_json.is_file():
        raise FileNotFoundError(f"inference_json does not exist: {inference_json}")
    if not annotation_det_dir.exists():
        raise FileNotFoundError(f"annotation_det_dir does not exist: {annotation_det_dir}")

    updates, load_stats = load_prediction_updates(inference_json)
    if not updates:
        raise ValueError(f"No valid instance-video predictions were parsed from {inference_json}")

    aggregated_path = get_existing_aggregated_annotation_path(annotation_det_dir)
    if aggregated_path is not None:
        update_stats = update_aggregated_annotation_file(aggregated_path, updates, config.dry_run)
        annotation_mode = "aggregated"
        target_path = aggregated_path
    else:
        update_stats = update_legacy_annotation_dirs(annotation_det_dir, updates, config.dry_run)
        annotation_mode = "legacy"
        target_path = annotation_det_dir

    summary = build_update_summary(
        inference_json=inference_json,
        annotation_det_dir=annotation_det_dir,
        annotation_mode=annotation_mode,
        target_path=target_path,
        dry_run=config.dry_run,
        updates=updates,
        load_stats=load_stats,
        update_stats=update_stats,
    )

    summary_path = resolve_summary_path(config.summary_path, annotation_det_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded {load_stats.results_loaded} results from {inference_json}")
    print(f"Valid prediction mappings: {load_stats.valid_predictions}")
    print(f"Conflicting duplicate mappings: {load_stats.conflicts}")
    print(f"Skipped invalid results: {load_stats.skipped_results}")
    print(f"Annotation mode: {annotation_mode}")
    print(f"Updated sequences: {update_stats.updated_sequences}")
    print(f"Updated frames: {update_stats.updated_frames}")
    print(f"Updated labels: {update_stats.updated_labels}")
    print(f"Unmatched videos: {update_stats.unmatched_videos}")
    print(f"Saved summary to {summary_path}")
    return summary
