#!/usr/bin/env python3
"""Convert SAM2 detector outputs to KITTI format for evaluation."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SAM2_DETECTOR_DIR = SCRIPT_DIR.parent
if str(SAM2_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DETECTOR_DIR))

try:
    from utils.annotation_io import load_prediction_sequences
    from utils.fish_label_map import FINE_CLASSES, normalize_fish_class_name
except ModuleNotFoundError:
    from object_centric_extractor.utils.annotation_io import load_prediction_sequences
    from object_centric_extractor.utils.fish_label_map import FINE_CLASSES, normalize_fish_class_name

KITTI_COMPAT_CLASS = "Car"
FRAME_KEY_PATTERN = re.compile(r"^(?:mask_)?(?P<frame_id>\d+)$", re.IGNORECASE)
PREDICTION_EXPORT_FIELDNAMES = (
    "sequence_name",
    "frame_key",
    "frame_id",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
)


def parse_frame_id(frame_key: str) -> int | None:
    match = FRAME_KEY_PATTERN.match(frame_key)
    if match is None:
        return None
    return int(match.group("frame_id"))


def parse_prediction_frame(frame_key: str, annotation: dict) -> tuple[int | None, list[dict]]:
    frame_id = parse_frame_id(frame_key)
    if frame_id is None:
        return None, []

    records: list[dict] = []
    for label_data in annotation.get("labels", {}).values():
        records.append(
            {
                "frame_id": frame_id,
                "track_id": int(label_data["instance_id"]),
                "class_name": normalize_fish_class_name(label_data.get("class_name")),
                "x1": float(label_data["x1"]),
                "y1": float(label_data["y1"]),
                "x2": float(label_data["x2"]),
                "y2": float(label_data["y2"]),
                "score": float(label_data.get("logit", 0.0)),
            }
        )
    return frame_id, records


def convert_sequence_dir_to_kitti(
    frame_annotations: dict[str, dict],
    output_file: Path,
    target_class: str | None = None,
    allowed_frame_ids: set[int] | None = None,
    allowed_frame_range: tuple[int, int] | None = None,
    verbose: bool = False,
) -> int:
    """Convert one sequence prediction mapping to a KITTI prediction txt file."""
    target_class = normalize_fish_class_name(target_class)
    detections: list[str] = []
    skipped_frame_keys = 0

    for frame_key, annotation in frame_annotations.items():
        frame_id, records = parse_prediction_frame(frame_key, annotation)
        if frame_id is None:
            skipped_frame_keys += 1
            continue
        if allowed_frame_ids is not None and frame_id not in allowed_frame_ids:
            continue
        if allowed_frame_range is not None:
            min_frame_id, max_frame_id = allowed_frame_range
            if frame_id < min_frame_id or frame_id > max_frame_id:
                continue

        for record in records:
            if target_class and record["class_name"] != target_class:
                continue
            detections.append(
                (
                    f"{record['frame_id']} {record['track_id']} {KITTI_COMPAT_CLASS} -1 -1 -1 "
                    f"{record['x1']} {record['y1']} {record['x2']} {record['y2']} "
                    f"-1 -1 -1 -1 -1 -1 -1 {record['score']}\n"
                )
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.writelines(detections)
    if verbose and skipped_frame_keys:
        print(f"Skipped {skipped_frame_keys} non-numeric frame keys in {output_file.stem}")
    return len(detections)


def export_prediction_details(
    input_dir: Path,
    output_path: Path,
    target_class: str | None = None,
    allowed_sequence_names: set[str] | None = None,
) -> dict[str, int | str]:
    target_class = normalize_fish_class_name(target_class)
    sequences = load_prediction_sequences(input_dir)
    if allowed_sequence_names is not None:
        sequences = {
            seq_name: frame_annotations
            for seq_name, frame_annotations in sequences.items()
            if Path(seq_name).name in allowed_sequence_names
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    non_empty_sequence_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_EXPORT_FIELDNAMES)
        writer.writeheader()

        for seq_name, frame_annotations in sequences.items():
            output_sequence_name = Path(seq_name).name
            sequence_row_count = 0
            for frame_key, annotation in frame_annotations.items():
                frame_id, records = parse_prediction_frame(frame_key, annotation)
                if frame_id is None:
                    continue
                for record in records:
                    if target_class and record["class_name"] != target_class:
                        continue
                    writer.writerow(
                        {
                            "sequence_name": output_sequence_name,
                            "frame_key": frame_key,
                            "frame_id": frame_id,
                            "track_id": record["track_id"],
                            "class_name": record["class_name"],
                            "x1": record["x1"],
                            "y1": record["y1"],
                            "x2": record["x2"],
                            "y2": record["y2"],
                            "score": record["score"],
                        }
                    )
                    row_count += 1
                    sequence_row_count += 1
            if sequence_row_count > 0:
                non_empty_sequence_count += 1

    return {
        "row_count": row_count,
        "sequence_count": len(sequences),
        "non_empty_sequence_count": non_empty_sequence_count,
        "output_path": str(output_path),
    }


def convert_to_kitti_format(
    input_dir: Path,
    output_dir: Path,
    target_class: str | None = None,
    allowed_sequence_names: set[str] | None = None,
    allowed_frame_ids_by_sequence: dict[str, set[int]] | None = None,
    allowed_frame_ranges_by_sequence: dict[str, tuple[int, int]] | None = None,
    verbose: bool = False,
) -> dict[str, int]:
    """Convert SAM2 prediction annotations to KITTI txt files."""
    target_class = normalize_fish_class_name(target_class)
    if target_class and target_class not in FINE_CLASSES:
        raise ValueError(
            f"Unsupported target_class {target_class}. Expected one of: {', '.join(FINE_CLASSES)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_prediction_sequences(input_dir)
    if allowed_sequence_names is not None:
        sequences = {
            seq_name: frame_annotations
            for seq_name, frame_annotations in sequences.items()
            if Path(seq_name).name in allowed_sequence_names
        }
    if not sequences:
        raise FileNotFoundError(
            f"No prediction annotations found under {input_dir}. "
            "Expected a root prediction.json, a single sequence prediction.json, "
            "or legacy per-frame JSON directories."
        )

    normalized_sequence_names = [Path(seq_name).name for seq_name in sequences]
    duplicates = {
        seq_name
        for seq_name in normalized_sequence_names
        if normalized_sequence_names.count(seq_name) > 1
    }
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates))
        raise ValueError(
            "Found duplicated sequence names after normalizing prediction keys: "
            f"{duplicate_text}. Please ensure each video name is unique."
        )

    summary: dict[str, int] = {}
    for seq_name, frame_annotations in sequences.items():
        output_sequence_name = Path(seq_name).name
        output_file = output_dir / f"{output_sequence_name}.txt"
        count = convert_sequence_dir_to_kitti(
            frame_annotations,
            output_file,
            target_class=target_class,
            allowed_frame_ids=None
            if allowed_frame_ids_by_sequence is None
            else allowed_frame_ids_by_sequence.get(output_sequence_name, set()),
            allowed_frame_range=None
            if allowed_frame_ranges_by_sequence is None
            else allowed_frame_ranges_by_sequence.get(output_sequence_name),
            verbose=verbose,
        )
        summary[output_sequence_name] = count
        if verbose:
            print(f"Saved {count} detections to {output_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SAM2 detector outputs to KITTI prediction files for Fish evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="work_dirs/extraction_output/annotation_det",
        help="Prediction root containing prediction.json or legacy per-frame JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work_dirs/extraction_output/kitti_eval_run/pred/SAM2Detector/data",
        help="Output directory for KITTI format txt files",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default=None,
        help="Optional fine-grained class filter, e.g. carp",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sequence conversion details.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    if args.verbose:
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")

    if not input_dir.exists():
        print(f"Warning: Input directory {input_dir} does not exist")
        return

    summary = convert_to_kitti_format(
        input_dir,
        output_dir,
        target_class=args.target_class,
        verbose=args.verbose,
    )
    print(f"Conversion completed for {len(summary)} sequences.")


if __name__ == "__main__":
    main()
