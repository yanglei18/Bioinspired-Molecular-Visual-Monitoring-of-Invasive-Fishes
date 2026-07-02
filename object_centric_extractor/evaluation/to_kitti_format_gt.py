#!/usr/bin/env python3
"""Convert fish detection labels to KITTI format for evaluation."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SAM2_DETECTOR_DIR = SCRIPT_DIR.parent
if str(SAM2_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DETECTOR_DIR))

try:
    from utils.annotation_io import get_existing_aggregated_annotation_path, load_prediction_sequences
    from utils.fish_label_map import (
        FINE_CLASSES,
        LABEL_ID_TO_NAME,
        get_fish_class_name,
        normalize_fish_class_name,
    )
except ModuleNotFoundError:
    from object_centric_extractor.utils.annotation_io import (
        get_existing_aggregated_annotation_path,
        load_prediction_sequences,
    )
    from object_centric_extractor.utils.fish_label_map import (
        FINE_CLASSES,
        LABEL_ID_TO_NAME,
        get_fish_class_name,
        normalize_fish_class_name,
    )

KITTI_COMPAT_CLASS = "Car"




def parse_gt_records(input_file: str) -> list[dict]:
    """Parse raw label records into structured GT annotations."""
    records: list[dict] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue

            class_name = get_fish_class_name(parts[2])
            if class_name is None:
                continue

            x = float(parts[3])
            y = float(parts[4])
            w = float(parts[5])
            h = float(parts[6])
            records.append(
                {
                    "frame_id": int(parts[0]),
                    "track_id": int(parts[1]),
                    "label_id": parts[2],
                    "class_name": class_name,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "score": float(parts[7]),
                }
            )
    return records


def parse_gt_json_records(frame_annotations: dict[str, dict]) -> list[dict]:
    records: list[dict] = []
    for frame_key, annotation in frame_annotations.items():
        frame_token = frame_key
        if frame_token.startswith("mask_"):
            frame_token = frame_token[5:]
        try:
            frame_id = int(frame_token)
        except ValueError:
            continue

        for label_data in annotation.get("labels", {}).values():
            class_name = normalize_fish_class_name(label_data.get("class_name"))
            if class_name is None:
                continue
            instance_id = label_data.get("instance_id")
            if instance_id is None:
                continue
            records.append(
                {
                    "frame_id": frame_id,
                    "track_id": int(instance_id),
                    "label_id": None,
                    "class_name": class_name,
                    "x1": float(label_data["x1"]),
                    "y1": float(label_data["y1"]),
                    "x2": float(label_data["x2"]),
                    "y2": float(label_data["y2"]),
                    "score": float(label_data.get("logit", 1.0)),
                }
            )
    return records


def load_gt_sequences(input_path: Path) -> dict[str, list[dict]]:
    txt_files = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix == ".txt") if input_path.is_dir() else []
    if txt_files:
        return {
            txt_file.stem: parse_gt_records(str(txt_file))
            for txt_file in txt_files
        }

    aggregated_path = get_existing_aggregated_annotation_path(input_path)
    if aggregated_path is not None or input_path.is_file() or input_path.is_dir():
        sequences = load_prediction_sequences(input_path)
        if sequences:
            normalized_sequence_names = [Path(sequence_key).name for sequence_key in sequences]
            duplicates = {
                sequence_name
                for sequence_name in normalized_sequence_names
                if normalized_sequence_names.count(sequence_name) > 1
            }
            if duplicates:
                duplicate_text = ", ".join(sorted(duplicates))
                raise ValueError(
                    "Found duplicated sequence names after normalizing GT keys: "
                    f"{duplicate_text}. Please ensure each video name is unique."
                )
            normalized_sequences: dict[str, list[dict]] = {}
            for sequence_key, frame_annotations in sequences.items():
                sequence_name = Path(sequence_key).name
                normalized_sequences[sequence_name] = parse_gt_json_records(frame_annotations)
            return normalized_sequences

    raise FileNotFoundError(
        f"No supported GT annotations found under {input_path}. "
        "Expected label_rectified *.txt files or annotation_det-style JSON annotations."
    )


def collect_gt_sequence_names(input_path: Path) -> set[str]:
    return set(load_gt_sequences(input_path).keys())


def collect_gt_frame_ids_by_sequence(input_path: Path) -> dict[str, set[int]]:
    return {
        sequence_name: {record["frame_id"] for record in records}
        for sequence_name, records in load_gt_sequences(input_path).items()
    }




def convert_to_kitti_format(
    input_file: str,
    output_file: str,
    target_class: str | None = None,
) -> int:
    """
    Convert a single label file from fish format to KITTI format.

    Input format:
    frame_id,track_id,label_id,x,y,w,h,score,?,?
    """
    target_class = normalize_fish_class_name(target_class)
    written = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for record in parse_gt_records(input_file):
            if target_class and record["class_name"] != target_class:
                continue

            kitti_line = (
                f"{record['frame_id']} {record['track_id']} {KITTI_COMPAT_CLASS} -1 -1 -1 "
                f"{record['x1']:.6f} {record['y1']:.6f} {record['x2']:.6f} {record['y2']:.6f} "
                f"-1 -1 -1 -1000 -1000 -1000 -10 {record['score']:.6f}\n"
            )
            f_out.write(kitti_line)
            written += 1
    return written


def write_seqmaps(seqmap_dir: Path, seqmap_entries: list[str]) -> list[Path]:
    """Write both training/testing seqmaps for broader compatibility."""
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    seqmap_paths = [
        seqmap_dir / "evaluate_tracking.seqmap.training",
        seqmap_dir / "evaluate_tracking.seqmap.testing",
    ]
    for seqmap_path in seqmap_paths:
        with seqmap_path.open("w", encoding="utf-8") as f:
            for entry in seqmap_entries:
                f.write(entry + "\n")
    return seqmap_paths


def convert_directory_to_kitti(
    input_dir: Path,
    output_dir: Path,
    count_only: bool = False,
    target_class: str | None = None,
    verbose: bool = False,
) -> dict:
    """Convert a GT label directory to KITTI format and write seqmaps."""
    target_class = normalize_fish_class_name(target_class)
    if target_class and target_class not in FINE_CLASSES:
        raise ValueError(
            f"Unsupported target_class {target_class}. Expected one of: {', '.join(FINE_CLASSES)}"
        )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    gt_sequences = load_gt_sequences(input_dir)
    if not gt_sequences:
        raise FileNotFoundError(f"No supported GT annotations found in {input_dir}")

    if not count_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    seqmap_entries: list[str] = []
    total_category_counts = Counter()
    file_detection_counts: dict[str, int] = {}

    for sequence_name, records in sorted(gt_sequences.items()):
        file_counts = Counter()
        filtered_records: list[dict] = []
        for record in records:
            class_name = record["class_name"]
            if target_class and class_name != target_class:
                continue
            filtered_records.append(record)
            file_counts[class_name] += 1
        total_category_counts.update(file_counts)

        if verbose:
            print(f"\nFile: {sequence_name}.txt")
            for category, count in sorted(file_counts.items()):
                print(f"  {category}: {count}")

        if count_only:
            continue

        output_path = output_dir / f"{sequence_name}.txt"
        if verbose:
            print(f"Converting {sequence_name}.txt...")
        with output_path.open("w", encoding="utf-8") as f_out:
            for record in filtered_records:
                kitti_line = (
                    f"{record['frame_id']} {record['track_id']} {KITTI_COMPAT_CLASS} -1 -1 -1 "
                    f"{record['x1']:.6f} {record['y1']:.6f} {record['x2']:.6f} {record['y2']:.6f} "
                    f"-1 -1 -1 -1000 -1000 -1000 -10 {record['score']:.6f}\n"
                )
                f_out.write(kitti_line)
        written = len(filtered_records)
        file_detection_counts[sequence_name] = written
        if records:
            frame_ids = [record["frame_id"] for record in records]
            min_frame, max_frame = min(frame_ids), max(frame_ids)
            seqmap_entries.append(f"{sequence_name} empty {min_frame:06d} {max_frame + 1:06d}")

    seqmap_paths: list[Path] = []
    if not count_only:
        seqmap_paths = write_seqmaps(output_dir.parent, seqmap_entries)
        if verbose:
            for seqmap_path in seqmap_paths:
                print(f"Seqmap file generated at: {seqmap_path}")

    return {
        "file_count": len(gt_sequences),
        "category_counts": dict(total_category_counts),
        "total_objects": sum(total_category_counts.values()),
        "seqmap_paths": [str(path) for path in seqmap_paths],
        "target_class": target_class,
        "file_detection_counts": file_detection_counts,
        "label_map": LABEL_ID_TO_NAME,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fish detection labels to KITTI format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/robotfish-data/label_rectified.json",
        help="Input directory containing label_rectified txt files or annotation_det-style JSON annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work_dirs/extraction_output/kitti_eval_run/gt/label_02",
        help="Output directory for KITTI format labels",
    )
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Only count labels without converting to KITTI format",
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
        help="Print per-file conversion details.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not args.count_only:
        print(f"Converting labels from {input_dir} to {output_dir}")
    else:
        print(f"Counting labels in {input_dir}")

    summary = convert_directory_to_kitti(
        input_dir,
        output_dir,
        count_only=args.count_only,
        target_class=args.target_class,
        verbose=args.verbose,
    )

    print("\nTotal category counts across all files:")
    for category, count in sorted(summary["category_counts"].items()):
        print(f"  {category}: {count}")
    print(f"Total objects: {summary['total_objects']}")


if __name__ == "__main__":
    main()
