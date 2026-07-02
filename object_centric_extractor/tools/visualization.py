#!/usr/bin/env python3
"""Render SAM2 prediction annotations into visualization images and videos."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from tools.common import (
    ResolvedVideoTask,
    build_annotation_index,
    build_frame_mappings,
    discover_video_tasks,
    resolve_video_task,
    sort_frame_names,
)
from utils.visualization_utils import (
    IMAGE_EXTENSIONS,
    create_video_from_images,
    has_completed_outputs,
    render_frame,
    resolve_mask_path,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderStats:
    total_videos: int = 0
    processed_videos: int = 0
    skipped_videos: int = 0
    failed_videos: int = 0
    rendered_frames: int = 0
    rendered_videos: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize prediction annotation_det + annotation_mask on top of "
            "original video frames and export frame images + mp4 videos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/robotfish-data/video"),
        help="Input root containing video/<video_name> or video/<fish_type>/<video_name> frame directories.",
    )
    parser.add_argument(
        "--annotation_det_dir",
        type=Path,
        default=Path("work_dirs/extraction_output/annotation_det"),
        help="Prediction annotation directory. Supports prediction.json and legacy per-frame JSON folders.",
    )
    parser.add_argument(
        "--annotation_mask_dir",
        type=Path,
        default=Path("work_dirs/extraction_output/annotation_mask"),
        help="Mask directory containing per-frame .npy instance masks.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=Path,
        default=Path("work_dirs/extraction_output/visualization_image"),
        help="Output directory for per-frame visualization images. Images are written under <output_image_dir>/<video_name>/.",
    )
    parser.add_argument(
        "--output_video_dir",
        type=Path,
        default=Path("work_dirs/extraction_output/visualization_video"),
        help="Output directory for mp4 visualization videos named <video_name>.mp4.",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=15,
        help="Frame rate for exported visualization videos.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        choices=[".jpg", ".png"],
        help="Image extension for rendered frame outputs.",
    )
    parser.add_argument(
        "--class_filter",
        nargs="+",
        default=None,
        help="Optional list of class names to visualize. By default all classes are shown.",
    )
    parser.add_argument(
        "--background_dim_factor",
        type=float,
        default=0.9,
        help="Brightness factor applied to non-target background regions. Use 1.0 to keep background unchanged.",
    )
    parser.add_argument(
        "--foreground_feather_sigma",
        type=float,
        default=11.0,
        help="Gaussian blur sigma used to softly blend foreground and darkened background.",
    )
    parser.add_argument(
        "--foreground_gain",
        type=float,
        default=1.35,
        help="Brightness multiplier applied to target foreground regions.",
    )
    parser.add_argument(
        "--outline_glow_sigma",
        type=float,
        default=7.0,
        help="Gaussian blur sigma used for the mask contour glow.",
    )
    parser.add_argument(
        "--box_thickness",
        type=int,
        default=5,
        help="Bounding box thickness.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos whose visualization frames and mp4 already exist.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_rate < 1:
        raise ValueError("--frame_rate must be >= 1")
    if not 0.0 < args.background_dim_factor <= 1.0:
        raise ValueError("--background_dim_factor must be in (0, 1]")
    if args.foreground_gain <= 0:
        raise ValueError("--foreground_gain must be > 0")
    if args.foreground_feather_sigma <= 0:
        raise ValueError("--foreground_feather_sigma must be > 0")
    if args.outline_glow_sigma <= 0:
        raise ValueError("--outline_glow_sigma must be > 0")
    if args.box_thickness < 1:
        raise ValueError("--box_thickness must be >= 1")


def normalize_class_filter(class_filter: list[str] | None) -> set[str] | None:
    if not class_filter:
        return None
    return {class_name.strip().lower() for class_name in class_filter}


def process_video(
    resolved_task: ResolvedVideoTask,
    output_image_root: Path,
    output_video_root: Path,
    image_ext: str,
    class_filter: set[str] | None,
    background_dim_factor: float,
    foreground_gain: float,
    foreground_feather_sigma: float,
    outline_glow_sigma: float,
    box_thickness: int,
    frame_rate: int,
    skip_existing: bool,
) -> tuple[int, int] | None:
    video_name = resolved_task.task.video_name
    image_output_dir = output_image_root / video_name
    video_output_path = output_video_root / f"{video_name}.mp4"

    if skip_existing and has_completed_outputs(image_output_dir, video_output_path):
        LOGGER.info("Skipping %s: visualization outputs already exist.", video_name)
        return 0, 0

    image_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_root.mkdir(parents=True, exist_ok=True)

    frame_names = sort_frame_names(
        [
            path.name
            for path in resolved_task.task.video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    original_to_padded, _ = build_frame_mappings(frame_names)

    rendered_frame_count = 0
    for frame_name in tqdm(frame_names, desc=f"Rendering {video_name}", leave=False):
        frame_path = resolved_task.task.video_dir / frame_name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            LOGGER.warning("Skipping unreadable frame %s", frame_path)
            continue

        padded_name = original_to_padded[frame_name]
        frame_key = Path(padded_name).stem
        frame_annotation = resolved_task.frame_annotations.get(frame_key)
        mask_array = None
        mask_path = resolve_mask_path(resolved_task.mask_sequence_dir, frame_key)
        if mask_path is not None:
            try:
                mask_array = np.load(mask_path)
            except Exception as exc:
                LOGGER.warning("Failed to load mask %s: %s", mask_path, exc)

        rendered = render_frame(
            frame=frame,
            frame_annotation=frame_annotation,
            mask_array=mask_array,
            class_filter=class_filter,
            background_dim_factor=background_dim_factor,
            foreground_gain=foreground_gain,
            feather_sigma=foreground_feather_sigma,
            outline_glow_sigma=outline_glow_sigma,
            box_thickness=box_thickness,
            show_labels=True,
        )
        output_image_path = image_output_dir / f"{Path(frame_name).stem}{image_ext}"
        cv2.imwrite(str(output_image_path), rendered)
        rendered_frame_count += 1

    video_written = create_video_from_images(image_output_dir, video_output_path, frame_rate=frame_rate)
    return rendered_frame_count, int(video_written)


def resolve_summary_path(args: argparse.Namespace, output_video_dir: Path) -> Path:
    if args.summary_path is not None:
        return args.summary_path.expanduser().resolve()
    return output_video_dir / "visualization_summary.json"


def main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = args.input_dir.expanduser().resolve()
    annotation_det_dir = args.annotation_det_dir.expanduser().resolve()
    annotation_mask_dir = args.annotation_mask_dir.expanduser().resolve()
    output_image_dir = args.output_image_dir.expanduser().resolve()
    output_video_dir = args.output_video_dir.expanduser().resolve()
    class_filter = normalize_class_filter(args.class_filter)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not annotation_det_dir.exists():
        raise FileNotFoundError(f"annotation_det_dir does not exist: {annotation_det_dir}")
    if not annotation_mask_dir.exists():
        raise FileNotFoundError(f"annotation_mask_dir does not exist: {annotation_mask_dir}")

    tasks = discover_video_tasks(input_dir)
    if not tasks:
        raise FileNotFoundError(f"No frame directories found under {input_dir}")

    annotation_index = build_annotation_index(annotation_det_dir)
    stats = RenderStats(total_videos=len(tasks))

    for task in tasks:
        resolved_task = resolve_video_task(task, annotation_index, annotation_mask_dir)
        if resolved_task is None:
            LOGGER.warning("No annotations found for %s (sequence_key=%s).", task.video_name, task.sequence_key)
            stats = RenderStats(
                total_videos=stats.total_videos,
                processed_videos=stats.processed_videos,
                skipped_videos=stats.skipped_videos,
                failed_videos=stats.failed_videos + 1,
                rendered_frames=stats.rendered_frames,
                rendered_videos=stats.rendered_videos,
            )
            continue

        try:
            result = process_video(
                resolved_task=resolved_task,
                output_image_root=output_image_dir,
                output_video_root=output_video_dir,
                image_ext=args.image_ext,
                class_filter=class_filter,
                background_dim_factor=args.background_dim_factor,
                foreground_gain=args.foreground_gain,
                foreground_feather_sigma=args.foreground_feather_sigma,
                outline_glow_sigma=args.outline_glow_sigma,
                box_thickness=args.box_thickness,
                frame_rate=args.frame_rate,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            LOGGER.exception("Failed to visualize %s: %s", task.video_name, exc)
            stats = RenderStats(
                total_videos=stats.total_videos,
                processed_videos=stats.processed_videos,
                skipped_videos=stats.skipped_videos,
                failed_videos=stats.failed_videos + 1,
                rendered_frames=stats.rendered_frames,
                rendered_videos=stats.rendered_videos,
            )
            continue

        if result == (0, 0) and args.skip_existing:
            stats = RenderStats(
                total_videos=stats.total_videos,
                processed_videos=stats.processed_videos,
                skipped_videos=stats.skipped_videos + 1,
                failed_videos=stats.failed_videos,
                rendered_frames=stats.rendered_frames,
                rendered_videos=stats.rendered_videos,
            )
            continue

        rendered_frames, rendered_videos = result or (0, 0)
        stats = RenderStats(
            total_videos=stats.total_videos,
            processed_videos=stats.processed_videos + 1,
            skipped_videos=stats.skipped_videos,
            failed_videos=stats.failed_videos,
            rendered_frames=stats.rendered_frames + rendered_frames,
            rendered_videos=stats.rendered_videos + rendered_videos,
        )

    summary_path = resolve_summary_path(args, output_video_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dir": str(input_dir),
        "annotation_det_dir": str(annotation_det_dir),
        "annotation_mask_dir": str(annotation_mask_dir),
        "output_image_dir": str(output_image_dir),
        "output_video_dir": str(output_video_dir),
        "frame_rate": args.frame_rate,
        "image_ext": args.image_ext,
        "class_filter": sorted(class_filter) if class_filter is not None else None,
        "background_dim_factor": args.background_dim_factor,
        "foreground_gain": args.foreground_gain,
        "foreground_feather_sigma": args.foreground_feather_sigma,
        "outline_glow_sigma": args.outline_glow_sigma,
        "stats": asdict(stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
