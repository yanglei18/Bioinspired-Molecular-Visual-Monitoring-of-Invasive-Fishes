#!/usr/bin/env python3
"""Visualize GT boxes by prompting SAM2 image predictor to obtain masks."""

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

from tools.common import build_frame_mappings, discover_video_tasks, sort_frame_names
from utils.gt_visualization_utils import (
    build_label_index,
    build_sam2_image_predictor,
    parse_label_file,
    predict_mask_canvas,
)
from utils.visualization_utils import IMAGE_EXTENSIONS, create_video_from_images, has_completed_outputs, render_frame


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
            "Visualize GT labels from label_rectified by using GT boxes as SAM2 "
            "prompts to predict instance masks, then render images and videos."
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
        "--label_dir",
        type=Path,
        default=Path("data/robotfish-data/label_rectified"),
        help="Directory containing label_rectified *.txt files.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=Path,
        default=Path("work_dirs/extraction_output/visualization_gt_image"),
        help="Output directory for per-frame GT visualization images under <output_image_dir>/<video_name>/.",
    )
    parser.add_argument(
        "--output_video_dir",
        type=Path,
        default=Path("work_dirs/extraction_output/visualization_gt_video"),
        help="Output directory for GT visualization mp4 videos named <video_name>.mp4.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=Path,
        default=Path("./checkpoints/sam2.1_hiera_large.pt"),
        help="Path to the SAM2 checkpoint used for box-to-mask prompting.",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for SAM2 inference: auto, cuda, or cpu.",
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
        help="Brightness factor applied to non-target background regions.",
    )
    parser.add_argument(
        "--foreground_gain",
        type=float,
        default=1.35,
        help="Brightness multiplier applied to target foreground regions.",
    )
    parser.add_argument(
        "--foreground_feather_sigma",
        type=float,
        default=11.0,
        help="Gaussian blur sigma used to softly blend foreground and background.",
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
        "--show_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to render class labels near the boxes.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos whose GT visualization frames and mp4 already exist.",
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
    return {class_name.strip().lower().replace("-", "_").replace(" ", "_") for class_name in class_filter}


def resolve_summary_path(args: argparse.Namespace, output_video_dir: Path) -> Path:
    if args.summary_path is not None:
        return args.summary_path.expanduser().resolve()
    return output_video_dir / "visualization_gt_summary.json"


def process_video(
    task,
    label_file: Path,
    image_predictor,
    output_image_root: Path,
    output_video_root: Path,
    image_ext: str,
    class_filter: set[str] | None,
    background_dim_factor: float,
    foreground_gain: float,
    foreground_feather_sigma: float,
    outline_glow_sigma: float,
    box_thickness: int,
    show_labels: bool,
    frame_rate: int,
    skip_existing: bool,
) -> tuple[int, int]:
    video_name = task.video_name
    image_output_dir = output_image_root / video_name
    video_output_path = output_video_root / f"{video_name}.mp4"

    if skip_existing and has_completed_outputs(image_output_dir, video_output_path):
        LOGGER.info("Skipping %s: GT visualization outputs already exist.", video_name)
        return 0, 0

    frame_annotations = parse_label_file(label_file)
    image_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_root.mkdir(parents=True, exist_ok=True)

    frame_names = sort_frame_names(
        [path.name for path in task.video_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    original_to_padded, _ = build_frame_mappings(frame_names)

    rendered_frame_count = 0
    for frame_name in tqdm(frame_names, desc=f"Rendering GT {video_name}", leave=False):
        frame_path = task.video_dir / frame_name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            LOGGER.warning("Skipping unreadable frame %s", frame_path)
            continue

        padded_name = original_to_padded[frame_name]
        frame_key = Path(padded_name).stem
        frame_annotation = frame_annotations.get(frame_key)
        mask_canvas = predict_mask_canvas(
            image_predictor=image_predictor,
            frame_bgr=frame,
            frame_annotation=frame_annotation,
            class_filter=class_filter,
        )
        rendered = render_frame(
            frame=frame,
            frame_annotation=frame_annotation,
            mask_array=mask_canvas,
            class_filter=class_filter,
            background_dim_factor=background_dim_factor,
            foreground_gain=foreground_gain,
            feather_sigma=foreground_feather_sigma,
            outline_glow_sigma=outline_glow_sigma,
            box_thickness=box_thickness,
            show_labels=show_labels,
        )
        output_image_path = image_output_dir / f"{Path(frame_name).stem}{image_ext}"
        cv2.imwrite(str(output_image_path), rendered)
        rendered_frame_count += 1

    video_written = create_video_from_images(image_output_dir, video_output_path, frame_rate=frame_rate)
    return rendered_frame_count, int(video_written)


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = args.input_dir.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    output_image_dir = args.output_image_dir.expanduser().resolve()
    output_video_dir = args.output_video_dir.expanduser().resolve()
    sam2_checkpoint = args.sam2_checkpoint.expanduser().resolve()
    class_filter = normalize_class_filter(args.class_filter)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_dir}")

    tasks = discover_video_tasks(input_dir)
    if not tasks:
        raise FileNotFoundError(f"No frame directories found under {input_dir}")

    label_index = build_label_index(label_dir)
    if not label_index:
        raise FileNotFoundError(f"No .txt files found in {label_dir}")

    image_predictor, resolved_device = build_sam2_image_predictor(
        sam2_config=args.sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        device_name=args.device,
    )
    LOGGER.info("Using SAM2 device: %s", resolved_device)

    stats = RenderStats(total_videos=len(tasks))
    for task in tasks:
        label_file = label_index.get(task.video_name)
        if label_file is None:
            LOGGER.warning("No GT label file found for %s.", task.video_name)
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
            rendered_frames, rendered_videos = process_video(
                task=task,
                label_file=label_file,
                image_predictor=image_predictor,
                output_image_root=output_image_dir,
                output_video_root=output_video_dir,
                image_ext=args.image_ext,
                class_filter=class_filter,
                background_dim_factor=args.background_dim_factor,
                foreground_gain=args.foreground_gain,
                foreground_feather_sigma=args.foreground_feather_sigma,
                outline_glow_sigma=args.outline_glow_sigma,
                box_thickness=args.box_thickness,
                show_labels=args.show_labels,
                frame_rate=args.frame_rate,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            LOGGER.exception("Failed to visualize GT for %s: %s", task.video_name, exc)
            stats = RenderStats(
                total_videos=stats.total_videos,
                processed_videos=stats.processed_videos,
                skipped_videos=stats.skipped_videos,
                failed_videos=stats.failed_videos + 1,
                rendered_frames=stats.rendered_frames,
                rendered_videos=stats.rendered_videos,
            )
            continue

        if rendered_frames == 0 and rendered_videos == 0 and args.skip_existing:
            stats = RenderStats(
                total_videos=stats.total_videos,
                processed_videos=stats.processed_videos,
                skipped_videos=stats.skipped_videos + 1,
                failed_videos=stats.failed_videos,
                rendered_frames=stats.rendered_frames,
                rendered_videos=stats.rendered_videos,
            )
            continue

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
        "label_dir": str(label_dir),
        "output_image_dir": str(output_image_dir),
        "output_video_dir": str(output_video_dir),
        "sam2_checkpoint": str(sam2_checkpoint),
        "sam2_config": args.sam2_config,
        "device": resolved_device,
        "frame_rate": args.frame_rate,
        "image_ext": args.image_ext,
        "class_filter": sorted(class_filter) if class_filter is not None else None,
        "background_dim_factor": args.background_dim_factor,
        "foreground_gain": args.foreground_gain,
        "foreground_feather_sigma": args.foreground_feather_sigma,
        "outline_glow_sigma": args.outline_glow_sigma,
        "show_labels": args.show_labels,
        "stats": asdict(stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
