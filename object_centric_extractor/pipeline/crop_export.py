"""Instance crop export helpers for SAM2 detector outputs."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    from utils.annotation_io import load_sequence_annotations
except ModuleNotFoundError:
    from object_centric_extractor.utils.annotation_io import load_sequence_annotations


def create_black_background_crop(
    cropped_img: np.ndarray,
    instance_mask: np.ndarray,
) -> np.ndarray:
    if len(cropped_img.shape) == 2:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    black_background = np.zeros_like(cropped_img)
    mask_bool = instance_mask.astype(bool)
    if np.any(mask_bool):
        black_background[mask_bool] = cropped_img[mask_bool]
    return black_background


def build_empty_frame_annotation(
    mask_name: str,
    mask_height: int,
    mask_width: int,
    promote_type: str = "mask",
) -> dict[str, Any]:
    return {
        "mask_name": mask_name,
        "mask_height": mask_height,
        "mask_width": mask_width,
        "promote_type": promote_type,
        "labels": {},
    }


def is_valid_box(box: tuple[int, int, int, int]) -> bool:
    """Check if a bounding box is valid with positive width and height."""
    x1, y1, x2, y2 = box
    return (x1 < x2 and y1 < y2) and not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0)


def calculate_instance_sizes(
    json_dir: str,
    padding: int = 10,
    scale_factor: float = 1.0,
    percentile: int = 80,
    sequence_key: str | None = None,
) -> dict[int, tuple[int, int, int, int]]:
    """Pre-scan JSON files to determine optimal window size for each instance ID."""
    instance_boxes: defaultdict[int, list[tuple[int, int, int, int, int, int, int, int]]] = defaultdict(list)

    frame_annotations = load_sequence_annotations(json_dir, sequence_key=sequence_key)

    print("Pre-scanning bounding boxes to determine optimal window sizes...")
    for frame_id in tqdm(sorted(frame_annotations.keys()), desc="Scanning JSON files"):
        detection_data = frame_annotations[frame_id]
        try:
            if "labels" not in detection_data:
                continue

            for obj_id, obj_data in detection_data["labels"].items():
                instance_id = obj_data.get("instance_id", int(obj_id))

                x1 = int(obj_data.get("x1", 0))
                y1 = int(obj_data.get("y1", 0))
                x2 = int(obj_data.get("x2", 0))
                y2 = int(obj_data.get("y2", 0))

                if is_valid_box((x1, y1, x2, y2)):
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    instance_boxes[instance_id].append(
                        (x1, y1, x2, y2, center_x, center_y, width, height)
                    )
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")

    instance_sizes = {}

    small_obj_threshold = 150
    large_obj_threshold = 500
    max_padding = padding

    for instance_id, boxes in instance_boxes.items():
        if not boxes:
            continue

        widths = [box[6] for box in boxes]
        heights = [box[7] for box in boxes]

        if len(widths) >= 5:
            width_percentile = np.percentile(widths, percentile)
            height_percentile = np.percentile(heights, percentile)
        else:
            width_percentile = max(widths)
            height_percentile = max(heights)

        max_width = max(widths)
        max_height = max(heights)

        obj_size = max(width_percentile, height_percentile)

        if obj_size <= small_obj_threshold:
            dynamic_padding = max_padding
        elif obj_size >= large_obj_threshold:
            dynamic_padding = 0
        else:
            ratio = (obj_size - small_obj_threshold) / (large_obj_threshold - small_obj_threshold)
            dynamic_padding = int(max_padding * (1 - ratio))

        window_width = int(width_percentile * scale_factor) + dynamic_padding * 2
        window_height = int(height_percentile * scale_factor) + dynamic_padding * 2

        min_width_threshold = max_width * 0.8
        min_height_threshold = max_height * 0.8

        window_width = max(window_width, int(min_width_threshold) + dynamic_padding * 2)
        window_height = max(window_height, int(min_height_threshold) + dynamic_padding * 2)

        crop_size = max(window_width, window_height)
        window_width = crop_size
        window_height = crop_size

        window_width = (window_width + 1) // 2 * 2
        window_height = (window_height + 1) // 2 * 2

        instance_sizes[instance_id] = (window_width, window_height, max_width, max_height)

    return instance_sizes


def get_centered_crop_coordinates(
    image_shape: tuple[int, ...],
    center_x: int,
    center_y: int,
    crop_width: int,
    crop_height: int,
    obj_width: int,
    obj_height: int,
) -> tuple[int, int, int, int]:
    """Calculate crop coordinates ensuring the crop is a square with the object centered."""
    del obj_width, obj_height
    img_height, img_width = image_shape[:2]

    crop_size = max(crop_width, crop_height)

    x1 = int(center_x - crop_size / 2)
    y1 = int(center_y - crop_size / 2)

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1 + crop_size > img_width:
        x1 = max(0, img_width - crop_size)
    if y1 + crop_size > img_height:
        y1 = max(0, img_height - crop_size)

    x2 = min(img_width, x1 + crop_size)
    y2 = min(img_height, y1 + crop_size)

    if x2 - x1 < crop_size and x1 > 0:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size and y1 > 0:
        y1 = max(0, y2 - crop_size)

    return int(x1), int(y1), int(x2), int(y2)


def process_frame(
    frame: np.ndarray,
    frame_annotation: dict[str, Any],
    frame_id: str,
    mask_path: str,
    webp_output_path: str,
    video_name: str,
    frame_idx: int,
    instance_sizes: dict[int, tuple[int, int, int, int]],
    min_size: int,
    padding: int,
    fixed_window: bool,
    class_filter: list[str] | None,
    valid_instance_ids: list[int] | None = None,
) -> int:
    """Process a single frame and extract instance crops."""
    instances_processed = 0

    mask = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask = np.load(mask_path)
        except Exception as e:
            print(f"  Error loading mask file {mask_path}: {str(e)}")

    try:
        detection_data = frame_annotation

        if "labels" in detection_data:
            for obj_id, obj_data in detection_data["labels"].items():
                class_name = obj_data.get("class_name", "unknown")
                instance_id = obj_data.get("instance_id", int(obj_id))
                instance_id_str = str(instance_id)

                if valid_instance_ids is not None and instance_id not in valid_instance_ids:
                    continue

                if class_filter is not None and class_name not in class_filter:
                    continue

                x1 = int(obj_data.get("x1", 0))
                y1 = int(obj_data.get("y1", 0))
                x2 = int(obj_data.get("x2", 0))
                y2 = int(obj_data.get("y2", 0))

                if not is_valid_box((x1, y1, x2, y2)):
                    continue

                obj_width = x2 - x1
                obj_height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if obj_width < min_size or obj_height < min_size:
                    continue

                img_height, img_width = frame.shape[:2]

                if fixed_window and instance_id in instance_sizes:
                    crop_width, crop_height, max_obj_width, max_obj_height = instance_sizes[instance_id]

                    crop_size = max(crop_width, crop_height)
                    crop_width = crop_size
                    crop_height = crop_size

                    x1, y1, x2, y2 = get_centered_crop_coordinates(
                        frame.shape,
                        center_x,
                        center_y,
                        crop_width,
                        crop_height,
                        obj_width,
                        obj_height,
                    )
                else:
                    scale = 1.0
                    crop_size = max(
                        int(obj_width * scale) + (padding * 2),
                        int(obj_height * scale) + (padding * 2),
                    )

                    crop_size = max(crop_size, padding * 4)
                    crop_size = (crop_size + 1) // 2 * 2

                    x1, y1, x2, y2 = get_centered_crop_coordinates(
                        frame.shape,
                        center_x,
                        center_y,
                        crop_size,
                        crop_size,
                        obj_width,
                        obj_height,
                    )

                if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= img_width or y1 >= img_height:
                    continue

                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue

                if mask is not None:
                    try:
                        if mask.shape[:2] == frame.shape[:2]:
                            cropped_mask = mask[y1:y2, x1:x2]

                            instance_mask = None

                            if cropped_mask.size > 0:
                                if np.issubdtype(mask.dtype, np.integer):
                                    instance_mask = (cropped_mask == instance_id).astype(np.uint8)
                                elif mask.dtype == np.bool_ or (
                                    mask.dtype == np.uint8 and np.max(mask) <= 1
                                ):
                                    instance_mask = cropped_mask.astype(np.uint8)
                                elif len(mask.shape) == 3 and mask.shape[2] > 1:
                                    for channel in range(mask.shape[2]):
                                        if np.any(cropped_mask[:, :, channel]):
                                            instance_mask = cropped_mask[:, :, channel].astype(np.uint8)
                                            break

                            if instance_mask is not None and instance_mask.size > 0:
                                original_img = cropped_img.copy()

                                dilated_mask = instance_mask.copy()
                                fish_type = (
                                    mask_path.split("/")[4]
                                    if "/" in mask_path
                                    else mask_path.split(os.sep)[4]
                                )
                                if fish_type in ["guppy", "mosquitofish"]:
                                    kernel_size = 20
                                else:
                                    kernel_size = 1
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=1)

                                black_background_img = create_black_background_crop(
                                    original_img,
                                    dilated_mask,
                                )
                                bg_removed_filename = f"{video_name}_{instance_id_str}_{frame_id}.webp"
                                bg_removed_path = os.path.join(webp_output_path, bg_removed_filename)
                                cv2.imwrite(bg_removed_path, black_background_img)
                                instances_processed += 1
                    except Exception as e:
                        print(f"  Error applying mask for instance {instance_id}: {str(e)}")

    except Exception as e:
        print(f"  Error processing frame {frame_idx}: {str(e)}")

    return instances_processed


def count_instance_occurrences(
    det_output_path: str,
    min_frames: int = 100,
    sequence_key: str | None = None,
) -> tuple[list[int], defaultdict[int, int]]:
    """Count instance occurrences and return instances present for at least min_frames."""
    instance_counts: defaultdict[int, int] = defaultdict(int)
    frame_annotations = load_sequence_annotations(det_output_path, sequence_key=sequence_key)

    print(f"Counting instance occurrences in {len(frame_annotations)} frame annotations...")
    for frame_id in tqdm(sorted(frame_annotations.keys()), desc="Analyzing tracking length"):
        detection_data = frame_annotations[frame_id]
        try:
            if "labels" in detection_data:
                for obj_id, obj_data in detection_data["labels"].items():
                    instance_id = obj_data.get("instance_id", int(obj_id))
                    instance_counts[instance_id] += 1
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")

    valid_instance_ids = [
        instance_id
        for instance_id, count in instance_counts.items()
        if count >= min_frames
    ]
    filtered_count = len(instance_counts) - len(valid_instance_ids)

    print(f"Found {len(instance_counts)} unique instances.")
    print(f"Keeping {len(valid_instance_ids)} instances with >= {min_frames} frames.")
    print(f"Filtered out {filtered_count} instances with < {min_frames} frames.")

    if valid_instance_ids:
        print("\nValid instances and their frame counts:")
        for instance_id in valid_instance_ids:
            print(f"  Instance {instance_id}: {instance_counts[instance_id]} frames")

    return valid_instance_ids, instance_counts
