"""Instance video export helpers for cropped SAM2 instances."""

from __future__ import annotations

import os
import re
from collections import defaultdict

import cv2
import numpy as np


def collect_instance_webp_groups(
    webp_output_dir: str,
    video_name: str,
) -> defaultdict[str, list[tuple[int, str]]]:
    grouped_files: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    if not os.path.isdir(webp_output_dir):
        return grouped_files

    pattern = re.compile(rf"^{re.escape(video_name)}_(\d+)_(\d+)\.webp$", re.IGNORECASE)
    for filename in sorted(os.listdir(webp_output_dir)):
        match = pattern.match(filename)
        if not match:
            continue
        instance_id = match.group(1)
        frame_id = int(match.group(2))
        grouped_files[instance_id].append((frame_id, os.path.join(webp_output_dir, filename)))
    return grouped_files


def create_instance_video(
    image_paths: list[str],
    output_video_path: str,
    frame_rate: int = 15,
) -> int:
    valid_images = []
    max_height = 0
    max_width = 0

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Cannot read cropped instance image {image_path}, skipping")
            continue
        height, width = image.shape[:2]
        max_height = max(max_height, height)
        max_width = max(max_width, width)
        valid_images.append(image_path)

    if not valid_images or max_height == 0 or max_width == 0:
        return 0

    if max_width % 2 != 0:
        max_width += 1
    if max_height % 2 != 0:
        max_height += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (max_width, max_height))
    written_frames = 0

    for image_path in valid_images:
        image = cv2.imread(image_path)
        if image is None:
            continue
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        height, width = image.shape[:2]
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        y_offset = max((max_height - height) // 2, 0)
        x_offset = max((max_width - width) // 2, 0)
        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = image
        video_writer.write(canvas)
        written_frames += 1

    video_writer.release()
    return written_frames


def export_instance_mp4s(
    webp_output_dir: str,
    mp4_output_dir: str,
    video_name: str,
    frame_rate: int = 15,
) -> int:
    os.makedirs(mp4_output_dir, exist_ok=True)
    instance_groups = collect_instance_webp_groups(webp_output_dir, video_name)

    exported_count = 0
    for instance_id, frame_entries in instance_groups.items():
        ordered_image_paths = [image_path for _, image_path in sorted(frame_entries)]
        output_video_path = os.path.join(mp4_output_dir, f"{video_name}_{instance_id}.mp4")
        written_frames = create_instance_video(
            ordered_image_paths,
            output_video_path,
            frame_rate=frame_rate,
        )
        if written_frames > 0:
            exported_count += 1

    return exported_count
