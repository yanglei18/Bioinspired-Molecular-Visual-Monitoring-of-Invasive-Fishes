"""Video frame extraction helpers for the SAM2 pipeline."""

from __future__ import annotations

import os
import shutil

import cv2


def build_temp_frame_dir(input_root_dir: str, video_file_path: str) -> tuple[str, str]:
    temp_root_dir = f"{os.path.normpath(input_root_dir)}_image"
    relative_video_path = os.path.relpath(video_file_path, input_root_dir)
    relative_video_stem = os.path.splitext(relative_video_path)[0]
    temp_frame_dir = os.path.join(temp_root_dir, relative_video_stem)
    return temp_root_dir, temp_frame_dir


def cleanup_empty_parent_dirs(start_dir: str, stop_dir: str) -> None:
    current_dir = os.path.normpath(start_dir)
    stop_dir = os.path.normpath(stop_dir)
    while current_dir.startswith(stop_dir):
        if current_dir == stop_dir:
            if os.path.isdir(current_dir) and not os.listdir(current_dir):
                os.rmdir(current_dir)
            break
        if not os.path.isdir(current_dir) or os.listdir(current_dir):
            break
        os.rmdir(current_dir)
        current_dir = os.path.dirname(current_dir)


def extract_video_to_temp_frames(
    video_file_path: str,
    input_root_dir: str,
) -> tuple[str, str, int]:
    temp_root_dir, temp_frame_dir = build_temp_frame_dir(input_root_dir, video_file_path)
    if os.path.isdir(temp_frame_dir):
        shutil.rmtree(temp_frame_dir, ignore_errors=True)
    os.makedirs(temp_frame_dir, exist_ok=True)

    capture = cv2.VideoCapture(video_file_path)
    if not capture.isOpened():
        shutil.rmtree(temp_frame_dir, ignore_errors=True)
        raise RuntimeError(f"Could not open video file: {video_file_path}")

    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_path = os.path.join(temp_frame_dir, f"{frame_count:06d}.jpg")
            if not cv2.imwrite(frame_path, frame):
                raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
            frame_count += 1
    finally:
        capture.release()

    if frame_count == 0:
        shutil.rmtree(temp_frame_dir, ignore_errors=True)
        raise RuntimeError(f"No frames could be extracted from video: {video_file_path}")

    return temp_root_dir, temp_frame_dir, frame_count
