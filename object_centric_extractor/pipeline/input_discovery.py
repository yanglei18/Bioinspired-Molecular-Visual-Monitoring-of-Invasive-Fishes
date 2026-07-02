"""Input discovery and output layout helpers for the SAM2 pipeline."""

from __future__ import annotations

import os


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_supported_video_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTENSIONS)


def resolve_video_output_layout(source_path: str) -> tuple[str, str, str | None]:
    normalized_source = os.path.normpath(source_path)
    path_parts = normalized_source.split(os.sep)
    video_basename = os.path.basename(normalized_source)
    video_name = (
        os.path.splitext(video_basename)[0]
        if is_supported_video_file(normalized_source)
        else video_basename
    )

    fish_type = None
    for index, part in enumerate(path_parts):
        if part != "video":
            continue
        remaining_parts = path_parts[index + 1:]
        if len(remaining_parts) >= 2:
            fish_type = remaining_parts[0]
        break

    if fish_type:
        return video_name, os.path.join(fish_type, video_name), fish_type
    return video_name, video_name, None


def is_frame_directory(directory_path: str) -> bool:
    """Return True when a directory directly contains image frames."""
    if not os.path.isdir(directory_path):
        return False
    return any(
        os.path.isfile(os.path.join(directory_path, filename))
        and filename.lower().endswith(IMAGE_EXTENSIONS)
        for filename in os.listdir(directory_path)
    )


def discover_video_inputs(input_dir: str) -> list[str]:
    """Discover supported inputs under either flat or fish_type/video_name layouts."""
    discovered_inputs = []

    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)

        if is_supported_video_file(item_path):
            discovered_inputs.append(item_path)
            continue

        if not os.path.isdir(item_path):
            continue

        if is_frame_directory(item_path):
            discovered_inputs.append(item_path)
            continue

        # Support nested layout: video/<fish_type>/<video_name>
        for nested_item in sorted(os.listdir(item_path)):
            nested_path = os.path.join(item_path, nested_item)
            if is_supported_video_file(nested_path):
                discovered_inputs.append(nested_path)
                continue
            if is_frame_directory(nested_path):
                discovered_inputs.append(nested_path)

    return discovered_inputs
