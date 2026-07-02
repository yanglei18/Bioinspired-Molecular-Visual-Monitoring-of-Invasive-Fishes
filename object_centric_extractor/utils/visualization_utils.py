"""Rendering helpers for SAM2 visualization export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# OpenCV uses BGR channel order.
DEFAULT_BOX_COLOR = (0, 191, 255)  # hex #ffbf00 (yellow-orange)
CLASS_BOX_COLORS = {
    "carp": (0, 0, 255),  # red
}
LABEL_TEXT_COLOR = (220, 220, 220)
LABEL_BG_COLOR = (22, 22, 22)
CONTOUR_COLOR = (255, 225, 140)


def format_class_label(class_name: str) -> str:
    normalized = class_name.strip().replace("_", " ")
    return normalized.title() if normalized else "Fish"


def get_box_color(class_name: str) -> tuple[int, int, int]:
    normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
    return CLASS_BOX_COLORS.get(normalized, DEFAULT_BOX_COLOR)


def resolve_mask_path(mask_sequence_dir: Path, frame_key: str) -> Path | None:
    direct_path = mask_sequence_dir / f"{frame_key}.npy"
    if direct_path.is_file():
        return direct_path
    prefixed_path = mask_sequence_dir / f"mask_{frame_key}.npy"
    if prefixed_path.is_file():
        return prefixed_path
    return None


def create_foreground_mask(
    image_shape: tuple[int, int, int],
    objects: list[dict],
) -> np.ndarray:
    foreground_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for obj in objects:
        mask = obj.get("mask")
        if mask is not None and np.any(mask):
            foreground_mask[mask] = 255
            continue

        x1, y1, x2, y2 = obj["box"]
        foreground_mask[y1:y2, x1:x2] = 255
    return foreground_mask


def apply_foreground_focus(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    background_dim_factor: float,
    foreground_gain: float,
    feather_sigma: float,
) -> np.ndarray:
    original = image.astype(np.float32)
    dimmed = original * background_dim_factor
    if not np.any(foreground_mask):
        return dimmed.astype(np.uint8)

    boosted = np.clip(original * foreground_gain, 0, 255)

    feathered_mask = cv2.GaussianBlur(
        foreground_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma,
    )
    feathered_mask[foreground_mask > 0] = 1.0
    alpha = feathered_mask[..., None]
    rendered = dimmed * (1.0 - alpha) + boosted * alpha
    return np.clip(rendered, 0, 255).astype(np.uint8)


def add_mask_outline(
    image: np.ndarray,
    binary_mask: np.ndarray,
    contour_color: tuple[int, int, int],
    glow_sigma: float,
) -> np.ndarray:
    if not np.any(binary_mask):
        return image

    mask_u8 = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    overlay = np.zeros_like(image)
    cv2.drawContours(overlay, contours, -1, contour_color, thickness=2)
    glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=glow_sigma, sigmaY=glow_sigma)
    rendered = cv2.addWeighted(image, 1.0, glow, 0.45, 0)
    cv2.drawContours(rendered, contours, -1, contour_color, thickness=2)
    return rendered


def draw_class_label(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    class_name: str,
) -> None:
    label = format_class_label(class_name)
    text_scale = 0.78
    text_thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
    )

    image_height, image_width = image.shape[:2]
    x1, y1, x2, _ = box
    horizontal_padding = 14
    vertical_padding = 12
    label_width = text_width + horizontal_padding
    label_height = text_height + baseline + vertical_padding

    # Anchor the label near the top-right corner of the box while letting the
    # background width follow the rendered text width instead of the box width.
    label_x2 = min(image_width - 2, x2 - 2)
    label_x1 = max(0, label_x2 - label_width)
    if label_x2 - label_x1 < label_width:
        label_x1 = 0
        label_x2 = min(image_width, label_width)

    label_y1 = max(0, y1 + 2)
    label_y2 = min(image_height, label_y1 + label_height)
    if label_y2 - label_y1 < label_height:
        label_y1 = max(0, label_y2 - label_height)

    overlay = image.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), LABEL_BG_COLOR, thickness=-1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, dst=image)
    cv2.putText(
        image,
        label,
        (label_x1 + 7, label_y2 - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        LABEL_TEXT_COLOR,
        text_thickness,
        lineType=cv2.LINE_AA,
    )


def render_frame(
    frame: np.ndarray,
    frame_annotation: dict | None,
    mask_array: np.ndarray | None,
    class_filter: set[str] | None,
    background_dim_factor: float,
    foreground_gain: float,
    feather_sigma: float,
    outline_glow_sigma: float,
    box_thickness: int,
    show_labels: bool = True,
) -> np.ndarray:
    if frame_annotation is None:
        return frame

    labels = frame_annotation.get("labels", {})
    objects = []
    sorted_items = sorted(
        labels.items(),
        key=lambda item: int(item[1].get("instance_id", item[0])),
    )

    for object_id, object_data in sorted_items:
        class_name = str(object_data.get("class_name", "fish")).strip().lower()
        if class_filter is not None and class_name not in class_filter:
            continue

        instance_id = int(object_data.get("instance_id", object_id))
        x1 = int(object_data.get("x1", 0))
        y1 = int(object_data.get("y1", 0))
        x2 = int(object_data.get("x2", 0))
        y2 = int(object_data.get("y2", 0))
        if x1 >= x2 or y1 >= y2:
            continue

        instance_mask = None
        if mask_array is not None:
            instance_mask = mask_array == instance_id
            if instance_mask is not None and not np.any(instance_mask):
                instance_mask = None

        objects.append(
            {
                "instance_id": instance_id,
                "class_name": class_name,
                "box": (x1, y1, x2, y2),
                "mask": instance_mask,
            }
        )

    if not objects:
        return frame

    foreground_mask = create_foreground_mask(frame.shape, objects)
    rendered = apply_foreground_focus(
        image=frame,
        foreground_mask=foreground_mask,
        background_dim_factor=background_dim_factor,
        foreground_gain=foreground_gain,
        feather_sigma=feather_sigma,
    )

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        instance_mask = obj["mask"]
        if instance_mask is not None:
            rendered = add_mask_outline(
                image=rendered,
                binary_mask=instance_mask,
                contour_color=CONTOUR_COLOR,
                glow_sigma=outline_glow_sigma,
            )

        cv2.rectangle(
            rendered,
            (x1, y1),
            (x2, y2),
            get_box_color(obj["class_name"]),
            thickness=box_thickness,
            lineType=cv2.LINE_AA,
        )
        if show_labels:
            draw_class_label(rendered, (x1, y1, x2, y2), obj["class_name"])

    return rendered


def create_video_from_images(image_dir: Path, output_path: Path, frame_rate: int) -> bool:
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        return False

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        return False

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        return False

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()
    return True


def has_completed_outputs(image_output_dir: Path, video_output_path: Path) -> bool:
    has_images = image_output_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in image_output_dir.iterdir()
    )
    return has_images and video_output_path.is_file()
