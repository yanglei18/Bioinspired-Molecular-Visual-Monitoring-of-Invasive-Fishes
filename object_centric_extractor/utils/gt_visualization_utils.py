"""Helpers for GT visualization with label_rectified and SAM2 box prompting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .fish_label_map import get_fish_class_name


def parse_label_file(label_file: Path) -> dict[str, dict]:
    frame_annotations: dict[str, dict] = {}
    with label_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 8:
                continue

            class_name = get_fish_class_name(parts[2])
            if class_name is None:
                continue

            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[3])
                y = float(parts[4])
                width = float(parts[5])
                height = float(parts[6])
            except ValueError:
                continue

            frame_key = f"{frame_id:06d}"
            frame_payload = frame_annotations.setdefault(frame_key, {"labels": {}})
            frame_payload["labels"][str(track_id)] = {
                "instance_id": track_id,
                "class_name": class_name,
                "x1": int(round(x)),
                "y1": int(round(y)),
                "x2": int(round(x + width)),
                "y2": int(round(y + height)),
            }
    return frame_annotations


def build_label_index(label_dir: Path) -> dict[str, Path]:
    return {
        label_file.stem: label_file
        for label_file in sorted(label_dir.glob("*.txt"))
    }


def build_sam2_image_predictor(
    sam2_config: str,
    sam2_checkpoint: Path,
    device_name: str,
) -> tuple[Any, str]:
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not available in the current environment. "
            "Please install sam2 and run this script inside the SAM2 environment."
        ) from exc

    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")

    sam2_image_model = build_sam2(sam2_config, str(sam2_checkpoint), device=device)
    predictor = SAM2ImagePredictor(sam2_image_model)
    return predictor, device


def predict_mask_canvas(
    image_predictor: Any,
    frame_bgr: np.ndarray,
    frame_annotation: dict | None,
    class_filter: set[str] | None,
) -> np.ndarray | None:
    if frame_annotation is None:
        return None

    objects: list[tuple[int, tuple[int, int, int, int]]] = []
    for object_id, object_data in sorted(
        frame_annotation.get("labels", {}).items(),
        key=lambda item: int(item[1].get("instance_id", item[0])),
    ):
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
        objects.append((instance_id, (x1, y1, x2, y2)))

    if not objects:
        return None

    image_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    input_boxes = np.array([box for _, box in objects], dtype=np.float32)
    masks, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 2:
        masks = masks[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    mask_canvas = np.zeros(frame_bgr.shape[:2], dtype=np.uint16)
    for mask, (instance_id, _) in zip(masks, objects):
        binary_mask = mask > 0
        if np.any(binary_mask):
            mask_canvas[binary_mask] = instance_id
    return mask_canvas
