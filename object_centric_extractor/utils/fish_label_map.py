"""Shared fish label mapping utilities for SAM2 evaluation."""

from __future__ import annotations

from typing import Iterable


LABEL_ID_TO_NAME = {
    "0": "redeye_barbel",
    "1": "mud_carp",
    "2": "serrated_barb",
    "3": "carp",
    "4": "black_carp",
    "5": "schizothorax_fish",
    "6": "chinese_paddlefish",
    "7": "wuchang_bream",
    "8": "chinese_sucker",
    "9": "chinese_labeo",
}

FINE_CLASSES = tuple(LABEL_ID_TO_NAME.values())


def normalize_fish_class_name(class_name: str | None) -> str | None:
    if class_name is None:
        return None
    normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized.rstrip(".")


def get_fish_class_name(label_id: str | int) -> str | None:
    return LABEL_ID_TO_NAME.get(str(label_id))


def is_supported_fine_class(class_name: str | None) -> bool:
    normalized = normalize_fish_class_name(class_name)
    return normalized in LABEL_ID_TO_NAME.values()


def filter_supported_fine_classes(class_names: Iterable[str]) -> list[str]:
    filtered = []
    for class_name in class_names:
        normalized = normalize_fish_class_name(class_name)
        if normalized in LABEL_ID_TO_NAME.values():
            filtered.append(normalized)
    return filtered
