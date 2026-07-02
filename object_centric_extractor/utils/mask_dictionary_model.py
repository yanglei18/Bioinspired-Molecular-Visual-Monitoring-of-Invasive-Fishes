from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _to_numpy_array(value: Any) -> np.ndarray:
    """Convert tensor-like values to numpy without importing torch at module load."""
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _to_2d_bool_mask(mask: Any) -> np.ndarray:
    mask_array = _to_numpy_array(mask)
    while mask_array.ndim > 2 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask_array.shape}.")
    return mask_array.astype(bool, copy=False)


def _mask_nonzero_count(mask: Any) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(_to_numpy_array(mask)))


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class MaskDictionaryModel:
    mask_name: str = ""
    mask_height: int = 1080
    mask_width: int = 1920
    promote_type: str = "mask"
    labels: dict[int, ObjectInfo] = field(default_factory=dict)

    def add_new_frame_annotation(
        self,
        mask_list: Any,
        box_list: Any,
        label_list: list[str],
        background_value: int = 0,
    ) -> None:
        mask_height, mask_width = int(mask_list.shape[-2]), int(mask_list.shape[-1])
        anno_2d: dict[int, ObjectInfo] = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_index = background_value + idx + 1

            if int(mask.shape[-2]) != mask_height or int(mask.shape[-1]) != mask_width:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            new_annotation = ObjectInfo(
                instance_id=final_index,
                mask=mask,
                class_name=label,
                x1=_to_python_scalar(box[0]),
                y1=_to_python_scalar(box[1]),
                x2=_to_python_scalar(box[2]),
                y2=_to_python_scalar(box[3]),
            )
            anno_2d[final_index] = new_annotation

        self.mask_height = mask_height
        self.mask_width = mask_width
        self.labels = anno_2d

    def update_masks(
        self,
        tracking_annotation_dict: MaskDictionaryModel,
        iou_threshold: float = 0.8,
        objects_count: int = 0,
    ) -> int:
        updated_masks: dict[int, ObjectInfo] = {}

        for _, seg_mask in self.labels.items():
            flag = 0 
            new_mask_copy = ObjectInfo()
            if _mask_nonzero_count(seg_mask.mask) == 0:
                continue
            
            for _, object_info in tracking_annotation_dict.labels.items():
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break
                
            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count

    def get_target_class_name(self, instance_id: int) -> str:
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id: int) -> float:
        return self.labels[instance_id].logit
    
    @staticmethod
    def calculate_iou(mask1: Any, mask2: Any) -> float:
        mask1_array = _to_2d_bool_mask(mask1)
        mask2_array = _to_2d_bool_mask(mask2)
        if mask1_array.shape != mask2_array.shape:
            raise ValueError(
                f"Mask shapes must match to calculate IoU: "
                f"{mask1_array.shape} != {mask2_array.shape}."
            )

        intersection = np.logical_and(mask1_array, mask2_array).sum()
        union = np.logical_or(mask1_array, mask2_array).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)


    def save_empty_mask_and_json(
        self,
        mask_data_dir: str,
        json_data_dir: str,
        image_name_list: list[str] | None = None,
    ) -> None:
        mask_img = np.zeros((self.mask_height, self.mask_width), dtype=np.uint16)
        if image_name_list:
            for image_base_name in image_name_list:
                image_base_name = image_base_name.split(".")[0]+".npy"
                mask_name = "mask_"+image_base_name
                np.save(os.path.join(mask_data_dir, mask_name), mask_img)

                json_data_path = os.path.join(json_data_dir, mask_name.replace(".npy", ".json"))
                print("save_empty_mask_and_json", json_data_path)
                self.to_json(json_data_path)
        else:
            np.save(os.path.join(mask_data_dir, self.mask_name), mask_img)
            json_data_path = os.path.join(json_data_dir, self.mask_name.replace(".npy", ".json"))
            print("save_empty_mask_and_json", json_data_path)
            self.to_json(json_data_path)


    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file: str) -> None:
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def from_dict(self, data: dict[str, Any]) -> MaskDictionaryModel:
        self.mask_name = data["mask_name"]
        self.mask_height = data["mask_height"]
        self.mask_width = data["mask_width"]
        self.promote_type = data["promote_type"]
        self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self
            
    def from_json(self, json_file: str) -> MaskDictionaryModel:
        with open(json_file, "r") as f:
            data = json.load(f)
        return self.from_dict(data)


@dataclass
class ObjectInfo:
    instance_id: int = 0
    mask: Any = None
    class_name: str = ""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    logit: float = 0.0

    def get_mask(self) -> Any:
        return self.mask
    
    def get_id(self) -> int:
        return self.instance_id

    def update_box(self) -> list[int] | None:
        nonzero_indices = np.argwhere(_to_2d_bool_mask(self.mask))
        if nonzero_indices.shape[0] == 0:
            return []

        y_min, x_min = nonzero_indices.min(axis=0)
        y_max, x_max = nonzero_indices.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
        return None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": _to_python_scalar(self.instance_id),
            "class_name": self.class_name,
            "x1": _to_python_scalar(self.x1),
            "y1": _to_python_scalar(self.y1),
            "x2": _to_python_scalar(self.x2),
            "y2": _to_python_scalar(self.y2),
            "logit": _to_python_scalar(self.logit),
        }
