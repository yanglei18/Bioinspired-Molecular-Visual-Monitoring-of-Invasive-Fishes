"""Tracking stage for the Grounded-SAM2 detector pipeline."""

from __future__ import annotations

import copy
import os

import numpy as np
from PIL import Image

try:
    from pipeline.config import DetectionConfig
    from pipeline.crop_export import build_empty_frame_annotation
    from pipeline.runtime import PipelineRuntime
    from utils.annotation_io import write_sequence_annotations
    from utils.video_utils import create_video_from_images
except ModuleNotFoundError:
    from object_centric_extractor.pipeline.config import DetectionConfig
    from object_centric_extractor.pipeline.crop_export import build_empty_frame_annotation
    from object_centric_extractor.pipeline.runtime import PipelineRuntime
    from object_centric_extractor.utils.annotation_io import write_sequence_annotations
    from object_centric_extractor.utils.video_utils import create_video_from_images


def run_tracking_stage(
    video_path: str,
    frame_names: list[str],
    original_to_padded: dict[str, str],
    det_output_path: str,
    det_sequence_key: str,
    mask_output_path: str,
    masked_image_output_path: str | None,
    masked_video_output_path: str | None,
    video_name: str,
    detection_config: DetectionConfig,
    runtime: PipelineRuntime,
    enable_visualization: bool,
) -> dict[str, dict]:
    try:
        from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
    except ModuleNotFoundError:
        from object_centric_extractor.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

    sequence_annotations: dict[str, dict] = {}
    torch_module = runtime.torch
    inference_state = runtime.video_predictor.init_state(video_path=video_path)
    step = detection_config.step

    sam2_masks = MaskDictionaryModel()
    prompt_type_for_video = "mask"
    objects_count = 0
    frame_object_count = {}

    print(f"Processing {video_name} - Total frames: {len(frame_names)}")
    for start_frame_idx in range(0, len(frame_names), step):
        print(f"  Frame {start_frame_idx}/{len(frame_names)}")
        img_path = os.path.join(video_path, frame_names[start_frame_idx])
        image = Image.open(img_path).convert("RGB")
        padded_base_name = original_to_padded[frame_names[start_frame_idx]].split(".")[0]
        mask_dict = MaskDictionaryModel(
            promote_type=prompt_type_for_video,
            mask_name=f"{padded_base_name}.npy",
        )

        inputs = runtime.processor(
            images=image,
            text=detection_config.text_prompt,
            return_tensors="pt",
        ).to(runtime.device)
        with torch_module.no_grad():
            outputs = runtime.grounding_model(**inputs)

        results = runtime.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]],
        )

        runtime.image_predictor.set_image(np.array(image.convert("RGB")))
        all_scores = results[0]["scores"]
        all_boxes = results[0]["boxes"]
        all_labels = results[0]["labels"]

        print("img_path: ", img_path)

        filtered_indices = all_scores >= detection_config.box_threshold
        input_boxes = all_boxes[filtered_indices]
        dino_scores = all_scores[filtered_indices]
        objects = [all_labels[i] for i in torch_module.where(filtered_indices)[0]]

        if input_boxes.shape[0] > 0:
            widths = input_boxes[:, 2] - input_boxes[:, 0]
            heights = input_boxes[:, 3] - input_boxes[:, 1]
            min_edges = torch_module.min(widths, heights)
            edge_mask = min_edges >= detection_config.min_edge_threshold

            input_boxes = input_boxes[edge_mask]
            dino_scores = dino_scores[edge_mask]
            objects = [obj for i, obj in enumerate(objects) if edge_mask[i]]

        if input_boxes.shape[0] != 0:
            masks, scores, logits = runtime.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(
                    mask_list=torch_module.tensor(masks).to(runtime.device),
                    box_list=torch_module.tensor(input_boxes),
                    label_list=objects,
                )
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")
        else:
            print(f"No object detected in frame {frame_names[start_frame_idx]}")
            mask_dict = sam2_masks

        objects_count = mask_dict.update_masks(
            tracking_annotation_dict=sam2_masks,
            iou_threshold=detection_config.iou_threshold,
            objects_count=objects_count,
        )
        frame_object_count[start_frame_idx] = objects_count
        if len(mask_dict.labels) == 0:
            image_height, image_width = image.size[1], image.size[0]
            for empty_frame_name in frame_names[start_frame_idx:start_frame_idx + step]:
                padded_base_name = original_to_padded[empty_frame_name].split(".")[0]
                mask_name = f"{padded_base_name}.npy"
                empty_mask = np.zeros((image_height, image_width), dtype=np.uint16)
                np.save(os.path.join(mask_output_path, mask_name), empty_mask)
                sequence_annotations[padded_base_name] = build_empty_frame_annotation(
                    mask_name=mask_name,
                    mask_height=image_height,
                    mask_width=image_width,
                    promote_type=prompt_type_for_video,
                )
            print(f"No object detected in frame {start_frame_idx}")
            continue

        runtime.video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            runtime.video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )

        object_id_to_index = {}
        for idx, (obj_id, obj_info) in enumerate(mask_dict.labels.items()):
            object_id_to_index[obj_id] = idx

        object_id_to_score = {}
        if "dino_scores" in locals() and dino_scores.numel() > 0:
            for obj_id, idx in object_id_to_index.items():
                if idx < dino_scores.numel():
                    object_id_to_score[obj_id] = dino_scores[idx].item()
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in runtime.video_predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=step,
            start_frame_idx=start_frame_idx,
        ):
            frame_masks = MaskDictionaryModel()
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = out_mask_logits[i] > 0.0
                if out_obj_id in object_id_to_score:
                    logit_value = object_id_to_score[out_obj_id]
                else:
                    logit_value = float(torch_module.max(out_mask_logits[i]))
                object_info = ObjectInfo(
                    instance_id=out_obj_id,
                    mask=out_mask[0],
                    class_name=mask_dict.get_target_class_name(out_obj_id),
                    logit=logit_value,
                )
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                padded_base_name = original_to_padded[frame_names[out_frame_idx]].split(".")[0]
                frame_masks.mask_name = f"{padded_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch_module.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id
            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_output_path, frame_masks_info.mask_name), mask_img)
            frame_base_name = frame_masks_info.mask_name.replace(".npy", "")
            sequence_annotations[frame_base_name] = frame_masks_info.to_dict()

    try:
        print(f"Performing reverse tracking for {video_name}")
        start_object_id = 0
        object_info_dict = {}
        for frame_idx, current_object_count in frame_object_count.items():
            try:
                masks_added = False
                if frame_idx != 0:
                    runtime.video_predictor.reset_state(inference_state)
                    padded_base_name = original_to_padded[frame_names[frame_idx]].split(".")[0]
                    mask_data_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")

                    if padded_base_name not in sequence_annotations or not os.path.exists(mask_data_path):
                        print(f"Warning: Required files not found for frame {frame_idx}, skipping reverse tracking")
                        continue

                    json_data = MaskDictionaryModel().from_dict(sequence_annotations[padded_base_name])
                    try:
                        mask_array = np.load(mask_data_path)
                    except Exception as e:
                        print(f"Error loading mask file for frame {frame_idx}: {str(e)}")
                        continue

                    new_objects_count = 0
                    for object_id in range(start_object_id + 1, current_object_count + 1):
                        if object_id in json_data.labels:
                            object_info_dict[object_id] = json_data.labels[object_id]
                            runtime.video_predictor.add_new_mask(
                                inference_state,
                                frame_idx,
                                object_id,
                                mask_array == object_id,
                            )
                            new_objects_count += 1
                    masks_added = new_objects_count > 0
                start_object_id = current_object_count
                if masks_added:
                    try:
                        for out_frame_idx, out_obj_ids, out_mask_logits in runtime.video_predictor.propagate_in_video(
                            inference_state,
                            max_frame_num_to_track=step * 2,
                            start_frame_idx=frame_idx,
                            reverse=True,
                        ):
                            try:
                                padded_base_name = original_to_padded[frame_names[out_frame_idx]].split(".")[0]
                                mask_data_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")

                                if padded_base_name not in sequence_annotations or not os.path.exists(mask_data_path):
                                    print(f"Warning: Required output files not found for frame {out_frame_idx}, skipping")
                                    continue

                                try:
                                    json_data = MaskDictionaryModel().from_dict(sequence_annotations[padded_base_name])
                                    mask_array = np.load(mask_data_path)
                                except Exception as e:
                                    print(f"Error loading output files for frame {out_frame_idx}: {str(e)}")
                                    continue

                                for i, out_obj_id in enumerate(out_obj_ids):
                                    try:
                                        out_mask = (out_mask_logits[i] > 0.0).cpu()
                                        if out_mask.sum() == 0:
                                            continue
                                        if out_obj_id not in object_info_dict:
                                            print(f"Warning: Object ID {out_obj_id} not found in tracking dictionary, skipping")
                                            continue
                                        object_info = object_info_dict[out_obj_id]
                                        object_info.mask = out_mask[0]
                                        object_info.update_box()
                                        json_data.labels[out_obj_id] = object_info
                                        mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                                        mask_array[object_info.mask] = out_obj_id
                                    except Exception as e:
                                        print(f"Error processing object ID {out_obj_id}: {str(e)}")
                                        continue

                                np.save(mask_data_path, mask_array)
                                sequence_annotations[padded_base_name] = json_data.to_dict()
                            except Exception as e:
                                print(f"Error processing output frame {out_frame_idx}: {str(e)}")
                                continue
                    except Exception as e:
                        print(f"Error during reverse propagation for frame {frame_idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing frame {frame_idx} for reverse tracking: {str(e)}")
                continue
    except Exception as e:
        print(f"Error performing reverse tracking: {str(e)}")
        print("Continuing with visualization despite reverse tracking errors")

    aggregated_annotation_path = write_sequence_annotations(
        det_output_path,
        video_name,
        sequence_annotations,
        sequence_key=det_sequence_key,
    )
    print(f"Saved aggregated annotations to {aggregated_annotation_path}")

    if enable_visualization and masked_image_output_path and masked_video_output_path:
        try:
            from utils.common_utils import CommonUtils
        except ModuleNotFoundError:
            from object_centric_extractor.utils.common_utils import CommonUtils

        print("Drawing masks and creating visualization outputs...")

        output_video_name = f"{video_name}.mp4"
        output_video_path = os.path.join(masked_video_output_path, output_video_name)

        print("video_path", video_path)
        print("mask_output_path", mask_output_path)
        print("det_output_path", det_output_path)
        print("masked_image_output_path", masked_image_output_path)

        CommonUtils.draw_masks_and_box_with_supervision(
            video_path,
            mask_output_path,
            det_output_path,
            masked_image_output_path,
            draw_labels=False,
            draw_boxes=False,
            sequence_key=det_sequence_key,
        )

        print("Creating video from masked images...")
        create_video_from_images(masked_image_output_path, output_video_path, frame_rate=15)

        print(f"Video saved to {output_video_path}")
        print(f"Masked images saved to {masked_image_output_path}")

    return sequence_annotations
