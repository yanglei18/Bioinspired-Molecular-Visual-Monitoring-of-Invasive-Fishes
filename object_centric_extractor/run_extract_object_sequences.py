#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import re
import shutil
import cv2
import argparse
from tqdm import tqdm
from utils.annotation_io import (
    count_sequence_annotation_frames,
    load_sequence_annotations,
)
from pipeline.config import (
    CropConfig,
    DetectionConfig,
    load_detector_config,
)
from pipeline.crop_export import (
    calculate_instance_sizes,
    count_instance_occurrences,
    process_frame,
)
from pipeline.frame_extraction import (
    cleanup_empty_parent_dirs,
    extract_video_to_temp_frames,
)
from pipeline.input_discovery import (
    IMAGE_EXTENSIONS,
    discover_video_inputs,
    is_supported_video_file,
    resolve_video_output_layout,
)
from pipeline.instance_video import export_instance_mp4s
from pipeline.runtime import build_runtime
from pipeline.tracking import run_tracking_stage


DEFAULT_CONFIG_CLI_PATH = "object_centric_extractor/configs/default.yaml"


# Helper functions for instance cropping
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def release_torch_memory():
    """Release Python and CUDA caches between videos."""
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_files_with_suffix(directory, suffix):
    if not directory or not os.path.isdir(directory):
        return 0
    return sum(1 for name in os.listdir(directory) if name.endswith(suffix))




def has_completed_tracking_outputs(
    det_output_path,
    expected_frames,
    sequence_key=None,
    enable_visualization=False,
    masked_video_output_path=None,
    video_name=None,
    require_mask_outputs=False,
    mask_output_path=None,
):
    if expected_frames <= 0:
        return False

    annotation_frame_count = count_sequence_annotation_frames(
        det_output_path,
        sequence_key=sequence_key,
    )
    if annotation_frame_count < expected_frames:
        return False

    if require_mask_outputs:
        mask_count = count_files_with_suffix(mask_output_path, ".npy")
        if mask_count < expected_frames:
            return False

    if enable_visualization:
        if not masked_video_output_path or not video_name:
            return False
        expected_video_path = os.path.join(masked_video_output_path, f"{video_name}.mp4")
        if not os.path.isfile(expected_video_path):
            return False

    return True


def has_completed_cropping_outputs(instance_image_output_path, instance_video_output_path, video_name):
    if not os.path.isdir(instance_image_output_path) or not os.path.isdir(instance_video_output_path):
        return False

    webp_pattern = re.compile(rf"^{re.escape(video_name)}_\d+_\d+\.webp$", re.IGNORECASE)
    mp4_pattern = re.compile(rf"^{re.escape(video_name)}_\d+\.mp4$", re.IGNORECASE)

    has_webp = any(webp_pattern.match(filename) for filename in os.listdir(instance_image_output_path))
    has_mp4 = any(mp4_pattern.match(filename) for filename in os.listdir(instance_video_output_path))
    return has_webp and has_mp4


def process_video(video_path, annotation_det_path, annotation_mask_path, instance_image_root_dir, instance_video_root_dir, text_prompt="fish.", 
                do_tracking=True, do_cropping=True, cleanup_temp=True, enable_visualization=True,
                output_masked_image_dir=None, output_masked_video_dir=None, source_path=None,
                crop_config=None, detection_config=None, runtime=None):
    """Process a video with tracking and instance cropping"""
    crop_config = crop_config or CropConfig()
    detection_config = detection_config or DetectionConfig(text_prompt=text_prompt)
    text_prompt = detection_config.text_prompt
    source_reference = source_path or video_path
    video_name, output_subpath, masked_video_subpath = resolve_video_output_layout(source_reference)

    # Create output directories
    det_output_path = annotation_det_path
    det_sequence_key = output_subpath
    mask_output_path = os.path.join(annotation_mask_path, output_subpath)
    instance_webp_output_path = instance_image_root_dir
    instance_mp4_output_path = instance_video_root_dir

    # Only create visualization paths if visualization is enabled
    masked_image_output_path = None
    masked_video_output_path = None
    if enable_visualization and output_masked_image_dir and output_masked_video_dir:
        masked_image_output_path = os.path.join(output_masked_image_dir, output_subpath)
        if masked_video_subpath is None:
            masked_video_output_path = output_masked_video_dir
        else:
            masked_video_output_path = os.path.join(output_masked_video_dir, masked_video_subpath)
    
    # Create temporary directories if they'll be cleaned up
    temp_dirs = []
    if cleanup_temp:
        # Only create these as temp dirs if we're going to clean them up
        if do_tracking:
            temp_dirs.append(mask_output_path)
            if enable_visualization and masked_image_output_path:
                temp_dirs.append(masked_image_output_path)
    
    ensure_dir(det_output_path)
    ensure_dir(mask_output_path)
    if do_cropping:
        ensure_dir(instance_webp_output_path)
        ensure_dir(instance_mp4_output_path)
    
    # Create visualization directories if enabled
    if enable_visualization and masked_image_output_path and masked_video_output_path:
        ensure_dir(masked_image_output_path)
        ensure_dir(masked_video_output_path)
    
    # Find all frame images
    frame_names = []
    for f in os.listdir(video_path):
        if os.path.splitext(f)[-1].lower() in IMAGE_EXTENSIONS:
            if os.path.isfile(os.path.join(video_path, f)):
                frame_names.append(f)
    
    if not frame_names:
        print(f"No frame images found in {video_path}")
        return
    
    # Sort frames by frame number
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        print(f"Warning: Could not sort frames by number, using lexicographic sorting")
        frame_names.sort()
    
    # Create filename mapping
    original_to_padded = {}
    padded_to_original = {}
    
    for frame in frame_names:
        name, ext = os.path.splitext(frame)
        try:
            number = int(name)
            padded_name = f"{number:06d}{ext}"
        except ValueError:
            print(f"Warning: Frame name {name} is not a numeric value, using as is")
            padded_name = frame
        
        original_to_padded[frame] = padded_name
        padded_to_original[padded_name] = frame

    expected_frame_count = len(frame_names)
    run_tracking = do_tracking
    run_cropping = do_cropping

    if run_tracking and has_completed_tracking_outputs(
        det_output_path=det_output_path,
        expected_frames=expected_frame_count,
        sequence_key=det_sequence_key,
        enable_visualization=enable_visualization,
        masked_video_output_path=masked_video_output_path,
        video_name=video_name,
        require_mask_outputs=run_cropping,
        mask_output_path=mask_output_path,
    ):
        print(f"Skipping tracking for {video_name}: existing annotations found in {det_output_path}")
        run_tracking = False

    if run_cropping and has_completed_cropping_outputs(instance_webp_output_path, instance_mp4_output_path, video_name):
        print(
            f"Skipping instance cropping for {video_name}: existing outputs found in "
            f"{instance_webp_output_path} and {instance_mp4_output_path}"
        )
        run_cropping = False

    if not run_tracking:
        temp_dirs = []

    if not run_tracking and not run_cropping:
        print(f"Skipping {video_name}: all requested outputs already exist.")
        return

    sequence_annotations = {} if run_tracking else load_sequence_annotations(
        det_output_path,
        sequence_key=det_sequence_key,
    )
    
    # PART 1: Object Detection and Tracking
    if run_tracking:
        if runtime is None:
            raise ValueError("PipelineRuntime is required when do_tracking is enabled.")
        sequence_annotations = run_tracking_stage(
            video_path=video_path,
            frame_names=frame_names,
            original_to_padded=original_to_padded,
            det_output_path=det_output_path,
            det_sequence_key=det_sequence_key,
            mask_output_path=mask_output_path,
            masked_image_output_path=masked_image_output_path,
            masked_video_output_path=masked_video_output_path,
            video_name=video_name,
            detection_config=detection_config,
            runtime=runtime,
            enable_visualization=enable_visualization,
        )
    
    # PART 2: Instance Cropping
    if run_cropping:
        if count_sequence_annotation_frames(det_output_path, sequence_key=det_sequence_key) == 0 or not os.path.exists(mask_output_path):
            print(f"Detection or mask files do not exist for {video_name}, skipping instance cropping")
            return
        
        print(f"Cropping instances from {video_name}")
        
        # Count frames per instance ID; only process instances appearing more than the threshold
        valid_instance_ids, instance_counts = count_instance_occurrences(
            det_output_path,
            min_frames=crop_config.min_tracking_frames,
            sequence_key=det_sequence_key,
        )
        
        if not valid_instance_ids:
            print(f"No instances with tracking length >= {crop_config.min_tracking_frames} found. Skipping cropping.")
            return
        
        # Calculate optimal window sizes for each instance
        instance_sizes = {}
        if crop_config.fixed_window:
            instance_sizes = calculate_instance_sizes(
                det_output_path, 
                padding=crop_config.padding,
                scale_factor=crop_config.scale_factor,
                percentile=crop_config.percentile,
                sequence_key=det_sequence_key,
            )
        
        # Process frames
        total_instances = 0
        for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Cropping instances")):
            # Load the image
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Error: Cannot read frame {frame_path}")
                continue
            
            # Find annotation file
            padded_base_name = original_to_padded[frame_name].split(".")[0]
            mask_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")

            frame_annotation = sequence_annotations.get(padded_base_name)
            if frame_annotation is None:
                continue
            
            # Process the frame
            instances_processed = process_frame(
                frame,
                frame_annotation,
                padded_base_name,
                mask_path if os.path.exists(mask_path) else None,
                instance_webp_output_path,
                video_name,
                frame_idx,
                instance_sizes,
                crop_config.min_size,
                crop_config.padding,
                crop_config.fixed_window,
                list(crop_config.class_filter),
                valid_instance_ids=valid_instance_ids  # pass the list of valid instance IDs
            )
            
            total_instances += instances_processed
        
        print(f"Total instances cropped: {total_instances}")
        exported_videos = export_instance_mp4s(
            instance_webp_output_path,
            instance_mp4_output_path,
            video_name,
            frame_rate=15,
        )
        print(f"Exported {exported_videos} instance video(s) to {instance_mp4_output_path}")
    
    # Clean up temporary directories
    if cleanup_temp:
        print(f"Cleaning up temporary directories...")
        
        # Delete temporary directories
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                print(f"Removing: {temp_dir}")
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error removing directory {temp_dir}: {str(e)}")
        
        # Delete parent directories if they're empty
        if run_tracking and mask_output_path:
            parent_dir = os.path.dirname(mask_output_path)
            try:
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    print(f"Removing empty parent directory: {parent_dir}")
                    os.rmdir(parent_dir)
                
                grandparent_dir = os.path.dirname(parent_dir)
                if os.path.exists(grandparent_dir) and not os.listdir(grandparent_dir):
                    print(f"Removing empty grandparent directory: {grandparent_dir}")
                    os.rmdir(grandparent_dir)
            except Exception as e:
                print(f"Error removing parent directories: {str(e)}")
        
        # For masked_image directories
        if run_tracking and enable_visualization and masked_image_output_path:
            parent_dir = os.path.dirname(masked_image_output_path)
            try:
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    print(f"Removing empty parent directory: {parent_dir}")
                    os.rmdir(parent_dir)
                
                grandparent_dir = os.path.dirname(parent_dir)
                if os.path.exists(grandparent_dir) and not os.listdir(grandparent_dir):
                    print(f"Removing empty grandparent directory: {grandparent_dir}")
                    os.rmdir(grandparent_dir)
            except Exception as e:
                print(f"Error removing parent directories: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Grounded SAM2 tracking, cropping, and instance export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Instance export layout:\n"
            "  outputs.instance_image_dir/*.webp -> <video_name>_<instance_id>_<frame_id>.webp\n"
            "  outputs.instance_video_dir/*.mp4  -> <video_name>_<instance_id>.mp4"
        ),
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_CLI_PATH,
                        help="Path to the SAM2 detector YAML config file")
    args = parser.parse_args()

    detector_config = load_detector_config(args.config)
    pipeline_config = detector_config.pipeline
    runtime_config = detector_config.runtime

    print(
        "Loaded SAM2 detector config: "
        f"{args.config} | input_dir={pipeline_config.input_dir} | "
        f"det_dir={pipeline_config.outputs.det_dir}"
    )

    # Create base output directories
    required_dirs = [pipeline_config.outputs.det_dir]
    if not pipeline_config.cleanup_temp:
        required_dirs.append(pipeline_config.outputs.mask_dir)
    if pipeline_config.do_cropping:
        required_dirs.extend([
            pipeline_config.outputs.instance_image_dir,
            pipeline_config.outputs.instance_video_dir,
        ])
    
    # Add visualization directories if visualization is enabled
    if pipeline_config.enable_visualization:
        required_dirs.extend(
            directory
            for directory in (
                pipeline_config.outputs.masked_image_dir,
                pipeline_config.outputs.masked_video_dir,
            )
            if directory
        )
    
    for directory in required_dirs:
        ensure_dir(directory)
    
    discovered_inputs = discover_video_inputs(pipeline_config.input_dir)
    if not discovered_inputs:
        print(f"No supported videos or frame directories found under {pipeline_config.input_dir}")
        return

    runtime = (
        build_runtime(
            sam2_checkpoint=runtime_config.sam2_checkpoint,
            model_cfg=runtime_config.sam2_model_cfg,
            grounding_model_id=runtime_config.grounding_model_id,
        )
        if pipeline_config.do_tracking
        else None
    )

    # Look for videos or frame directories
    for video_dir in discovered_inputs:
        prepared_input_path = video_dir
        transient_frame_dir = None
        transient_frame_root_dir = None
        # Process the video directory
        try:
            print(f"Processing {video_dir}...")

            if is_supported_video_file(video_dir):
                print(f"Extracting frames from input video {video_dir}...")
                transient_frame_root_dir, transient_frame_dir, extracted_frame_count = extract_video_to_temp_frames(
                    video_dir,
                    pipeline_config.input_dir,
                )
                prepared_input_path = transient_frame_dir
                print(f"Extracted {extracted_frame_count} frame(s) to temporary directory {prepared_input_path}")
            
            process_video(
                prepared_input_path,
                pipeline_config.outputs.det_dir,
                pipeline_config.outputs.mask_dir,
                pipeline_config.outputs.instance_image_dir,
                pipeline_config.outputs.instance_video_dir,
                pipeline_config.detection.text_prompt,
                pipeline_config.do_tracking,
                pipeline_config.do_cropping,
                pipeline_config.cleanup_temp,
                pipeline_config.enable_visualization,
                pipeline_config.outputs.masked_image_dir,
                pipeline_config.outputs.masked_video_dir,
                source_path=video_dir,
                crop_config=pipeline_config.crop,
                detection_config=pipeline_config.detection,
                runtime=runtime,
            )
        except Exception as e:
            print(f"Error processing {video_dir}: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if transient_frame_dir and os.path.isdir(transient_frame_dir):
                print(f"Removing temporary extracted frames for {video_dir}...")
                shutil.rmtree(transient_frame_dir, ignore_errors=True)
                if transient_frame_root_dir and os.path.isdir(transient_frame_root_dir):
                    cleanup_empty_parent_dirs(os.path.dirname(transient_frame_dir), transient_frame_root_dir)
            print(f"Releasing memory after {video_dir}...")
            release_torch_memory()

if __name__ == "__main__":
    main()
