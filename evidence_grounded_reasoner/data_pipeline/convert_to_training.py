# -*- coding: utf-8 -*-
"""
Convert Stage 2 CoT results to SFT / RL training data format.

Reads per-video JSON files from stage2_cot_results and produces:
  - SFT training JSON (with full CoT assistant response)
  - RL training JSON (with solution + reasoning_content)

Usage:
    python data_pipeline/convert_to_training.py \
        --stage2-dir stage2_cot_results \
        --video-root /path/to/videos \
        --sft-output data/sft_train.json \
        --rl-output data/rl_train.json \
        --options "Common carp" "Crucian carp" "Grass carp"
"""

import os
import json
import argparse
import glob


SYSTEM_PROMPT = (
    "You are an expert ichthyologist. Your mission is to analyze the provided video "
    "and identify the fish species. You must present your full, step-by-step reasoning "
    "process. Structure your response with:\n"
    "- `<think>`: A list of initial appearance and behavior observations with confidence scores.\n"
    "- `<rethink>`: A detailed analysis including supporting evidence, exclusion of alternatives, "
    "and uncertainty reasoning.\n"
    "- `<answer>`: The final species identification."
)


def build_cot_text(rethink_annotation: dict) -> str:
    """Convert rethink_annotation dict to the <think>...<rethink>...<answer> format."""
    evidence = rethink_annotation.get("evidence_selection", [])
    supporting = rethink_annotation.get("supporting_reasoning", "")
    exclusion = rethink_annotation.get("exclusion_counterfactual", "")
    uncertainty = rethink_annotation.get("uncertainty reasoning", "") or rethink_annotation.get("uncertainty_reasoning", "")
    summary = rethink_annotation.get("summary", "")

    # Build observation lines
    obs_lines = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str):
                obs_lines.append(f"- {item}")
            elif isinstance(item, dict):
                name = item.get("name", "")
                reason = item.get("reason", "")
                obs_lines.append(f"- {name}: {reason}" if reason else f"- {name}")
    obs_text = "\n".join(obs_lines)

    think_section = f"Observations:\n{obs_text}" if obs_text else "Observations: (none)"
    rethink_section = f"Supporting Reasoning: {supporting}\n\nExclusion/Counterfactual: {exclusion}\n\nUncertainty: {uncertainty}" if supporting else ""
    answer_section = summary

    return f"<think>{think_section}</think>\n\n<rethink>{rethink_section}</rethink><answer>{answer_section}</answer>"


def find_video_path(video_id: str, video_root: str) -> str:
    """Find the video file for a given video_id under video_root."""
    for ext in (".mp4", ".mov", ".avi"):
        path = os.path.join(video_root, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    # Try recursive search
    matches = glob.glob(os.path.join(video_root, "**", f"{video_id}.*"), recursive=True)
    return matches[0] if matches else ""


def main():
    parser = argparse.ArgumentParser(description="Convert Stage 2 results to training data")
    parser.add_argument("--stage2-dir", type=str, required=True,
                        help="Root directory of Stage 2 results")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing video files (same structure as stage2-dir)")
    parser.add_argument("--sft-output", type=str, default="data/sft_train.json",
                        help="Output SFT training JSON path")
    parser.add_argument("--rl-output", type=str, default="data/rl_train.json",
                        help="Output RL training JSON path")
    parser.add_argument("--options", nargs="+", default=None,
                        help="Candidate species names for the multiple-choice question (e.g. 'Common carp' 'Crucian carp')")
    args = parser.parse_args()

    sft_data = []
    rl_data = []

    # Iterate over species subdirectories
    for species_dir in sorted(glob.glob(os.path.join(args.stage2_dir, "*"))):
        if not os.path.isdir(species_dir):
            continue
        fish_code = os.path.basename(species_dir)
        video_dir = os.path.join(args.video_root, fish_code)

        for json_file in sorted(glob.glob(os.path.join(species_dir, "*.json"))):
            with open(json_file, "r", encoding="utf-8") as f:
                result = json.load(f)

            if "error" in result:
                continue

            video_id = result.get("video_id", "")
            fish_type = result.get("fish_type", "")
            rethink = result.get("rethink_annotation", {})

            if not rethink:
                continue

            video_path = find_video_path(video_id, video_dir)
            if not video_path:
                print(f"Warning: video not found for {video_id}, skipping")
                continue

            cot_text = build_cot_text(rethink)

            # Build user question
            if args.options:
                option_str = "\nOptions:\n" + "\n".join(
                    f"({chr(65+i)}) {opt}" for i, opt in enumerate(args.options)
                )
                correct_label = f"({chr(65 + args.options.index(fish_type))}) {fish_type}" if fish_type in args.options else fish_type
            else:
                option_str = ""
                correct_label = fish_type

            user_content = f"<video>\nDescribe what kind of fish is in this video.{option_str}"

            # SFT format
            sft_item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": cot_text},
                ],
                "videos": [video_path],
            }
            sft_data.append(sft_item)

            # RL format
            rl_item = {
                "messages": [
                    {"role": "user", "content": user_content},
                ],
                "videos": [video_path],
                "solution": [correct_label],
                "reasoning_content": cot_text,
            }
            rl_data.append(rl_item)

    # Write outputs
    os.makedirs(os.path.dirname(args.sft_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.rl_output) or ".", exist_ok=True)

    with open(args.sft_output, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    with open(args.rl_output, "w", encoding="utf-8") as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)

    print(f"SFT: {len(sft_data)} samples -> {args.sft_output}")
    print(f"RL:  {len(rl_data)} samples -> {args.rl_output}")


if __name__ == "__main__":
    main()
