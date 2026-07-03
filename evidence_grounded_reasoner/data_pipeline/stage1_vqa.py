# -*- coding: utf-8 -*-
"""
Stage 1: Video-level VQA Attribute Extraction

For each fish video, send the video + expert knowledge rules to a VLM
and extract appearance / behavior attributes with visual evidence.

Usage:
    python data_pipeline/stage1_vqa.py --config configs/data_pipeline.yaml
"""

import os
import json
import argparse
import asyncio
import base64
import re
import tempfile
import cv2
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI, APIConnectionError, RateLimitError, InternalServerError, BadRequestError, AuthenticationError

from expert_kd import get_fish_expert_knowledge

SYSTEM_PROMPT = """\
You are an expert Ichthyologist and AI Vision Assistant.
Your task is to analyze a video clip of a specific fish species.
You will be provided with a set of Expert Knowledge Rules.
Check if the visual evidence in the video matches or contradicts these expert definitions.
For each attribute provided in the prompt, determine its status based solely on the video.
Be objective. If an attribute is not visible, state "Not visible". Output purely valid JSON.
"""


def encode_video_limited_frames(video_path: str, max_frames: int = 50) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if 0 < total_frames <= max_frames:
        cap.release()
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    step = max(1, total_frames // max_frames) if total_frames > 0 else 1
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    count = 0
    frame_idx = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out.write(frame)
            count += 1
        frame_idx += 1

    cap.release()
    out.release()

    try:
        with open(temp_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return encoded_string


def clean_json_string(response_text: str) -> str:
    clean_text = re.sub(r"^```json|```$", "", response_text.strip(), flags=re.MULTILINE)
    clean_text = clean_text.replace("\u2013", "-")
    return clean_text.strip()


async def get_response_with_retry(client, messages, model_name, max_retries=None):
    attempt = 0
    while True:
        attempt += 1
        try:
            return await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=16384,
                temperature=0.2,
            )
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if "429" in str(e) or isinstance(e, RateLimitError):
                continue
            raise
        except Exception as e:
            if "429" in str(e):
                continue
            raise


async def process_single_video(client, video_path, fish_code, fish_real_name,
                                max_frame_limit, output_dir, model_name):
    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    out_dir = os.path.join(output_dir, fish_code)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{video_id}.json")

    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if "error" not in existing:
                return existing
        except Exception:
            pass

    try:
        base64_video = await asyncio.to_thread(
            encode_video_limited_frames, video_path, max_frame_limit
        )
        expert_info = get_fish_expert_knowledge(fish_real_name)
        app_dict = expert_info.get("appearance_attributes", {})
        beh_dict = expert_info.get("behavior_attributes", {})

        app_rules_str = "".join(
            f"- Attribute: **{k}**\n  Standard Definition: {v}\n"
            for k, v in app_dict.items()
        )
        beh_rules_str = "".join(
            f"- Attribute: **{k}**\n  Standard Definition: {v}\n"
            for k, v in beh_dict.items()
        )

        user_prompt = f"""\
Target Species: **{fish_real_name}**

I provided a video clip (sampled to at most {max_frame_limit} frames).
Please verify the following Expert Rules against the visual evidence.

### Part 1: Appearance Verification
{app_rules_str}

### Part 2: Behavior Verification
{beh_rules_str}

REQUIRED JSON OUTPUT FORMAT:
{{
  "appearance_attributes": [
    {{"name": "Exact Attribute Key from rules", "status": "Describe what is actually seen", "confidence": "High/Medium/Low", "evidence": "Visual reasoning"}}
  ],
  "behavior_attributes": [
    {{"name": "Exact Attribute Key from rules", "status": "Describe actual behavior", "confidence": "High/Medium/Low", "evidence": "Visual reasoning"}}
  ]
}}
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{base64_video}"}},
                ],
            },
        ]

        response = await get_response_with_retry(client, messages, model_name)
        raw_content = response.choices[0].message.content
        clean_content = clean_json_string(raw_content)

        try:
            parsed_json = json.loads(clean_content)
        except json.JSONDecodeError:
            parsed_json = {"error": "json_parse_fail", "raw_content": raw_content}

        result = {
            "video_id": video_id,
            "fish_type": fish_real_name,
            "fish_code": fish_code,
            "appearance_results": {"appearance_attributes": parsed_json.get("appearance_attributes", [])},
            "behavior_results": {"behavior_attributes": parsed_json.get("behavior_attributes", [])},
            "raw_stage1_result": parsed_json,
        }
    except BadRequestError as e:
        result = {"video_id": video_id, "error": f"400 Bad Request: {e}", "file_path": video_path}
    except AuthenticationError as e:
        result = {"video_id": video_id, "error": f"Authentication Error: {e}", "file_path": video_path}
    except Exception as e:
        result = {"video_id": video_id, "error": str(e), "file_path": video_path}

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def load_config(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main():
    parser = argparse.ArgumentParser(description="Stage 1: VQA Attribute Extraction")
    parser.add_argument("--config", type=str, default="configs/data_pipeline.yaml",
                        help="Path to data pipeline config YAML")
    parser.add_argument("--fish-code", type=str, default=None,
                        help="Fish code (overrides config)")
    parser.add_argument("--fish-name", type=str, default=None,
                        help="Fish real name (overrides config)")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Video directory (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage1_cfg = cfg.get("stage1", {})

    api_key = os.environ.get("FGVLM_API_KEY", stage1_cfg.get("api_key", ""))
    endpoint = os.environ.get("FGVLM_ENDPOINT", stage1_cfg.get("endpoint", ""))
    api_version = os.environ.get("FGVLM_API_VERSION", stage1_cfg.get("api_version", "2024-03-01-preview"))
    model_name = os.environ.get("FGVLM_MODEL_NAME", stage1_cfg.get("model_name", ""))

    fish_code = args.fish_code or stage1_cfg.get("fish_code", "")
    fish_real_name = args.fish_name or stage1_cfg.get("fish_real_name", "")
    video_dir = args.video_dir or stage1_cfg.get("video_dir", "")
    output_dir = args.output_dir or stage1_cfg.get("output_dir", "stage1_vqa_results")
    max_frame_limit = stage1_cfg.get("max_frame_limit", 50)
    max_concurrency = stage1_cfg.get("max_concurrency", 50)
    max_file_size_mb = stage1_cfg.get("max_file_size_mb", 500)

    client = AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    candidates = []
    for file in sorted(os.listdir(video_dir)):
        if file.lower().endswith((".mp4", ".mov", ".avi")):
            full_path = os.path.join(video_dir, file)
            if os.path.getsize(full_path) < max_file_size_mb * 1024 * 1024:
                candidates.append(full_path)

    print(f"Found {len(candidates)} {fish_real_name} videos.")
    os.makedirs(os.path.join(output_dir, fish_code), exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(path):
        async with semaphore:
            return await process_single_video(
                client, path, fish_code, fish_real_name,
                max_frame_limit, output_dir, model_name
            )

    results = await tqdm_asyncio.gather(
        *(worker(path) for path in candidates),
        desc=f"Stage1 {fish_real_name}"
    )
    summary_path = os.path.join(output_dir, f"{fish_code}_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Stage1 done. Results saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
