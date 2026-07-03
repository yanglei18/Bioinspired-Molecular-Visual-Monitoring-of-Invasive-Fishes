# -*- coding: utf-8 -*-
"""
Stage 2: LLM-based Attribute Validation + CoT Reasoning Generation

Reads Stage 1 VQA results, validates extracted attributes against expert
definitions using an LLM checker, then generates logic-gated CoT reasoning.

Usage:
    python data_pipeline/stage2_cot.py --config configs/data_pipeline.yaml
"""

import os
import json
import argparse
import asyncio
import re
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncAzureOpenAI, APIConnectionError, RateLimitError, InternalServerError, BadRequestError, AuthenticationError

from expert_kd import get_fish_expert_knowledge
from llm_check_prompt import AppearanceLLMCheckPrompt, BehaviorLLMCheckPrompt
from llm_reason_prompt import RethinkPrompt


def extract_json_from_text(text):
    if not text:
        return None
    text = text.strip()
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        json_str = text[start:end + 1] if start != -1 and end != -1 else text
    json_str = json_str.replace("\u2013", "-")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


async def call_model(client, messages, model_name, system_text=None):
    full_messages = []
    if system_text:
        full_messages.append({"role": "system", "content": system_text})
    if isinstance(messages, str):
        full_messages.append({"role": "user", "content": messages})
    else:
        full_messages.extend(messages)

    while True:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                max_tokens=16384,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if "429" in str(e) or isinstance(e, RateLimitError):
                continue
            raise
        except Exception as e:
            if "429" in str(e):
                continue
            raise


async def process_single_file(client, file_path, fish_code, fish_real_name,
                               vqa_results_dir, output_dir, model_name):
    video_id = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_dir, fish_code)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{video_id}.json")

    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("rethink_annotation"):
                return existing
        except Exception:
            pass

    with open(file_path, "r", encoding="utf-8") as f:
        vqa_results = json.load(f)

    if "error" in vqa_results:
        final_results = {"video_id": video_id, "fish_type": fish_real_name, "error": vqa_results["error"]}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        return final_results

    expert_knowledge = get_fish_expert_knowledge(fish_real_name)
    appearance_vocab = expert_knowledge.get("appearance_attributes", {})
    behavior_vocab = expert_knowledge.get("behavior_attributes", {})
    confusions_vocab = expert_knowledge.get("confusions", [])

    # Build prefixed definitions for LLM checker
    appearance_definitions_with_prefix = []
    appearance_attributes_dict = {}
    for i, key in enumerate(appearance_vocab.keys()):
        val = f"A{i + 1}. {appearance_vocab[key]}"
        appearance_definitions_with_prefix.append(val)
        appearance_attributes_dict[key] = val

    behavior_definitions_with_prefix = []
    behavior_attributes_dict = {}
    for i, key in enumerate(behavior_vocab.keys()):
        val = f"B{i + 1}. {behavior_vocab[key]}"
        behavior_definitions_with_prefix.append(val)
        behavior_attributes_dict[key] = val

    # Map Stage 1 attribute names to prefixed versions
    orig_app = vqa_results.get("appearance_results", {}).get("appearance_attributes", [])
    updated_app = []
    for attr in orig_app:
        raw_name = attr.get("name", "")
        if raw_name in appearance_attributes_dict:
            new_attr = attr.copy()
            new_attr["name"] = appearance_attributes_dict[raw_name]
            updated_app.append(new_attr)

    appearance_keys = list(appearance_vocab.keys())
    updated_app.sort(
        key=lambda x: appearance_keys.index(
            next(k for k, v in appearance_attributes_dict.items() if v == x["name"])
        )
    )

    orig_beh = vqa_results.get("behavior_results", {}).get("behavior_attributes", [])
    updated_beh = []
    for attr in orig_beh:
        raw_name = attr.get("name", "")
        if raw_name in behavior_attributes_dict:
            new_attr = attr.copy()
            new_attr["name"] = behavior_attributes_dict[raw_name]
            updated_beh.append(new_attr)

    behavior_keys = list(behavior_vocab.keys())
    updated_beh.sort(
        key=lambda x: behavior_keys.index(
            next(k for k, v in behavior_attributes_dict.items() if v == x["name"])
        )
    )

    # Step 1: Validate appearance attributes
    app_checker = AppearanceLLMCheckPrompt(expert_definitions=appearance_definitions_with_prefix)
    app_prompt = app_checker.appearance_prompt(video_id=video_id, appearance_candidates=updated_app)
    app_resp = await call_model(client, app_prompt, model_name, system_text=app_checker.appearance_system())
    validated_appearance = extract_json_from_text(app_resp) or {"validated_appearance": []}

    # Step 2: Validate behavior attributes
    beh_checker = BehaviorLLMCheckPrompt(expert_definitions=behavior_definitions_with_prefix)
    beh_prompt = beh_checker.behavior_prompt(video_id=video_id, behavior_candidates=updated_beh)
    beh_resp = await call_model(client, beh_prompt, model_name, system_text=beh_checker.behavior_system())
    validated_behavior = extract_json_from_text(beh_resp) or {"validated_behavior": []}

    # Step 3: Generate CoT reasoning
    rethinkor = RethinkPrompt()
    rethink_prompt = rethinkor.rethink_prompt(
        validated_appearance=validated_appearance.get("validated_appearance", []),
        validated_behavior=validated_behavior.get("validated_behavior", []),
        confusions_vocab=confusions_vocab,
        task_spec={"Classification": "fish species identification"},
    )
    rethink_resp = await call_model(client, rethink_prompt, model_name, system_text=rethinkor.system_prompt())
    rethink_results = extract_json_from_text(rethink_resp) or {}

    final_results = {
        "video_id": video_id,
        "fish_type": fish_real_name,
        "appearance_vocab": appearance_vocab,
        "behavior_vocab": behavior_vocab,
        "extracted_appearance_attributes": vqa_results.get("appearance_results", {}),
        "extracted_behavior_attributes": vqa_results.get("behavior_results", {}),
        "validated_appearance_attributes": validated_appearance,
        "validated_behavior_attributes": validated_behavior,
        "rethink_annotation": rethink_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    return final_results


def load_config(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main():
    parser = argparse.ArgumentParser(description="Stage 2: CoT Validation & Reasoning")
    parser.add_argument("--config", type=str, default="configs/data_pipeline.yaml",
                        help="Path to data pipeline config YAML")
    parser.add_argument("--fish-code", type=str, default=None,
                        help="Fish code (overrides config)")
    parser.add_argument("--fish-name", type=str, default=None,
                        help="Fish real name (overrides config)")
    parser.add_argument("--vqa-dir", type=str, default=None,
                        help="Stage 1 VQA results directory (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage2_cfg = cfg.get("stage2", {})

    api_key = os.environ.get("FGVLM_API_KEY", stage2_cfg.get("api_key", ""))
    endpoint = os.environ.get("FGVLM_ENDPOINT", stage2_cfg.get("endpoint", ""))
    api_version = os.environ.get("FGVLM_API_VERSION", stage2_cfg.get("api_version", "2024-03-01-preview"))
    model_name = os.environ.get("FGVLM_MODEL_NAME", stage2_cfg.get("model_name", ""))

    fish_code = args.fish_code or stage2_cfg.get("fish_code", "")
    fish_real_name = args.fish_name or stage2_cfg.get("fish_real_name", "")
    vqa_results_dir = args.vqa_dir or stage2_cfg.get("vqa_results_dir", "stage1_vqa_results")
    output_dir = args.output_dir or stage2_cfg.get("output_dir", "stage2_cot_results")
    max_concurrency = stage2_cfg.get("max_concurrency", 50)

    client = AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    in_dir = os.path.join(vqa_results_dir, fish_code)
    os.makedirs(os.path.join(output_dir, fish_code), exist_ok=True)
    candidates = [os.path.join(in_dir, f) for f in sorted(os.listdir(in_dir)) if f.endswith(".json")]
    print(f"Found {len(candidates)} stage1 JSON files for {fish_real_name}.")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(path):
        async with semaphore:
            try:
                return await process_single_file(
                    client, path, fish_code, fish_real_name,
                    vqa_results_dir, output_dir, model_name
                )
            except (BadRequestError, AuthenticationError) as e:
                video_id = os.path.splitext(os.path.basename(path))[0]
                result = {"video_id": video_id, "fish_type": fish_real_name, "error": str(e)}
                out_file = os.path.join(output_dir, fish_code, f"{video_id}.json")
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                return result

    results = await tqdm_asyncio.gather(
        *(worker(path) for path in candidates),
        desc=f"Stage2 {fish_real_name}"
    )
    summary_path = os.path.join(output_dir, f"{fish_code}_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Stage2 done. Results saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
