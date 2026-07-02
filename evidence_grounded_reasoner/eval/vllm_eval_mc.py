"""Fast vLLM-based MC eval for fish-species checkpoints.

Mirrors scripts/eval_parallel_mc_local.py semantics (same system prompt, greedy
decode, <answer> extraction, option-or-species judge, macro accuracy) but uses
vLLM for ~10-20x throughput. Intended as a *probe* to pick the best checkpoint;
the released numbers are produced by the canonical PtEngine script.

One process owns one GPU (CUDA_VISIBLE_DEVICES set by launcher). Each process
evals a shard of val_mc and writes worker_<gpu>.json. A separate --reduce call
aggregates shards into eval_results.json.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

# Match the canonical eval's media settings exactly.
os.environ.setdefault("MAX_PIXELS", str(1024 * 1024 * 2))
os.environ.setdefault("VIDEO_MAX_PIXELS", "401408")
os.environ.setdefault("FPS_MAX_FRAMES", "512")

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert ichthyologist. Analyze the provided video and identify the fish species "
    "from the options in the user prompt. You must present your full, step-by-step reasoning "
    "process using exactly the following sections:\n"
    "- <think>: Initial appearance and behavior observations with confidence scores.\n"
    "- <rethink>: Detailed analysis including supporting evidence, exclusion of alternatives, "
    "and uncertainty reasoning.\n"
    "- <answer>: The final answer in the exact option format from the user prompt, such as "
    "'(A) black carp'."
)


def is_correct_mc(prediction, ground_truth):
    def _parse_answer(text):
        text = text.lower().strip()
        match = re.search(r"\(([a-z])\)\s*(.*)", text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, text

    gt_option, gt_species = _parse_answer(ground_truth)
    pred_option, pred_species = _parse_answer(prediction)
    if not gt_option or not gt_species:
        return False
    if gt_option and pred_option and gt_option == pred_option:
        return True
    if gt_species and pred_species and gt_species == pred_species:
        return True
    return False


def extract_prediction(model_raw_output):
    text = (model_raw_output or "").strip()
    if not text:
        return ""
    if "<answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1].strip()
        if tail:
            return tail.splitlines()[0].strip()
    matches = re.findall(r"\([A-Za-z]\)\s*[^\n\r]+", text)
    if matches:
        return matches[-1].strip()
    return text.splitlines()[-1].strip()


def summarize(results):
    species = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt = r.get("ground_truth")
        if not gt:
            continue
        m = re.search(r"\)\s*(.*)", gt)
        name = m.group(1).strip() if m else gt
        species[name]["total"] += 1
        if r.get("is_correct"):
            species[name]["correct"] += 1
    per = {}
    for name, s in sorted(species.items()):
        acc = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
        per[name] = {"correct": s["correct"], "total": s["total"], "accuracy": f"{acc:.2f}%"}
    macro = sum((s["correct"] / s["total"] * 100) if s["total"] else 0.0
                for s in species.values()) / len(species) if species else 0.0
    return per, macro


def run_worker(args):
    from vllm import LLM, SamplingParams

    with open(args.benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # shard by global index so shards are disjoint and reproducible
    shard = [d for i, d in enumerate(data) if i % args.num_shards == args.shard_id]

    llm = LLM(
        model=args.checkpoint_path,
        limit_mm_per_prompt={"video": 1},
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        allowed_local_media_path="/",
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    results = []
    valid = []
    prompts = []
    for item in shard:
        vp = item.get("video_path") or (item.get("videos") or [None])[0]
        if not vp or not os.path.exists(vp):
            continue
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "file://" + vp}},
                {"type": "text", "text": item["question"]},
            ]},
        ]
        prompts.append(messages)
        valid.append(item)

    # vLLM chat handles the multimodal template + video loading via qwen_vl_utils
    outputs = llm.chat(prompts, sp)
    for item, out in zip(valid, outputs):
        raw = out.outputs[0].text
        pred = extract_prediction(raw)
        ok = is_correct_mc(pred, item["answer"])
        results.append({
            "video_path": item.get("video_path"),
            "question": item["question"],
            "ground_truth": item["answer"],
            "prediction": pred,
            "model_raw_output": raw,
            "is_correct": bool(ok),
        })

    os.makedirs(args.run_dir, exist_ok=True)
    shard_path = os.path.join(args.run_dir, f"worker_{args.shard_id:02d}.json")
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[shard {args.shard_id}] wrote {len(results)} results -> {shard_path}")


def run_reduce(args):
    results = []
    for fn in sorted(os.listdir(args.run_dir)):
        if fn.startswith("worker_") and fn.endswith(".json"):
            with open(os.path.join(args.run_dir, fn), "r", encoding="utf-8") as f:
                results.extend(json.load(f))
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    per, macro = summarize(results)
    sample_acc = (correct / total * 100) if total else 0.0
    out = {
        "summary": {
            "checkpoint_path": args.checkpoint_path,
            "benchmark_file": args.benchmark_file,
            "total_samples": total,
            "correct_samples": correct,
            "macro_accuracy": f"{macro:.2f}%",
            "sample_accuracy": f"{sample_acc:.2f}%",
            "engine": "vllm",
        },
        "species_accuracy": per,
    }
    with open(os.path.join(args.run_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print("\nPer-species:")
    for k, v in per.items():
        print(f"  {k:22s} {v['accuracy']:>8s}  ({v['correct']}/{v['total']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["worker", "reduce"], required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--benchmark-file", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = p.parse_args()
    if args.mode == "worker":
        run_worker(args)
    else:
        run_reduce(args)


if __name__ == "__main__":
    main()
