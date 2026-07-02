import json
import multiprocessing as mp
import os
import re
import sys
import gc
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

_transformers_vendor_path = os.environ.get("TRANSFORMERS_VENDOR_PATH", "").strip()
if _transformers_vendor_path:
    transformers_pkg_dir = os.path.join(_transformers_vendor_path, "transformers")
    if os.path.isdir(transformers_pkg_dir) and _transformers_vendor_path not in sys.path:
        sys.path.insert(0, _transformers_vendor_path)

os.environ["MAX_PIXELS"] = str(1024 * 1024 * 2)
os.environ["VIDEO_MAX_PIXELS"] = "401408"
os.environ["FPS_MAX_FRAMES"] = "512"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BENCHMARK = os.path.join(ROOT_DIR, "data", "val.json")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT_DIR, "eval_outputs")
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
DEFAULT_MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "2048"))
DEFAULT_MAX_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "4"))


def load_swift_infer_runtime():
    try:
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
        return InferRequest, RequestConfig, TransformersEngine
    except ModuleNotFoundError as exc:
        if exc.name != "swift.infer_engine":
            raise
        # Newer ms-swift exposes the PyTorch inference engine via swift.llm.
        from swift.llm import InferRequest, RequestConfig, PtEngine
        return InferRequest, RequestConfig, PtEngine


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


def _coerce_message_content_to_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return ""


def normalize_benchmark_item(item):
    normalized = dict(item)

    if not normalized.get("video_path"):
        videos = normalized.get("videos")
        if isinstance(videos, list) and videos:
            normalized["video_path"] = videos[0]
        elif isinstance(videos, str) and videos:
            normalized["video_path"] = videos

    if not normalized.get("question"):
        messages = normalized.get("messages") or []
        user_message = next(
            (message for message in messages if isinstance(message, dict) and message.get("role") == "user"),
            None,
        )
        if user_message:
            normalized["question"] = _coerce_message_content_to_text(user_message.get("content"))

    if not normalized.get("answer"):
        solution = normalized.get("solution")
        if isinstance(solution, list):
            normalized["answer"] = solution[0] if solution else ""
        elif isinstance(solution, str):
            normalized["answer"] = solution

    return normalized


def parse_arguments():
    parser = ArgumentParser(description="Parallel evaluation of a local checkpoint; writes results to the experiment directory")
    parser.add_argument("--num-gpus", type=int, default=8, help="number of GPUs to use")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="comma-separated physical GPU ids; empty defaults to 0..num_gpus-1",
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--benchmark-file", type=str, default=DEFAULT_BENCHMARK, help="benchmark file path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_ROOT, help="output directory")
    parser.add_argument("--run-name", type=str, default="", help="optional fixed run directory name")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="system prompt")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max generated tokens")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="max batch size per worker; override via EVAL_BATCH_SIZE",
    )
    parser.add_argument("--model-type", type=str, default="qwen3_vl", help="swift model_type (default: qwen3_vl)")
    return parser.parse_args()


def worker_process(worker_args):
    worker_idx, gpu_id, data_chunk, args, run_dir = worker_args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    InferRequest, RequestConfig, TransformersEngine = load_swift_infer_runtime()

    pid = os.getpid()
    print(f"Process {pid} (GPU {gpu_id}): loading inference engine from '{args.checkpoint_path}'...")
    engine = TransformersEngine(args.checkpoint_path, max_batch_size=args.max_batch_size, model_type=args.model_type)
    request_config = RequestConfig(max_tokens=args.max_tokens, temperature=0, stream=False)
    print(f"Process {pid} (GPU {gpu_id}): inference engine loaded.")

    local_results = []
    shard_path = os.path.join(run_dir, f"worker_{worker_idx:02d}_gpu{gpu_id}.json")
    try:
        for i in tqdm(range(0, len(data_chunk), args.max_batch_size), desc=f"GPU-{gpu_id} progress", position=gpu_id):
            batch_items = data_chunk[i : i + args.max_batch_size]
            infer_requests = []
            valid_batch_items = []
            for item in batch_items:
                video_path = item["video_path"]
                if not os.path.exists(video_path):
                    tqdm.write(f"Warning (GPU-{gpu_id}): video file '{video_path}' not found, skipping.")
                    continue

                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": item["question"]},
                ]
                infer_requests.append(InferRequest(messages=messages, videos=[video_path]))
                valid_batch_items.append(item)

            if not infer_requests:
                continue

            try:
                resp_list = engine.infer(infer_requests, request_config)
                for item, resp in zip(valid_batch_items, resp_list):
                    model_raw_output = resp.choices[0].message.content
                    prediction = extract_prediction(model_raw_output)
                    judge = 1 if is_correct_mc(prediction, item["answer"]) else 0
                    local_results.append(
                        {
                            "benchmark_index": item.get("benchmark_index"),
                            "video_path": item["video_path"],
                            "question": item["question"],
                            "ground_truth": item["answer"],
                            "prediction": prediction,
                            "model_raw_output": model_raw_output,
                            "judge": judge,
                            "is_correct": bool(judge),
                        }
                    )
            except Exception as e:
                tqdm.write(f"\nError (GPU-{gpu_id}): error while processing batch: {e}")
                for item in valid_batch_items:
                    local_results.append(
                        {
                            "benchmark_index": item.get("benchmark_index"),
                            "video_path": item["video_path"],
                            "question": item["question"],
                            "ground_truth": item["answer"],
                            "prediction": "ERROR",
                            "model_raw_output": f"Inference failed with error: {e}",
                            "judge": 0,
                            "is_correct": False,
                        }
                    )
    finally:
        try:
            del engine
        except Exception:
            pass
        gc.collect()

    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)

    return {
        "worker_idx": worker_idx,
        "gpu_id": gpu_id,
        "shard_path": shard_path,
        "num_results": len(local_results),
    }


def summarize_species(results):
    species_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for res in results:
        ground_truth = res.get("ground_truth")
        if not ground_truth:
            continue
        match = re.search(r"\)\s*(.*)", ground_truth)
        species_name = match.group(1).strip() if match else ground_truth
        species_results[species_name]["total"] += 1
        if res.get("is_correct"):
            species_results[species_name]["correct"] += 1
    return {
        species: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": f"{(stats['correct'] / stats['total'] * 100) if stats['total'] else 0:.2f}%",
        }
        for species, stats in sorted(species_results.items())
    }


def compute_macro_accuracy(species_accuracy):
    if not species_accuracy:
        return 0.0
    per_species_values = []
    for stats in species_accuracy.values():
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        per_species_values.append((correct / total * 100) if total else 0.0)
    return sum(per_species_values) / len(per_species_values)


def main():
    args = parse_arguments()
    import torch

    available_gpus = torch.cuda.device_count()
    if args.gpu_ids.strip():
        gpu_ids = [int(token.strip()) for token in args.gpu_ids.split(",") if token.strip()]
    else:
        gpu_ids = list(range(min(args.num_gpus, available_gpus)))

    if not gpu_ids:
        raise RuntimeError("No available GPU found.")

    num_gpus_to_use = min(len(gpu_ids), available_gpus if not args.gpu_ids.strip() else len(gpu_ids))

    with open(args.benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    benchmark_data = [normalize_benchmark_item(item) for item in benchmark_data]
    invalid_items = [
        idx
        for idx, item in enumerate(benchmark_data)
        if not item.get("video_path") or not item.get("question") or not item.get("answer")
    ]
    if invalid_items:
        preview = ",".join(str(idx) for idx in invalid_items[:10])
        raise ValueError(
            f"Benchmark items missing required fields after normalization: {preview}"
            + ("..." if len(invalid_items) > 10 else "")
        )
    benchmark_data = [{**item, "benchmark_index": idx} for idx, item in enumerate(benchmark_data)]

    checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    benchmark_name = os.path.basename(args.benchmark_file).replace(".json", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{checkpoint_name}_{benchmark_name}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Using {num_gpus_to_use} GPU(s) for parallel evaluation.")
    print(f"checkpoint: {args.checkpoint_path}")
    print(f"benchmark: {args.benchmark_file}")
    print(f"output dir: {run_dir}")
    print(f"Total samples: {len(benchmark_data)}")
    print(f"max tokens: {args.max_tokens}")
    print(f"max batch size: {args.max_batch_size}")

    data_chunks = [benchmark_data[i::num_gpus_to_use] for i in range(num_gpus_to_use)]
    worker_args_list = [(idx, gpu_ids[idx], data_chunks[idx], args, run_dir) for idx in range(num_gpus_to_use)]

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_gpus_to_use)
    try:
        shard_infos = pool.map(worker_process, worker_args_list)
    finally:
        pool.terminate()
        pool.join()

    current_results = []
    for shard_info in shard_infos:
        with open(shard_info["shard_path"], "r", encoding="utf-8") as f:
            current_results.extend(json.load(f))
    current_results.sort(key=lambda item: item.get("benchmark_index", -1))
    total = len(current_results)
    correct = sum(1 for res in current_results if res["is_correct"])
    sample_accuracy = (correct / total * 100) if total > 0 else 0
    species_accuracy = summarize_species(current_results)
    macro_accuracy = compute_macro_accuracy(species_accuracy)

    final_output = {
        "summary": {
            "checkpoint_path": args.checkpoint_path,
            "benchmark_file": args.benchmark_file,
            "total_samples": total,
            "correct_samples": correct,
            "accuracy": f"{macro_accuracy:.2f}%",
            "macro_accuracy": f"{macro_accuracy:.2f}%",
            "sample_accuracy": f"{sample_accuracy:.2f}%",
            "gpus_used": num_gpus_to_use,
            "max_tokens": args.max_tokens,
            "max_batch_size": args.max_batch_size,
            "system_prompt": args.system_prompt,
        },
        "species_accuracy": species_accuracy,
        "results": current_results,
    }

    output_file = os.path.join(run_dir, "eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint_path}\n")
        f.write(f"benchmark: {args.benchmark_file}\n")
        f.write(f"total: {total}\n")
        f.write(f"correct: {correct}\n")
        f.write(f"macro_accuracy: {macro_accuracy:.2f}%\n")
        f.write(f"sample_accuracy: {sample_accuracy:.2f}%\n")
        f.write(f"gpus_used: {num_gpus_to_use}\n")
        f.write(f"max_tokens: {args.max_tokens}\n")
        f.write(f"max_batch_size: {args.max_batch_size}\n")
        f.write("\nper-species accuracy:\n")
        for species, stats in final_output["species_accuracy"].items():
            f.write(f"{species}\t{stats['accuracy']}\t({stats['correct']}/{stats['total']})\n")

    latest_pointer = os.path.join(args.output_dir, "LATEST_EVAL")
    with open(latest_pointer, "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")

    print("\n========================= Final evaluation summary =========================")
    print(f"Total samples: {total}")
    print(f"Total correct: {correct}")
    print(f"Macro-average accuracy: {macro_accuracy:.2f}%")
    print(f"Micro-average (sample) accuracy: {sample_accuracy:.2f}%")
    print(f"Results file: {output_file}")
    print(f"Summary file: {summary_file}")
    print("================================================================")
    # Force-exit: bypass torch/CUDA destructor hangs that keep the process alive
    # after all work is complete and results are safely written to disk.
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
