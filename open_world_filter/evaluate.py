#!/usr/bin/env python3
"""Multi-GPU batch inference for coarse classifier using torchrun/DDP-style sharding."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Iterator
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import load_classifier_config

DEFAULT_CONFIG_CLI_PATH = "open_world_filter/configs/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Inference for Fish Classification Model with Ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_CLI_PATH,
        help="Path to the coarse classifier YAML config file",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return resolve_args(build_parser().parse_args())


def resolve_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_classifier_config(cli_args.config)
    return argparse.Namespace(
        config=cli_args.config,
        val_dir=config.data.val_dir,
        model_path=config.outputs.model_path,
        reference_dir=config.data.reference_dir,
        output_dir=config.outputs.evaluation_dir,
        batch_size=config.inference.batch_size,
        image_size=config.inference.image_size,
        num_workers=config.inference.num_workers,
        similarity_threshold=config.inference.similarity_threshold,
        median_threshold=config.inference.median_threshold,
        variance_threshold=config.inference.variance_threshold,
        l2_distance_threshold=config.inference.l2_distance_threshold,
        max_samples_per_class=config.inference.max_samples_per_class,
        max_reference_per_class=config.inference.max_reference_per_class,
        num_ensembles=config.inference.num_ensembles,
        distributed=False,
        local_rank=0,
    )


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parse_args()
    raise SystemExit

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

COARSE_CLASSIFIER_DIR = SCRIPT_DIR
DEFAULT_THRESHOLD_CONFIG_NAME = "configs/class_thresholds.json"
THRESHOLD_CONFIG_ENV = "COARSE_CLASSIFIER_THRESHOLD_CONFIG"

from dataset import InferenceDataset, inference_transform as transform
from matching import BatchInferenceModel
from plots import create_evaluation_visualizations, create_open_world_screening_visualization


@dataclass(frozen=True)
class DecisionThreshold:
    similarity_threshold: float
    l2_distance_threshold: float

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        fallback: "DecisionThreshold",
    ) -> "DecisionThreshold":
        return cls(
            similarity_threshold=float(
                data.get("similarity_threshold", fallback.similarity_threshold)
            ),
            l2_distance_threshold=float(
                data.get("l2_distance_threshold", fallback.l2_distance_threshold)
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "l2_distance_threshold": self.l2_distance_threshold,
        }


@dataclass(frozen=True)
class ThresholdConfig:
    config_path: Path | None
    default_thresholds: DecisionThreshold
    class_thresholds: dict[str, DecisionThreshold]

    def for_class(self, class_name: str) -> DecisionThreshold:
        normalized_name = class_name.strip()
        return self.class_thresholds.get(
            normalized_name,
            self.class_thresholds.get(normalized_name.lower(), self.default_thresholds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "default": self.default_thresholds.to_dict(),
            "classes": {
                class_name: thresholds.to_dict()
                for class_name, thresholds in sorted(self.class_thresholds.items())
            },
        }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (np.integer,),
        ):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass(frozen=True)
class DistributedConfig:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(args: argparse.Namespace) -> DistributedConfig:
    del args
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return DistributedConfig(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


@contextlib.contextmanager
def maybe_suppress_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def classify_screening_outcome(
    true_class: str,
    predicted_class: str,
    watchlist_classes: set[str],
) -> dict[str, Any]:
    is_true_in_scope = true_class in watchlist_classes
    is_retained = predicted_class in watchlist_classes

    if is_true_in_scope and is_retained:
        screening_outcome = "tp_screen"
    elif is_true_in_scope and not is_retained:
        screening_outcome = "fn_screen"
    elif not is_true_in_scope and not is_retained:
        screening_outcome = "tn_screen"
    else:
        screening_outcome = "fp_screen"

    return {
        "is_true_in_scope": is_true_in_scope,
        "is_retained": is_retained,
        "screening_outcome": screening_outcome,
    }


def result_with_screening_metadata(
    result: Mapping[str, Any],
    watchlist_classes: set[str],
) -> dict[str, Any]:
    screening_info = classify_screening_outcome(
        true_class=str(result["true_class"]),
        predicted_class=str(result["predicted_class"]),
        watchlist_classes=watchlist_classes,
    )
    merged = dict(result)
    merged.update(screening_info)
    return merged


def summarize_screening_counts(
    watchlist_classes: list[str],
    tp_screen: int,
    fn_screen: int,
    tn_screen: int,
    fp_screen: int,
) -> dict[str, Any]:
    in_scope_total = tp_screen + fn_screen
    out_of_scope_total = tn_screen + fp_screen
    retained_total = tp_screen + fp_screen
    rejected_total = tn_screen + fn_screen

    return {
        "watchlist_source": "reference_dir",
        "watchlist_name": "reference_dir_scope",
        "watchlist_classes": watchlist_classes,
        "in_scope_recall": _safe_ratio(tp_screen, in_scope_total),
        "out_of_scope_rejection_rate": _safe_ratio(tn_screen, out_of_scope_total),
        "candidate_purity": _safe_ratio(tp_screen, retained_total),
        "tp_screen": tp_screen,
        "fn_screen": fn_screen,
        "tn_screen": tn_screen,
        "fp_screen": fp_screen,
        "in_scope_total": in_scope_total,
        "out_of_scope_total": out_of_scope_total,
        "retained_total": retained_total,
        "rejected_total": rejected_total,
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def load_threshold_config(
    script_dir: Path,
    similarity_threshold: float,
    l2_distance_threshold: float,
) -> ThresholdConfig:
    default_thresholds = DecisionThreshold(
        similarity_threshold=similarity_threshold,
        l2_distance_threshold=l2_distance_threshold,
    )
    config_path = _resolve_threshold_path(script_dir)
    if not config_path.exists():
        return ThresholdConfig(
            config_path=None,
            default_thresholds=default_thresholds,
            class_thresholds={},
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Threshold config must be a JSON object, but got {type(raw_config).__name__}."
        )

    default_mapping = raw_config.get("default", {})
    if default_mapping and not isinstance(default_mapping, dict):
        raise ValueError("The 'default' field in threshold config must be a JSON object.")
    merged_default = DecisionThreshold.from_mapping(default_mapping, default_thresholds)

    class_entries = raw_config.get("classes", {})
    if class_entries and not isinstance(class_entries, dict):
        raise ValueError("The 'classes' field in threshold config must be a JSON object.")

    class_thresholds: dict[str, DecisionThreshold] = {}
    for class_name, class_mapping in class_entries.items():
        if not isinstance(class_mapping, dict):
            raise ValueError(
                f"Threshold config for class '{class_name}' must be a JSON object."
            )
        normalized_name = str(class_name).strip()
        class_thresholds[normalized_name] = DecisionThreshold.from_mapping(
            class_mapping,
            merged_default,
        )

    return ThresholdConfig(
        config_path=config_path,
        default_thresholds=merged_default,
        class_thresholds=class_thresholds,
    )


def _resolve_threshold_path(script_dir: Path) -> Path:
    config_override = os.environ.get(THRESHOLD_CONFIG_ENV)
    if config_override:
        return Path(config_override).expanduser().resolve()
    return (script_dir / DEFAULT_THRESHOLD_CONFIG_NAME).resolve()


class MultiGPUBatchInferenceModel(BatchInferenceModel):
    def __init__(
        self,
        *args,
        device: torch.device,
        threshold_config: ThresholdConfig,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.model.device = device
        self.model.to(device)
        self.threshold_config = threshold_config

    def _select_best_class(self, class_stats: dict[str, dict[str, float]]) -> str | None:
        best_class: str | None = None
        for class_name, stats in class_stats.items():
            if best_class is None:
                best_class = class_name
                continue

            best_stats = class_stats[best_class]
            if stats["l2_min"] < best_stats["l2_min"]:
                best_class = class_name
            elif stats["l2_min"] == best_stats["l2_min"] and stats["max"] > best_stats["max"]:
                best_class = class_name
        return best_class

    def _apply_decision_thresholds(self, result: dict) -> dict:
        best_class = self._select_best_class(result["class_stats"])
        if best_class is None:
            result["predicted_class"] = "unknown"
            result["similarity"] = 0.0
            result["l2_distance"] = float("inf")
            result["correct"] = result["true_class"].lower() == "unknown"
            result["decision_candidate"] = None
            result["decision_thresholds"] = self.threshold_config.default_thresholds.to_dict()
            return result

        best_stats = result["class_stats"][best_class]
        decision_thresholds = self.threshold_config.for_class(best_class)
        is_known_prediction = (
            best_stats["max"] >= decision_thresholds.similarity_threshold
            and best_stats["l2_min"] <= decision_thresholds.l2_distance_threshold
        )
        predicted_class = best_class if is_known_prediction else "unknown"

        result["predicted_class"] = predicted_class
        result["similarity"] = float(best_stats["max"])
        result["l2_distance"] = float(best_stats["l2_min"])
        result["correct"] = predicted_class.lower() == result["true_class"].lower()
        result["decision_candidate"] = best_class
        result["decision_thresholds"] = decision_thresholds.to_dict()
        return result

    def process_batch(self, batch: dict) -> list[dict]:
        batch_results = super().process_batch(batch)
        return [self._apply_decision_thresholds(result) for result in batch_results]

    def evaluate_distributed(
        self,
        dataloader: DataLoader,
        output_dir: str,
        dist_cfg: DistributedConfig,
    ) -> dict | None:
        if dist_cfg.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "similarity_distributions"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "ensemble_analysis"), exist_ok=True)

        start_time = time.time()
        local_results: list[dict] = []

        iterator = tqdm(
            dataloader,
            desc="Processing batches",
            disable=dist_cfg.distributed and not dist_cfg.is_main_process,
        )
        for batch in iterator:
            local_results.extend(self.process_batch(batch))

        if dist_cfg.distributed:
            gathered_results: list[list[dict] | None] = [None for _ in range(dist_cfg.world_size)]
            dist.all_gather_object(gathered_results, local_results)
            all_results = list(chain.from_iterable(result or [] for result in gathered_results))
            dist.barrier()
        else:
            all_results = local_results

        if not dist_cfg.is_main_process:
            return None

        execution_time = time.time() - start_time
        return self._finalize_results(all_results, output_dir, execution_time)

    def _finalize_results(
        self,
        all_results: list[dict],
        output_dir: str,
        execution_time: float,
    ) -> dict:
        all_results = [
            result_with_screening_metadata(result, self.watchlist_set)
            for result in all_results
        ]
        all_true_labels = [result["true_class"] for result in all_results]
        all_pred_labels = [result["predicted_class"] for result in all_results]
        similarity_scores = [float(result["similarity"]) for result in all_results]
        unknown_count = sum(1 for result in all_results if result["predicted_class"] == "unknown")
        tp_screen = sum(1 for result in all_results if result["screening_outcome"] == "tp_screen")
        fn_screen = sum(1 for result in all_results if result["screening_outcome"] == "fn_screen")
        tn_screen = sum(1 for result in all_results if result["screening_outcome"] == "tn_screen")
        fp_screen = sum(1 for result in all_results if result["screening_outcome"] == "fp_screen")

        class_correct: dict[str, int] = {}
        class_total: dict[str, int] = {}
        for result in all_results:
            true_class = result["true_class"]
            if true_class not in class_correct:
                class_correct[true_class] = 0
                class_total[true_class] = 0
            class_total[true_class] += 1
            if result["correct"]:
                class_correct[true_class] += 1

        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        class_accuracy = {}
        for cls in class_total:
            class_accuracy[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            if cls == "unknown":
                continue
            print(f"Class {cls}: {class_accuracy[cls]:.4f} ({class_correct[cls]}/{class_total[cls]})")

        known_correct = sum(class_correct[cls] for cls in class_total if cls != "unknown")
        known_total = sum(class_total[cls] for cls in class_total if cls != "unknown")
        known_accuracy = known_correct / known_total if known_total > 0 else 0.0
        print(f"Known classes accuracy: {known_accuracy:.4f} ({known_correct}/{known_total})")
        print(
            "Unknown classes accuracy: "
            f"{class_accuracy.get('unknown', 0):.4f} "
            f"({class_correct.get('unknown', 0)}/{class_total.get('unknown', 0)})"
        )
        screening_summary = summarize_screening_counts(
            watchlist_classes=self.watchlist_classes,
            tp_screen=tp_screen,
            fn_screen=fn_screen,
            tn_screen=tn_screen,
            fp_screen=fp_screen,
        )

        confidence_bins = defaultdict(lambda: {"correct": 0, "total": 0})
        for result in all_results:
            confidence_bin = self._build_confidence_bin(result["similarity"])
            confidence_bins[confidence_bin]["total"] += 1
            if result["correct"]:
                confidence_bins[confidence_bin]["correct"] += 1

        for bin_name, bin_data in confidence_bins.items():
            bin_data["accuracy"] = (
                bin_data["correct"] / bin_data["total"] if bin_data["total"] > 0 else 0.0
            )

        class_names = sorted(list(set(all_true_labels)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels,
            all_pred_labels,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=class_names)

        evaluation_results = {
            "accuracy": overall_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "class_accuracy": class_accuracy,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "unknown_count": unknown_count,
            "unknown_percentage": unknown_count / total_samples if total_samples > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
            "confidence_bins": {key: value for key, value in confidence_bins.items()},
            "execution_time": execution_time,
            "mean_similarity": float(np.mean(similarity_scores)) if similarity_scores else 0.0,
            "similarity_threshold": self.similarity_threshold,
            "decision_threshold_config": self.threshold_config.to_dict(),
            "open_world_screening": screening_summary,
        }
        screening_plot_path = create_open_world_screening_visualization(
            screening_summary,
            output_dir,
        )
        evaluation_results["open_world_screening"]["plot_path"] = screening_plot_path

        create_evaluation_visualizations(
            all_results,
            class_accuracy,
            cm,
            class_names,
            confidence_bins,
            output_dir,
            self.similarity_threshold,
        )

        predictions_file = os.path.join(output_dir, "detailed_predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "path": result["path"],
                        "true_class": result["true_class"],
                        "predicted_class": result["predicted_class"],
                        "decision_candidate": result.get("decision_candidate"),
                        "decision_thresholds": result.get("decision_thresholds"),
                        "similarity": result["similarity"],
                        "l2_distance": result["l2_distance"],
                        "correct": result["correct"],
                        "is_true_in_scope": result["is_true_in_scope"],
                        "is_retained": result["is_retained"],
                        "screening_outcome": result["screening_outcome"],
                        "class_stats": {
                            class_name: {
                                "similarity_stats": {
                                    "mean": stats["mean"],
                                    "median": stats["median"],
                                    "variance": stats["variance"],
                                    "min": min(result["full_similarities"][class_name]),
                                    "max": max(result["full_similarities"][class_name]),
                                },
                                "l2_distance_stats": {
                                    "mean": result["all_l2_means"][class_name],
                                    "median": result["all_l2_medians"][class_name],
                                    "variance": result["all_l2_variances"][class_name],
                                    "min": min(result["full_l2_distances"][class_name]),
                                    "max": max(result["full_l2_distances"][class_name]),
                                },
                            }
                            for class_name, stats in result["class_stats"].items()
                        },
                        "all_similarities": result["all_similarities"],
                        "all_l2_distances": result["all_l2_distances"],
                    }
                    for result in all_results
                ],
                f,
                indent=2,
                cls=NumpyEncoder,
            )

        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, cls=NumpyEncoder)

        print("\n===== Evaluation Results =====")
        print(f"Total images: {total_samples}")
        print(f"Correct predictions: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Unknown predictions: {unknown_count} ({evaluation_results['unknown_percentage']:.2%})")
        print(f"Execution time: {execution_time:.2f} seconds")
        print("\n===== Open-World Screening Metrics =====")
        print(
            f"Watchlist classes ({len(self.watchlist_classes)}): "
            + ", ".join(self.watchlist_classes)
        )
        in_scope_recall = screening_summary["in_scope_recall"]
        out_scope_rejection = screening_summary["out_of_scope_rejection_rate"]
        candidate_purity = screening_summary["candidate_purity"]
        in_scope_recall_text = "N/A" if in_scope_recall is None else f"{in_scope_recall:.4f}"
        out_scope_rejection_text = "N/A" if out_scope_rejection is None else f"{out_scope_rejection:.4f}"
        candidate_purity_text = "N/A" if candidate_purity is None else f"{candidate_purity:.4f}"
        print(
            "In-scope Recall: "
            f"{in_scope_recall_text} ({screening_summary['tp_screen']}/{screening_summary['in_scope_total']})"
        )
        print(
            "Out-of-scope Rejection Rate: "
            f"{out_scope_rejection_text} ({screening_summary['tn_screen']}/{screening_summary['out_of_scope_total']})"
        )
        print(
            "Candidate Purity: "
            f"{candidate_purity_text} ({screening_summary['tp_screen']}/{screening_summary['retained_total']})"
        )

        return evaluation_results


def build_dataloader(args: argparse.Namespace, dist_cfg: DistributedConfig) -> DataLoader:
    dataset = InferenceDataset(
        data_dir=args.val_dir,
        transform=transform,
        max_samples_per_class=args.max_samples_per_class,
    )
    sampler = None
    if dist_cfg.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_cfg.world_size,
            rank=dist_cfg.rank,
            shuffle=False,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
    )


def resolve_output_dir(model_path: Path, output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (model_path.parent / "evaluation").resolve()


def run_inference(args: argparse.Namespace) -> None:
    dist_cfg = setup_distributed(args)
    try:
        threshold_config = load_threshold_config(
            script_dir=COARSE_CLASSIFIER_DIR,
            similarity_threshold=args.similarity_threshold,
            l2_distance_threshold=args.l2_distance_threshold,
        )

        if dist_cfg.is_main_process:
            print(f"Using device: {dist_cfg.device}")
            if dist_cfg.distributed:
                print(f"Initialized distributed inference with world size {dist_cfg.world_size}")
            if threshold_config.config_path is not None:
                print(f"Loaded class decision thresholds from {threshold_config.config_path}")
            else:
                print("No class threshold config found. Falling back to global decision thresholds.")

        model_path = Path(args.model_path)
        reference_dir = Path(args.reference_dir)
        val_dir = Path(args.val_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        output_dir = resolve_output_dir(model_path=model_path, output_dir=args.output_dir)
        if dist_cfg.is_main_process:
            print(f"Saving evaluation results to: {output_dir}")

        suppress_non_main = dist_cfg.distributed and not dist_cfg.is_main_process
        with maybe_suppress_output(suppress_non_main):
            batch_model = MultiGPUBatchInferenceModel(
                model_path=str(model_path),
                reference_dir=str(reference_dir),
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                similarity_threshold=args.similarity_threshold,
                max_reference_per_class=args.max_reference_per_class,
                median_threshold=args.median_threshold,
                variance_threshold=args.variance_threshold,
                num_ensembles=args.num_ensembles,
                l2_distance_threshold=args.l2_distance_threshold,
                device=dist_cfg.device,
                threshold_config=threshold_config,
            )
            dataloader = build_dataloader(args, dist_cfg)

        results = batch_model.evaluate_distributed(
            dataloader=dataloader,
            output_dir=str(output_dir),
            dist_cfg=dist_cfg,
        )

        if results and dist_cfg.is_main_process:
            print("\n===== Ensemble Metrics =====")
            if "ensemble_agreement" in results:
                print(f"Average ensemble agreement: {results['ensemble_agreement']:.4f}")
            if "ensemble_diversity" in results:
                print(f"Ensemble diversity: {results['ensemble_diversity']:.4f}")
    finally:
        if dist_cfg.distributed and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
