from __future__ import annotations

import os
import re
import tempfile
from typing import Any

import numpy as np


CONFIDENCE_BIN_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$")


def create_open_world_screening_visualization(
    screening_summary: dict[str, Any],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(tempfile.gettempdir(), "matplotlib-open-world-screening")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    import matplotlib.pyplot as plt

    output_path = os.path.join(output_dir, "open_world_screening_metrics.png")

    metric_names = [
        "In-scope Recall",
        "Out-of-scope Rejection Rate",
        "Candidate Purity",
    ]
    metric_values = [
        _value_or_zero(screening_summary.get("in_scope_recall")),
        _value_or_zero(screening_summary.get("out_of_scope_rejection_rate")),
        _value_or_zero(screening_summary.get("candidate_purity")),
    ]
    metric_labels = [
        _format_metric(screening_summary.get("in_scope_recall")),
        _format_metric(screening_summary.get("out_of_scope_rejection_rate")),
        _format_metric(screening_summary.get("candidate_purity")),
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Open-World Screening Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, label in zip(bars, metric_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_evaluation_visualizations(
    results: list[dict[str, Any]],
    class_accuracy: dict[str, float],
    confusion_matrix: np.ndarray,
    class_names: list[str],
    confidence_bins: dict[str, dict[str, float]],
    output_dir: str,
    similarity_threshold: float,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 10))
    cm_normalized = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    sorted_classes = sorted(class_accuracy.keys(), key=lambda item: class_accuracy[item], reverse=True)
    accuracies = [class_accuracy[class_name] for class_name in sorted_classes]
    class_totals = {
        class_name: sum(1 for item in results if item["true_class"] == class_name)
        for class_name in sorted_classes
    }
    error_margins = [
        1.96 * np.sqrt((accuracy * (1 - accuracy)) / class_totals[class_name])
        if class_totals[class_name] > 0
        else 0.0
        for accuracy, class_name in zip(accuracies, sorted_classes)
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(sorted_classes)), accuracies, yerr=error_margins)
    plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy with 95% Confidence Intervals")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_accuracy.png"))
    plt.close()

    correct_predictions = [result["similarity"] for result in results if result["correct"]]
    incorrect_predictions = [result["similarity"] for result in results if not result["correct"]]

    plt.figure(figsize=(10, 6))
    plt.hist(
        [correct_predictions, incorrect_predictions],
        bins=20,
        alpha=0.7,
        label=["Correct", "Incorrect"],
    )
    if correct_predictions:
        plt.axvline(
            np.mean(correct_predictions),
            color="g",
            linestyle="--",
            label=f"Mean correct: {np.mean(correct_predictions):.3f}",
        )
    if incorrect_predictions:
        plt.axvline(
            np.mean(incorrect_predictions),
            color="r",
            linestyle="--",
            label=f"Mean incorrect: {np.mean(incorrect_predictions):.3f}",
        )
    plt.axvline(
        similarity_threshold,
        color="black",
        linestyle="-",
        label=f"Threshold: {similarity_threshold:.3f}",
    )
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.title("Distribution of Similarity Scores")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_distribution.png"))
    plt.close()

    _create_calibration_plot(confidence_bins, output_dir, similarity_threshold)
    _create_confused_pairs_plot(confusion_matrix, class_names, output_dir)
    _create_unknown_distribution_plot(results, output_dir)


def _create_calibration_plot(
    confidence_bins: dict[str, dict[str, float]],
    output_dir: str,
    similarity_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    sortable_bins = []
    for bin_name, bin_data in confidence_bins.items():
        parsed_bin = _parse_confidence_bin(bin_name)
        if parsed_bin is None:
            continue
        sortable_bins.append((parsed_bin[0], parsed_bin[1], bin_data))

    for bin_start, bin_end, bin_data in sorted(sortable_bins, key=lambda item: item[0]):
        if bin_data["total"] > 0:
            bin_accuracies.append(bin_data["accuracy"])
            bin_confidences.append((bin_start + bin_end) / 2)
            bin_sizes.append(bin_data["total"])

    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.scatter(bin_confidences, bin_accuracies, s=[size / 5 for size in bin_sizes], alpha=0.8)
    plt.plot(bin_confidences, bin_accuracies, "o-", label="Model calibration")
    for x_value, y_value, size in zip(bin_confidences, bin_accuracies, bin_sizes):
        plt.text(x_value, y_value, f"{size}", fontsize=9, ha="center", va="bottom")
    plt.axvline(
        similarity_threshold,
        color="red",
        linestyle="--",
        label=f"Unknown threshold: {similarity_threshold}",
    )
    plt.xlabel("Confidence (Predicted Similarity)")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
    plt.close()


def _create_confused_pairs_plot(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    class_confusion = {}
    for row_idx, true_class in enumerate(class_names):
        for col_idx, pred_class in enumerate(class_names):
            if row_idx != col_idx and confusion_matrix[row_idx, col_idx] > 0:
                class_confusion[(true_class, pred_class)] = confusion_matrix[row_idx, col_idx]

    top_confused = sorted(class_confusion.items(), key=lambda item: item[1], reverse=True)[:10]
    if not top_confused:
        return

    pair_labels = [f"{true} -> {pred}" for (true, pred), _ in top_confused]
    confusion_counts = [count for _, count in top_confused]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(pair_labels)), confusion_counts, color="salmon")
    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel("Count")
    plt.ylabel("True -> Predicted")
    plt.title("Top Most Confused Class Pairs")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "most_confused_pairs.png"))
    plt.close()


def _create_unknown_distribution_plot(
    results: list[dict[str, Any]],
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    unknown_preds = [result for result in results if result["predicted_class"] == "unknown"]
    if not unknown_preds:
        return

    unknown_true_classes: dict[str, int] = {}
    for result in unknown_preds:
        true_class = result["true_class"]
        unknown_true_classes[true_class] = unknown_true_classes.get(true_class, 0) + 1

    sorted_unknown = sorted(unknown_true_classes.items(), key=lambda item: item[1], reverse=True)
    class_labels = [class_name for class_name, _ in sorted_unknown]
    class_counts = [count for _, count in sorted_unknown]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(class_labels)), class_counts, color="lightblue")
    plt.yticks(range(len(class_labels)), class_labels)
    plt.xlabel("Count")
    plt.ylabel("True Class")
    plt.title("Distribution of True Classes in Unknown Predictions")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unknown_class_distribution.png"))
    plt.close()


def _parse_confidence_bin(bin_name: str) -> tuple[float, float] | None:
    match = CONFIDENCE_BIN_PATTERN.match(bin_name)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _value_or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"
