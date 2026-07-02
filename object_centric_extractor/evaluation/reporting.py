"""Reporting helpers for the SAM2 detector evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from utils.fish_label_map import FINE_CLASSES
except ModuleNotFoundError:
    from object_centric_extractor.utils.fish_label_map import FINE_CLASSES

FISH_EVAL_CLASS = "Fish"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)


def write_text_report(report_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text.rstrip() + "\n", encoding="utf-8")


def average_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def summarize_prediction_conversion(pred_summary: dict[str, int]) -> dict[str, int]:
    return {
        "total_boxes": int(sum(pred_summary.values())),
        "non_empty_file_count": int(sum(1 for count in pred_summary.values() if count > 0)),
        "file_count": int(len(pred_summary)),
    }


def summarize_gt_conversion(gt_summary: dict[str, Any]) -> dict[str, int]:
    file_detection_counts = gt_summary.get("file_detection_counts", {})
    return {
        "total_boxes": int(gt_summary.get("total_objects", 0)),
        "non_empty_file_count": int(sum(1 for count in file_detection_counts.values() if count > 0)),
        "file_count": int(gt_summary.get("file_count", 0)),
    }


def render_text_table(headers: list[str], rows: list[list[str]]) -> str:
    table_rows = [headers, *rows]
    col_widths = [
        max(len(str(row[col_index])) for row in table_rows)
        for col_index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return "  " + " | ".join(
            str(cell).ljust(col_widths[col_index])
            for col_index, cell in enumerate(row)
        )

    separator = "  " + "-+-".join("-" * width for width in col_widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def aggregate_detection_summaries(completed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not completed_summaries:
        return {
            "easy_ap": 0.0,
            "moderate_ap": 0.0,
            "hard_ap": 0.0,
            "total_map": 0.0,
            "easy_ap11": 0.0,
            "moderate_ap11": 0.0,
            "hard_ap11": 0.0,
            "total_map11": 0.0,
            "easy_ap41": 0.0,
            "moderate_ap41": 0.0,
            "hard_ap41": 0.0,
            "total_map41": 0.0,
            "evaluated_class_count": 0,
        }

    return {
        "easy_ap": average_metric([summary["easy_ap"] for summary in completed_summaries]),
        "moderate_ap": average_metric([summary["moderate_ap"] for summary in completed_summaries]),
        "hard_ap": average_metric([summary["hard_ap"] for summary in completed_summaries]),
        "total_map": average_metric([summary["total_map"] for summary in completed_summaries]),
        "easy_ap11": average_metric([summary.get("easy_ap11", summary["easy_ap"]) for summary in completed_summaries]),
        "moderate_ap11": average_metric(
            [summary.get("moderate_ap11", summary["moderate_ap"]) for summary in completed_summaries]
        ),
        "hard_ap11": average_metric([summary.get("hard_ap11", summary["hard_ap"]) for summary in completed_summaries]),
        "total_map11": average_metric(
            [summary.get("total_map11", summary["total_map"]) for summary in completed_summaries]
        ),
        "easy_ap41": average_metric([summary.get("easy_ap41", 0.0) for summary in completed_summaries]),
        "moderate_ap41": average_metric([summary.get("moderate_ap41", 0.0) for summary in completed_summaries]),
        "hard_ap41": average_metric([summary.get("hard_ap41", 0.0) for summary in completed_summaries]),
        "total_map41": average_metric([summary.get("total_map41", 0.0) for summary in completed_summaries]),
        "evaluated_class_count": len(completed_summaries),
    }


def build_detection_report_text(det_result: dict[str, Any] | None) -> str:
    if det_result is None:
        return "Detection\nNo detection evaluation was run.\n"

    frame_align_mode = det_result.get("frame_align_mode", "unknown")

    if det_result.get("status") == "skipped":
        return (
            "Detection\n"
            f"Mode                  {det_result.get('mode', 'coarse')}\n"
            f"Frame Align           {frame_align_mode}\n"
            "Skipped\n"
        )

    mode = det_result.get("mode", "coarse")
    if mode == "coarse":
        summary = det_result.get("summary")
        if summary is None:
            return (
                "Detection\n"
                "Mode                  coarse\n"
                f"Frame Align           {frame_align_mode}\n"
                "No matched data was found.\n"
            )

        lines = [
            "Detection",
            "Mode                  coarse",
            f"Frame Align           {frame_align_mode}",
            "Metric                Value",
            f"Class                 {FISH_EVAL_CLASS}",
            f"Easy AP41             {summary.get('easy_ap41', 0.0):.3f}",
            f"Moderate AP41         {summary.get('moderate_ap41', 0.0):.3f}",
            f"Hard AP41             {summary.get('hard_ap41', 0.0):.3f}",
            f"mAP41                 {summary.get('total_map41', 0.0):.3f}",
            f"Frames                {summary['frame_count']}",
            f"Files                 {summary['file_count']}",
        ]
        prediction_export = det_result.get("prediction_export")
        if prediction_export is not None:
            lines.append(f"Pred Boxes             {prediction_export['row_count']}")
            lines.append(f"Pred CSV              {prediction_export['output_path']}")
        if summary.get("frame_metrics_path"):
            lines.append(f"Frame Metrics         {summary['frame_metrics_path']}")
        if summary.get("summary_path"):
            lines.append(f"Summary JSON          {summary['summary_path']}")
        return "\n".join(lines) + "\n"

    macro_summary = det_result.get("macro_summary", {})
    lines = [
        "Detection",
        "Mode                  fine",
        f"Frame Align           {frame_align_mode}",
        f"Evaluated Classes     {macro_summary.get('evaluated_class_count', 0)}",
        f"Macro Easy AP41       {macro_summary.get('easy_ap41', 0.0):.3f}",
        f"Macro Moderate AP41   {macro_summary.get('moderate_ap41', 0.0):.3f}",
        f"Macro Hard AP41       {macro_summary.get('hard_ap41', 0.0):.3f}",
        f"Macro mAP41           {macro_summary.get('total_map41', 0.0):.3f}",
        "",
        "Per-class Detection",
        "Class                 Status           PredBoxes   GTBoxes    mAP41",
    ]
    for class_name in FINE_CLASSES:
        class_result = det_result.get("per_class", {}).get(class_name, {})
        status = class_result.get("status", "pending")
        summary = class_result.get("summary")
        prediction_export = class_result.get("prediction_export", {})
        gt_conversion_stats = class_result.get("gt_conversion_stats", {})
        pred_boxes = int(prediction_export.get("row_count", 0))
        gt_boxes = int(gt_conversion_stats.get("total_boxes", 0))
        if summary is None:
            lines.append(f"{class_name:<21} {status:<15} {pred_boxes:>9} {gt_boxes:>8} {'-':>8}")
        else:
            lines.append(
                f"{class_name:<21} {status:<15} {pred_boxes:>9} {gt_boxes:>8} "
                f"{summary.get('total_map41', 0.0):>8.3f}"
            )
    return "\n".join(lines) + "\n"


def build_sequence_coverage_report_text(sequence_coverage: dict[str, Any] | None) -> str:
    if sequence_coverage is None:
        return ""

    lines = [
        "Sequence Coverage",
        f"Prediction Sequences   {sequence_coverage['prediction_sequence_count']}",
        f"GT Sequences           {sequence_coverage['gt_sequence_count']}",
        f"Missing GT Sequences   {sequence_coverage['missing_gt_sequence_count']}",
    ]
    if sequence_coverage["missing_gt_sequences"]:
        lines.append("Prediction Sequences Without GT")
        for sequence_name in sequence_coverage["missing_gt_sequences"]:
            lines.append(f"  {sequence_name}")
    return "\n".join(lines) + "\n"


def build_combined_text_report(
    det_result: dict[str, Any] | None,
    tracking_report: dict[str, Any],
    sequence_coverage: dict[str, Any] | None = None,
) -> str:
    sections = [build_detection_report_text(det_result)]

    if tracking_report["status"] == "completed":
        sections.append("Tracking")
        frame_align_mode = tracking_report.get("frame_align_mode")
        if frame_align_mode:
            sections.append(f"Frame Align           {frame_align_mode}")
        rendered_output = tracking_report.get("rendered_output", "").strip()
        if rendered_output:
            sections.append(rendered_output)
        else:
            sections.append("Tracking completed, but no rendered table text was captured.")
    elif tracking_report["status"] == "skipped":
        sections.append("Tracking")
        sections.append(f"Skipped: {tracking_report.get('reason', 'disabled by user')}")
    else:
        sections.append("Tracking")
        sections.append("Pending")

    coverage_section = build_sequence_coverage_report_text(sequence_coverage)
    if coverage_section:
        sections.append(coverage_section.rstrip())

    return "\n\n".join(section for section in sections if section).strip() + "\n"
