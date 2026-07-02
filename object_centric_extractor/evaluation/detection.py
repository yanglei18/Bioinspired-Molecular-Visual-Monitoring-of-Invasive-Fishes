"""Detection AP orchestration for SAM2 detector evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from evaluation.kitti_eval import evaluate_directory
    from evaluation.reporting import (
        FISH_EVAL_CLASS,
        aggregate_detection_summaries,
        render_text_table,
        summarize_gt_conversion,
        summarize_prediction_conversion,
    )
    from evaluation.to_kitti_format_gt import collect_gt_sequence_names, convert_directory_to_kitti
    from evaluation.to_kitti_format_pred import convert_to_kitti_format, export_prediction_details
    from utils.fish_label_map import FINE_CLASSES
except ModuleNotFoundError:
    from object_centric_extractor.evaluation.kitti_eval import evaluate_directory
    from object_centric_extractor.evaluation.reporting import (
        FISH_EVAL_CLASS,
        aggregate_detection_summaries,
        render_text_table,
        summarize_gt_conversion,
        summarize_prediction_conversion,
    )
    from object_centric_extractor.evaluation.to_kitti_format_gt import collect_gt_sequence_names, convert_directory_to_kitti
    from object_centric_extractor.evaluation.to_kitti_format_pred import convert_to_kitti_format, export_prediction_details
    from object_centric_extractor.utils.fish_label_map import FINE_CLASSES


def run_detection_coarse(
    pred_json_dir: Path,
    gt_label_dir: Path,
    work_dir: Path,
    tracker_name: str,
    no_plot: bool,
    verbose: bool,
    frame_align_mode: str,
) -> dict[str, Any]:
    gt_root = work_dir / "gt"
    gt_label_02 = gt_root / "label_02"
    pred_root = work_dir / "pred"
    pred_data = pred_root / tracker_name / "data"
    plot_path = work_dir / "2d_result.png"
    frame_metrics_path = work_dir / "detection_frame_map.csv"
    summary_path = work_dir / "detection_summary.json"
    prediction_export_path = work_dir / "detection_predictions_with_class.csv"
    gt_sequence_names = collect_gt_sequence_names(gt_label_dir)

    print(f"[1/4] Convert predictions: {pred_json_dir} -> {pred_data}")
    pred_summary = convert_to_kitti_format(
        pred_json_dir,
        pred_data,
        allowed_sequence_names=gt_sequence_names,
        verbose=verbose,
    )
    print(f"Converted {len(pred_summary)} prediction sequences.")
    prediction_export = export_prediction_details(
        pred_json_dir,
        prediction_export_path,
        allowed_sequence_names=gt_sequence_names,
    )
    pred_conversion_stats = summarize_prediction_conversion(pred_summary)
    print(
        "Prediction conversion stats: "
        f"{pred_conversion_stats['total_boxes']} boxes, "
        f"{pred_conversion_stats['non_empty_file_count']}/{pred_conversion_stats['file_count']} non-empty files"
    )

    print(f"[2/4] Convert GT labels: {gt_label_dir} -> {gt_label_02}")
    gt_summary = convert_directory_to_kitti(gt_label_dir, gt_label_02, count_only=False, verbose=verbose)
    print(f"Converted {gt_summary['file_count']} GT files.")
    gt_conversion_stats = summarize_gt_conversion(gt_summary)
    print(
        "GT conversion stats: "
        f"{gt_conversion_stats['total_boxes']} boxes, "
        f"{gt_conversion_stats['non_empty_file_count']}/{gt_conversion_stats['file_count']} non-empty files"
    )

    print(f"[3/4] Run detection AP evaluation: {gt_label_02} vs {pred_data}")
    summary = evaluate_directory(
        str(gt_label_02),
        str(pred_data),
        FISH_EVAL_CLASS,
        frame_align_mode=frame_align_mode,
        plot=not no_plot,
        plot_path=str(plot_path),
        frame_metrics_path=str(frame_metrics_path),
        summary_path=str(summary_path),
        verbose=verbose,
        terminal_label="coarse",
    )
    if summary is None:
        print("Detection AP evaluation skipped because no matched data was found.")
        status = "no_matched_data"
    else:
        print("Detection AP summary:")
        print(
            render_text_table(
                headers=["Class", "Easy AP41", "Moderate AP41", "Hard AP41", "Total mAP41"],
                rows=[[
                    FISH_EVAL_CLASS,
                    f"{summary.get('easy_ap41', 0.0):.4f}",
                    f"{summary.get('moderate_ap41', 0.0):.4f}",
                    f"{summary.get('hard_ap41', 0.0):.4f}",
                    f"{summary.get('total_map41', 0.0):.4f}",
                ]],
            )
        )
        status = "completed"

    return {
        "mode": "coarse",
        "status": status,
        "frame_align_mode": frame_align_mode,
        "summary": summary,
        "prediction_conversion": pred_summary,
        "prediction_export": prediction_export,
        "gt_conversion": gt_summary,
        "paths": {
            "prediction_kitti_dir": str(pred_data),
            "prediction_export_path": str(prediction_export_path),
            "gt_kitti_dir": str(gt_label_02),
            "frame_metrics_path": str(frame_metrics_path),
            "summary_path": str(summary_path),
            "plot_path": str(plot_path),
        },
    }


def run_detection_fine(
    pred_json_dir: Path,
    gt_label_dir: Path,
    work_dir: Path,
    tracker_name: str,
    no_plot: bool,
    verbose: bool,
    frame_align_mode: str,
) -> dict[str, Any]:
    fine_root = work_dir / "detection_fine"
    per_class: dict[str, dict[str, Any]] = {}
    completed_summaries: list[dict[str, Any]] = []
    gt_sequence_names = collect_gt_sequence_names(gt_label_dir)

    print(f"[1/4] Run fine-grained detection evaluation under {fine_root}")
    for class_name in FINE_CLASSES:
        class_root = fine_root / class_name
        gt_label_02 = class_root / "gt" / "label_02"
        pred_data = class_root / "pred" / tracker_name / "data"
        plot_path = class_root / "2d_result.png"
        frame_metrics_path = class_root / "detection_frame_map.csv"
        summary_path = class_root / "detection_summary.json"
        prediction_export_path = class_root / "detection_predictions_with_class.csv"

        if verbose:
            print(f"  [fine] Converting predictions for {class_name}")
        pred_summary = convert_to_kitti_format(
            pred_json_dir,
            pred_data,
            target_class=class_name,
            allowed_sequence_names=gt_sequence_names,
            verbose=verbose,
        )
        prediction_export = export_prediction_details(
            pred_json_dir,
            prediction_export_path,
            target_class=class_name,
            allowed_sequence_names=gt_sequence_names,
        )
        pred_conversion_stats = summarize_prediction_conversion(pred_summary)
        if verbose:
            print(
                f"  [fine] Prediction stats for {class_name}: "
                f"{pred_conversion_stats['total_boxes']} boxes, "
                f"{pred_conversion_stats['non_empty_file_count']}/{pred_conversion_stats['file_count']} non-empty files"
            )

        if verbose:
            print(f"  [fine] Converting GT labels for {class_name}")
        gt_summary = convert_directory_to_kitti(
            gt_label_dir,
            gt_label_02,
            count_only=False,
            target_class=class_name,
            verbose=verbose,
        )
        gt_conversion_stats = summarize_gt_conversion(gt_summary)
        if verbose:
            print(
                f"  [fine] GT stats for {class_name}: "
                f"{gt_conversion_stats['total_boxes']} boxes, "
                f"{gt_conversion_stats['non_empty_file_count']}/{gt_conversion_stats['file_count']} non-empty files"
            )

        if verbose:
            print(f"  [fine] Evaluating detection AP for {class_name}")
        summary = evaluate_directory(
            str(gt_label_02),
            str(pred_data),
            FISH_EVAL_CLASS,
            frame_align_mode=frame_align_mode,
            plot=not no_plot,
            plot_path=str(plot_path),
            frame_metrics_path=str(frame_metrics_path),
            summary_path=str(summary_path),
            verbose=verbose,
            terminal_label=f"fine:{class_name}",
        )

        status = "completed" if summary is not None else "no_matched_data"
        if summary is not None:
            completed_summaries.append(summary)

        per_class[class_name] = {
            "status": status,
            "summary": summary,
            "prediction_conversion": pred_summary,
            "prediction_export": prediction_export,
            "prediction_conversion_stats": pred_conversion_stats,
            "gt_conversion": gt_summary,
            "gt_conversion_stats": gt_conversion_stats,
            "paths": {
                "prediction_kitti_dir": str(pred_data),
                "prediction_export_path": str(prediction_export_path),
                "gt_kitti_dir": str(gt_label_02),
                "frame_metrics_path": str(frame_metrics_path),
                "summary_path": str(summary_path),
                "plot_path": str(plot_path),
            },
        }

    macro_summary = aggregate_detection_summaries(completed_summaries)
    print("Fine Detection AP summary:")
    macro_status = "completed" if completed_summaries else "no_matched_data"
    print(f"  Summary: {macro_status}, Macro Total mAP41 = {macro_summary.get('total_map41', 0.0):.4f}")
    fine_rows: list[list[str]] = []
    for class_name in FINE_CLASSES:
        class_result = per_class.get(class_name, {})
        summary = class_result.get("summary")
        status = class_result.get("status", "pending")
        if summary is None:
            fine_rows.append([class_name, status, "-"])
            continue
        fine_rows.append([class_name, status, f"{summary.get('total_map41', 0.0):.4f}"])
    print(render_text_table(headers=["Class", "Status", "mAP41"], rows=fine_rows))
    return {
        "mode": "fine",
        "status": "completed" if completed_summaries else "no_matched_data",
        "frame_align_mode": frame_align_mode,
        "macro_summary": macro_summary,
        "per_class": per_class,
        "paths": {
            "fine_root": str(fine_root),
        },
    }
