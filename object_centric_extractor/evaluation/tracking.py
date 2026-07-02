"""Tracking evaluation orchestration for SAM2 detector outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from evaluation.reporting import to_jsonable
    from evaluation.to_kitti_format_gt import (
        collect_gt_frame_ids_by_sequence,
        collect_gt_sequence_names,
        convert_directory_to_kitti,
    )
    from evaluation.to_kitti_format_pred import convert_to_kitti_format
except ModuleNotFoundError:
    from object_centric_extractor.evaluation.reporting import to_jsonable
    from object_centric_extractor.evaluation.to_kitti_format_gt import (
        collect_gt_frame_ids_by_sequence,
        collect_gt_sequence_names,
        convert_directory_to_kitti,
    )
    from object_centric_extractor.evaluation.to_kitti_format_pred import convert_to_kitti_format

FISH_TRACKING_CLASS = "fish"


def build_tracking_frame_filters(
    gt_label_dir: Path,
    frame_align_mode: str,
) -> tuple[dict[str, set[int]] | None, dict[str, tuple[int, int]] | None]:
    if frame_align_mode == "union":
        return None, None

    gt_frame_ids_by_sequence = collect_gt_frame_ids_by_sequence(gt_label_dir)
    if frame_align_mode == "gt_present":
        return gt_frame_ids_by_sequence, None

    gt_frame_ranges_by_sequence: dict[str, tuple[int, int]] = {}
    for sequence_name, frame_ids in gt_frame_ids_by_sequence.items():
        if not frame_ids:
            continue
        gt_frame_ranges_by_sequence[sequence_name] = (min(frame_ids), max(frame_ids))
    return None, gt_frame_ranges_by_sequence


def _load_tracking_eval() -> tuple[Any | None, str | None]:
    try:
        from evaluation.run_kitti import run_tracking_eval
    except ImportError as first_exc:
        try:
            from object_centric_extractor.evaluation.run_kitti import run_tracking_eval
        except ImportError as second_exc:
            return None, f"failed to import TrackEval dependency ({second_exc or first_exc})"
    return run_tracking_eval, None


def run_tracking(
    pred_json_dir: Path,
    gt_label_dir: Path,
    work_dir: Path,
    tracker_name: str,
    verbose: bool,
    frame_align_mode: str,
) -> dict[str, Any]:
    gt_root = work_dir / "gt"
    gt_label_02 = gt_root / "label_02"
    pred_root = work_dir / "pred"
    pred_data = pred_root / tracker_name / "data"
    tracking_text_report_path = work_dir / "tracking_report.txt"
    tracking_output = work_dir / "tracking_results"
    gt_sequence_names = collect_gt_sequence_names(gt_label_dir)
    allowed_frame_ids_by_sequence, allowed_frame_ranges_by_sequence = build_tracking_frame_filters(
        gt_label_dir,
        frame_align_mode,
    )

    print(f"Tracking frame alignment mode: {frame_align_mode}")
    print(f"[4/4] Convert tracking predictions: {pred_json_dir} -> {pred_data}")
    convert_to_kitti_format(
        pred_json_dir,
        pred_data,
        allowed_sequence_names=gt_sequence_names,
        allowed_frame_ids_by_sequence=allowed_frame_ids_by_sequence,
        allowed_frame_ranges_by_sequence=allowed_frame_ranges_by_sequence,
        verbose=verbose,
    )

    print(f"[4/4] Convert tracking GT labels: {gt_label_dir} -> {gt_label_02}")
    convert_directory_to_kitti(gt_label_dir, gt_label_02, count_only=False, verbose=verbose)

    print(f"[4/4] Run tracking evaluation under {pred_root}")
    run_tracking_eval, import_error = _load_tracking_eval()
    if run_tracking_eval is None:
        reason = import_error or "failed to import TrackEval dependency"
        print(f"Tracking evaluation skipped: {reason}.")
        return {
            "status": "skipped",
            "reason": reason,
            "output_dir": str(tracking_output),
        }

    tracking_result = run_tracking_eval(
        gt_folder=str(gt_root),
        trackers_folder=str(pred_root),
        output_folder=str(tracking_output),
        trackers_to_eval=[tracker_name],
        metrics=["HOTA"],
        classes_to_eval=[FISH_TRACKING_CLASS],
        rendered_output_path=str(tracking_text_report_path),
        return_rendered_output=True,
    )
    print(f"Tracking evaluation outputs saved under {tracking_output}")
    return {
        "status": "completed",
        "frame_align_mode": frame_align_mode,
        "output_dir": str(tracking_output),
        "raw_result": to_jsonable(tracking_result["result"]),
        "rendered_output": tracking_result["rendered_output"],
        "rendered_output_path": tracking_result["rendered_output_path"],
        "paths": {
            "prediction_kitti_dir": str(pred_data),
            "gt_kitti_dir": str(gt_label_02),
        },
    }
