"""One-command evaluation pipeline for SAM2 detector outputs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from evaluation.detection import run_detection_coarse, run_detection_fine
    from evaluation.tracking import build_tracking_frame_filters, run_tracking
    from evaluation.to_kitti_format_gt import collect_gt_sequence_names
    from utils.annotation_update import UpdateConfig, run_update
    from utils.annotation_io import get_existing_aggregated_annotation_path, load_prediction_sequences
    from utils.fish_label_map import LABEL_ID_TO_NAME
    from evaluation.reporting import (
        build_combined_text_report,
        to_jsonable,
        write_json_report,
        write_text_report,
    )
except ModuleNotFoundError:
    from object_centric_extractor.evaluation.detection import run_detection_coarse, run_detection_fine
    from object_centric_extractor.evaluation.tracking import build_tracking_frame_filters, run_tracking
    from object_centric_extractor.evaluation.to_kitti_format_gt import collect_gt_sequence_names
    from object_centric_extractor.utils.annotation_update import UpdateConfig, run_update
    from object_centric_extractor.utils.annotation_io import get_existing_aggregated_annotation_path, load_prediction_sequences
    from object_centric_extractor.utils.fish_label_map import LABEL_ID_TO_NAME
    from object_centric_extractor.evaluation.reporting import (
        build_combined_text_report,
        to_jsonable,
        write_json_report,
        write_text_report,
    )

FRAME_ALIGN_MODE_OPTIONS = ("auto", "union", "gt_present", "gt_range")
WORK_SUBDIRS_TO_CLEAR = ("gt", "pred", "detection_fine", "tracking_results")


@dataclass(frozen=True)
class EvaluationPipelineConfig:
    pred_json_dir: str | Path
    gt_label_dir: str | Path
    work_dir: str | Path
    inference_json: str | Path | None = None
    tracker_name: str = "SAM2Detector"
    skip_detection: bool = False
    skip_tracking: bool = False
    no_plot: bool = False
    verbose: bool = False
    frame_align_mode: str = "auto"


def clear_work_subdirs(work_dir: Path) -> list[str]:
    cleared_dirs: list[str] = []
    for subdir_name in WORK_SUBDIRS_TO_CLEAR:
        target_dir = work_dir / subdir_name
        if not target_dir.exists():
            continue
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise ValueError(
                f"Expected {target_dir} to be a normal directory before cleanup, "
                "but found a non-directory path."
            )
        shutil.rmtree(target_dir)
        cleared_dirs.append(str(target_dir))
    return cleared_dirs


def resolve_frame_align_mode(gt_label_dir: Path, frame_align_mode: str) -> str:
    if frame_align_mode != "auto":
        return frame_align_mode
    return "gt_present"


def validate_json_gt_dir(gt_label_dir: Path) -> dict[str, dict]:
    sequences = load_prediction_sequences(gt_label_dir)
    if sequences:
        return sequences

    aggregated_path = get_existing_aggregated_annotation_path(gt_label_dir)
    if aggregated_path is not None:
        raise FileNotFoundError(
            f"GT JSON annotations were found at {aggregated_path}, but no valid sequences could be loaded."
        )

    txt_files = []
    if gt_label_dir.exists() and gt_label_dir.is_dir():
        txt_files = sorted(path.name for path in gt_label_dir.iterdir() if path.is_file() and path.suffix == ".txt")

    if txt_files:
        raise ValueError(
            "evaluate_pipeline.py now only supports annotation_det-style JSON GT. "
            f"Found txt GT files under {gt_label_dir}: {', '.join(txt_files[:5])}"
            + (" ..." if len(txt_files) > 5 else "")
        )

    raise FileNotFoundError(
        f"No supported GT JSON annotations found under {gt_label_dir}. "
        "Expected prediction.json, a single sequence JSON, or legacy per-frame JSON directories."
    )


def analyze_prediction_gt_coverage(pred_json_dir: Path, gt_label_dir: Path) -> dict:
    prediction_sequence_names = {
        Path(sequence_key).name
        for sequence_key in load_prediction_sequences(pred_json_dir).keys()
    }
    gt_sequence_names = collect_gt_sequence_names(gt_label_dir)
    missing_gt_sequences = sorted(prediction_sequence_names - gt_sequence_names)
    return {
        "prediction_sequence_count": len(prediction_sequence_names),
        "gt_sequence_count": len(gt_sequence_names),
        "missing_gt_sequences": missing_gt_sequences,
        "missing_gt_sequence_count": len(missing_gt_sequences),
    }


def print_prediction_gt_coverage(coverage: dict) -> None:
    print("Prediction / GT sequence coverage")
    print(f"  Prediction sequences: {coverage['prediction_sequence_count']}")
    print(f"  GT sequences: {coverage['gt_sequence_count']}")
    print(f"  Prediction sequences without GT: {coverage['missing_gt_sequence_count']}")
    if coverage["missing_gt_sequences"]:
        for sequence_name in coverage["missing_gt_sequences"]:
            print(f"    {sequence_name}")


def _resolve_repo_relative_path(path: str | Path, repo_root: Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return repo_root / resolved_path


def run_evaluation_pipeline(config: EvaluationPipelineConfig, repo_root: Path | None = None) -> dict:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    pred_json_dir = _resolve_repo_relative_path(config.pred_json_dir, repo_root)
    gt_label_dir = _resolve_repo_relative_path(config.gt_label_dir, repo_root)
    inference_json = (
        _resolve_repo_relative_path(config.inference_json, repo_root)
        if config.inference_json is not None
        else None
    )
    if not pred_json_dir.is_absolute():
        pred_json_dir = repo_root / pred_json_dir
    if not gt_label_dir.is_absolute():
        gt_label_dir = repo_root / gt_label_dir
    if inference_json is not None and not inference_json.is_absolute():
        inference_json = repo_root / inference_json

    validate_json_gt_dir(gt_label_dir)

    work_dir = _resolve_repo_relative_path(config.work_dir, repo_root)

    cleared_dirs = clear_work_subdirs(work_dir)
    if cleared_dirs:
        print("Cleared previous evaluation artifacts:")
        for cleared_dir in cleared_dirs:
            print(f"  {cleared_dir}")
    else:
        print(f"No previous evaluation artifacts found under {work_dir}")

    combined_report_path = work_dir / "evaluation_report.json"
    combined_text_report_path = work_dir / "evaluation_report.txt"
    prediction_update_summary_path = work_dir / "prediction_update_summary.json"

    prediction_update_report: dict | None = None
    detection_mode = "fine" if inference_json is not None else "coarse"
    if inference_json is not None:
        print(f"Updating prediction annotations from inference JSON: {inference_json}")
        prediction_update_report = run_update(
            UpdateConfig(
                inference_json=inference_json,
                annotation_det_dir=pred_json_dir,
                dry_run=False,
                summary_path=prediction_update_summary_path,
            )
        )
        print(f"Detection mode: {detection_mode} (updated prediction annotations)")
    else:
        print(f"Detection mode: {detection_mode} (no inference JSON provided)")

    sequence_coverage = analyze_prediction_gt_coverage(pred_json_dir, gt_label_dir)
    print_prediction_gt_coverage(sequence_coverage)

    det_result: dict | None = None
    frame_align_mode = resolve_frame_align_mode(gt_label_dir, config.frame_align_mode)
    print(f"Detection frame alignment mode: {frame_align_mode}")
    if config.skip_detection:
        det_result = {
            "mode": detection_mode,
            "status": "skipped",
            "frame_align_mode": frame_align_mode,
        }
    elif detection_mode == "fine":
        det_result = run_detection_fine(
            pred_json_dir=pred_json_dir,
            gt_label_dir=gt_label_dir,
            work_dir=work_dir,
            tracker_name=config.tracker_name,
            no_plot=config.no_plot,
            verbose=config.verbose,
            frame_align_mode=frame_align_mode,
        )
    else:
        det_result = run_detection_coarse(
            pred_json_dir=pred_json_dir,
            gt_label_dir=gt_label_dir,
            work_dir=work_dir,
            tracker_name=config.tracker_name,
            no_plot=config.no_plot,
            verbose=config.verbose,
            frame_align_mode=frame_align_mode,
        )

    if config.skip_tracking:
        tracking_report = {
            "status": "skipped",
            "reason": "disabled by user",
            "output_dir": str(work_dir / "tracking_results"),
        }
    else:
        tracking_report = run_tracking(
            pred_json_dir=pred_json_dir,
            gt_label_dir=gt_label_dir,
            work_dir=work_dir,
            tracker_name=config.tracker_name,
            verbose=config.verbose,
            frame_align_mode=frame_align_mode,
        )

    combined_report = {
        "prediction_root": str(pred_json_dir),
        "gt_label_root": str(gt_label_dir),
        "inference_json": str(inference_json) if inference_json is not None else None,
        "work_dir": str(work_dir),
        "label_map": LABEL_ID_TO_NAME,
        "detection_mode": detection_mode,
        "prediction_update": to_jsonable(prediction_update_report),
        "detection": to_jsonable(det_result),
        "tracking": tracking_report,
        "sequence_coverage": sequence_coverage,
        "artifacts": {
            "evaluation_text_report_path": str(combined_text_report_path),
            "evaluation_json_report_path": str(combined_report_path),
        },
    }

    write_json_report(combined_report, combined_report_path)
    write_text_report(
        build_combined_text_report(det_result, tracking_report, sequence_coverage),
        combined_text_report_path,
    )
    print(f"Combined report: {combined_report_path}")
    print(f"Combined text report: {combined_text_report_path}")
    print("One-command evaluation finished.")
    return combined_report
