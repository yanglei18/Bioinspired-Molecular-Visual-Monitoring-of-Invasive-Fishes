#!/usr/bin/env python3
"""Compatibility CLI wrapper for the SAM2 detector evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from pipeline.config import load_detector_config
    from evaluation.pipeline import (
        EvaluationPipelineConfig,
        run_evaluation_pipeline,
    )
except ModuleNotFoundError:
    from object_centric_extractor.pipeline.config import load_detector_config
    from object_centric_extractor.evaluation.pipeline import (
        EvaluationPipelineConfig,
        run_evaluation_pipeline,
    )


DEFAULT_CONFIG_CLI_PATH = "object_centric_extractor/configs/default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 detector evaluation in one command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_CLI_PATH,
        help="Path to the SAM2 detector YAML config file",
    )
    parser.add_argument(
        "--inference_json",
        type=str,
        default=None,
        help=(
            "Optional instance-video inference JSON. Overrides evaluation.inference_json "
            "from the config for this run only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_config = load_detector_config(args.config)
    eval_defaults = detector_config.evaluation
    config = EvaluationPipelineConfig(
        pred_json_dir=eval_defaults.pred_json_dir,
        gt_label_dir=eval_defaults.gt_label_dir,
        inference_json=(
            args.inference_json
            if args.inference_json is not None
            else eval_defaults.inference_json
        ),
        work_dir=eval_defaults.work_dir,
        tracker_name=eval_defaults.tracker_name,
        skip_detection=eval_defaults.skip_detection,
        skip_tracking=eval_defaults.skip_tracking,
        no_plot=eval_defaults.no_plot,
        verbose=eval_defaults.verbose,
        frame_align_mode=eval_defaults.frame_align_mode,
    )
    repo_root = Path(__file__).resolve().parent.parent
    run_evaluation_pipeline(config, repo_root=repo_root)


if __name__ == "__main__":
    main()
