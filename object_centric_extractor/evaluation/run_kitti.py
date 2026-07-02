
""" run_kitti.py

Run example:
run_kitti.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL CIWT

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box_train'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/kitti/kitti_2d_box_train/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['fish'],  # Fish is mapped to car for KITTI/TrackEval compatibility
        'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val', 'training_minus_val', 'test'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': ''  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['Hota']
"""

import sys
import os
import argparse
import contextlib
import io
from multiprocessing import freeze_support
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FISH_TRACKING_CLASS = 'fish'
TRACKING_COMPAT_CLASS = 'car'
FISH_TRACKING_CLASS_TITLE = 'Fish'
TRACKING_COMPAT_CLASS_TITLE = 'Car'


def _import_trackeval():
    import trackeval  # noqa: E402

    return trackeval


def relabel_tracking_text(text):
    replacements = [
        (f"-{TRACKING_COMPAT_CLASS_TITLE}", f"-{FISH_TRACKING_CLASS_TITLE}"),
        (f"_{TRACKING_COMPAT_CLASS_TITLE}", f"_{FISH_TRACKING_CLASS_TITLE}"),
        (f"-{TRACKING_COMPAT_CLASS}", f"-{FISH_TRACKING_CLASS}"),
        (f"_{TRACKING_COMPAT_CLASS}", f"_{FISH_TRACKING_CLASS}"),
        (TRACKING_COMPAT_CLASS_TITLE, FISH_TRACKING_CLASS_TITLE),
        (TRACKING_COMPAT_CLASS, FISH_TRACKING_CLASS),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def relabel_tracking_name(name):
    name = name.replace(TRACKING_COMPAT_CLASS_TITLE, FISH_TRACKING_CLASS_TITLE)
    name = name.replace(TRACKING_COMPAT_CLASS, FISH_TRACKING_CLASS)
    return name


def relabel_tracking_result(value):
    if isinstance(value, str):
        return relabel_tracking_text(value)
    if isinstance(value, dict):
        return {
            relabel_tracking_result(key): relabel_tracking_result(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [relabel_tracking_result(item) for item in value]
    if isinstance(value, tuple):
        return tuple(relabel_tracking_result(item) for item in value)
    return value


class RelabelingStdout(io.TextIOBase):
    def __init__(self, target, capture_buffer=None):
        self._target = target
        self._capture_buffer = capture_buffer

    def write(self, text):
        relabeled_text = relabel_tracking_text(text)
        if self._capture_buffer is not None:
            self._capture_buffer.write(relabeled_text)
        return self._target.write(relabeled_text)

    def flush(self):
        return self._target.flush()

    def writable(self):
        return True


def rewrite_tracking_artifacts(output_dir):
    if output_dir is None:
        return

    root = Path(output_dir)
    if not root.exists():
        return

    text_suffixes = {'.txt', '.csv', '.json', '.yaml', '.yml', '.md'}

    for file_path in sorted(root.rglob('*')):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in text_suffixes:
            continue
        content = file_path.read_text(encoding='utf-8')
        updated_content = relabel_tracking_text(content)
        if updated_content != content:
            file_path.write_text(updated_content, encoding='utf-8')

    rename_paths = sorted(root.rglob('*'), key=lambda path: len(path.parts), reverse=True)
    for path in rename_paths:
        updated_name = relabel_tracking_name(path.name)
        if updated_name == path.name:
            continue
        path.rename(path.with_name(updated_name))


def normalize_tracking_classes(classes_to_eval):
    if not classes_to_eval:
        return [TRACKING_COMPAT_CLASS]

    normalized_classes = []
    for cls in classes_to_eval:
        cls_lower = cls.lower()
        if cls_lower == FISH_TRACKING_CLASS:
            normalized_classes.append(TRACKING_COMPAT_CLASS)
            continue
        raise ValueError(f"Only {FISH_TRACKING_CLASS} tracking evaluation is supported, but got {cls}")
    return normalized_classes


def run_tracking_eval(
    gt_folder="data/kitti_format/gt/",
    trackers_folder="data/kitti_format/pred/",
    output_folder=None,
    trackers_to_eval=None,
    metrics=None,
    classes_to_eval=None,
    rendered_output_path=None,
    return_rendered_output=False,
):
    """Run TrackEval KITTI 2D Box evaluation with project defaults."""
    capture_buffer = io.StringIO()
    relabeled_stdout = RelabelingStdout(sys.stdout, capture_buffer=capture_buffer)
    relabeled_stderr = RelabelingStdout(sys.stderr, capture_buffer=capture_buffer)
    with contextlib.redirect_stdout(relabeled_stdout), contextlib.redirect_stderr(relabeled_stderr):
        trackeval = _import_trackeval()
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = False
        default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
        default_dataset_config['GT_FOLDER'] = gt_folder
        default_dataset_config['TRACKERS_FOLDER'] = trackers_folder
        if output_folder is not None:
            default_dataset_config['OUTPUT_FOLDER'] = output_folder
        default_dataset_config['CLASSES_TO_EVAL'] = normalize_tracking_classes(classes_to_eval)
        if trackers_to_eval is not None:
            default_dataset_config['TRACKERS_TO_EVAL'] = trackers_to_eval
        default_metrics_config = {'METRICS': metrics or ['HOTA']}

        evaluator = trackeval.Evaluator(default_eval_config)
        dataset_list = [trackeval.datasets.Kitti2DBox(default_dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            if metric.get_name() in default_metrics_config['METRICS']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        result = evaluator.evaluate(dataset_list, metrics_list)
    rendered_output = capture_buffer.getvalue().strip()
    if rendered_output_path is not None:
        output_path = Path(rendered_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered_output + '\n', encoding='utf-8')
    rewrite_tracking_artifacts(output_folder)
    relabeled_result = relabel_tracking_result(result)
    if return_rendered_output:
        return {
            'result': relabeled_result,
            'rendered_output': rendered_output,
            'rendered_output_path': rendered_output_path,
        }
    return relabeled_result

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description="Run Fish tracking evaluation with TrackEval compatibility")
    parser.add_argument("--GT_FOLDER", default="data/kitti_format/gt/")
    parser.add_argument("--TRACKERS_FOLDER", default="data/kitti_format/pred/")
    parser.add_argument("--OUTPUT_FOLDER", default=None)
    parser.add_argument("--TRACKERS_TO_EVAL", nargs='+', default=None)
    parser.add_argument("--METRICS", nargs='+', default=['HOTA'])
    parser.add_argument(
        "--CLASSES_TO_EVAL",
        nargs='+',
        default=[FISH_TRACKING_CLASS],
        help="Tracking class. Only fish is supported.",
    )
    args = parser.parse_args()

    print("---dataset_config: ", {
        'GT_FOLDER': args.GT_FOLDER,
        'TRACKERS_FOLDER': args.TRACKERS_FOLDER,
        'OUTPUT_FOLDER': args.OUTPUT_FOLDER,
        'TRACKERS_TO_EVAL': args.TRACKERS_TO_EVAL,
        'CLASSES_TO_EVAL': args.CLASSES_TO_EVAL,
    })
    run_tracking_eval(
        gt_folder=args.GT_FOLDER,
        trackers_folder=args.TRACKERS_FOLDER,
        output_folder=args.OUTPUT_FOLDER,
        trackers_to_eval=args.TRACKERS_TO_EVAL,
        metrics=args.METRICS,
        classes_to_eval=args.CLASSES_TO_EVAL,
    )
