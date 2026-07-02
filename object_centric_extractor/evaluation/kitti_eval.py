# -*- coding: utf-8 -*-
import csv
import json
import numpy as np
import os

FISH_CLASS = 'Fish'
DONTCARE_CLASS = 'DontCare'
KITTI_COMPAT_CLASS = 'Car'

MIN_HEIGHT = [20, 10, 10]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.1, 0.1]
MIN_OVERLAP = 0.3
N_SAMPLE_PTS = 41
AP11_SAMPLE_STEP = 4
FRAME_ALIGN_MODES = ("union", "gt_present", "gt_range")


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_ap11_from_precisions(precision_values):
    ap_value = 0.0
    for i in range(0, N_SAMPLE_PTS, AP11_SAMPLE_STEP):
        ap_value += precision_values[i]
    return 100.0 * ap_value / 11.0


def compute_ap41_from_precisions(precision_values):
    if len(precision_values) == 0:
        return 0.0
    return 100.0 * float(sum(precision_values)) / len(precision_values)




def average_ap(ap_values):
    if not ap_values:
        return 0.0
    return float(sum(ap_values) / len(ap_values))


def build_terminal_prefix(terminal_label):
    if not terminal_label:
        return ""
    return f"[{terminal_label}] "




def normalize_frame_align_mode(frame_align_mode):
    if frame_align_mode not in FRAME_ALIGN_MODES:
        raise ValueError(
            f"Unsupported frame_align_mode {frame_align_mode}. "
            f"Expected one of: {', '.join(FRAME_ALIGN_MODES)}"
        )
    return frame_align_mode


def normalize_eval_class_name(cls):
    if cls == FISH_CLASS:
        return FISH_CLASS
    raise ValueError(f"Only {FISH_CLASS} evaluation is supported, but got {cls}")


def normalize_loaded_class_name(raw_class):
    if raw_class == DONTCARE_CLASS:
        return DONTCARE_CLASS
    if raw_class in {FISH_CLASS, KITTI_COMPAT_CLASS}:
        return FISH_CLASS
    raise ValueError(
        f"Unsupported prediction class {raw_class}. "
        f"Expected one of: {FISH_CLASS}, {KITTI_COMPAT_CLASS}, {DONTCARE_CLASS}"
    )

def load_gt(path):
    gt_data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split()
            frame = int(line[0])
            if frame not in gt_data:
                gt_data[frame] = []
            
            # Use the new field names
            gt_data[frame].append({
                'frame': frame,
                'track_id': int(line[1]),
                'class': FISH_CLASS,
                'trunc': float(line[3]),  # truncation level
                'occ': float(line[4]),    # occlusion level
                'alpha': float(line[5]),  # observation angle
                'box': [float(x) for x in line[6:10]],  # [x1, y1, x2, y2]
                'dimensions': [float(x) for x in line[10:13]],  # [height, width, length]
                'location': [float(x) for x in line[13:16]],    # [x, y, z]
            })
    return gt_data

def load_pred(path):
    pred_data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split()
            frame = int(line[0])
            if frame not in pred_data:
                pred_data[frame] = []
            
            # Use the new field names
            pred_data[frame].append({
                'frame': frame,
                'track_id': int(line[1]),
                'class': normalize_loaded_class_name(line[2]),
                'trunc': float(line[3]),  # truncation level
                'occ': float(line[4]),    # occlusion level
                'alpha': float(line[5]),  # observation angle
                'box': [float(x) for x in line[6:10]],  # [x1, y1, x2, y2]
                'dimensions': [float(x) for x in line[10:13]],  # [height, width, length]
                'location': [float(x) for x in line[13:16]],    # [x, y, z]
                'score': float(line[17]) if len(line) > 17 and line[17] != '-1' else 1.0  # detection confidence; defaults to 1.0 if absent
            })
    return pred_data


def get_thresholds(v, n_groundTruth):
    if n_groundTruth <= 0 or len(v) == 0:
        return []
    v = np.array(v)
    sort_ind_desc = np.argsort(v * -1)
    vs = v[sort_ind_desc]

    t = []
    current_recall = 0

    for i in range(vs.shape[0]):
        l_recall = (i+1)/n_groundTruth

        if i < vs.shape[0] - 1:
            r_recall = (i+2)/n_groundTruth
        else:
            r_recall = l_recall

        if (r_recall - current_recall) < (current_recall - l_recall) and i < (vs.shape[0] - 1):
            continue
        t.append(vs[i])
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)
    return t


def get_iou(gt, pred, union=True):
    gxmin, gymin, gxmax, gymax = gt['box']
    pxmin, pymin, pxmax, pymax = pred['box']

    ixmin = np.maximum(gxmin, pxmin)
    iymin = np.maximum(gymin, pymin)
    ixmax = np.minimum(gxmax, pxmax)
    iymax = np.minimum(gymax, pymax)

    ih = np.maximum(0., iymax - iymin)
    iw = np.maximum(0., ixmax - ixmin)

    gvol = (gxmax - gxmin) * (gymax - gymin)
    pvol = (pxmax - pxmin) * (pymax - pymin)
    ivol = iw * ih

    if union:
        union_area = gvol + pvol - ivol
        # guard against division-by-zero
        if union_area == 0:
            return 0.0
        iou = ivol / union_area
    else:
        # guard against division-by-zero
        if pvol == 0:
            return 0.0
        iou = ivol / pvol
    return iou


def clean_data(gts, preds, cls, diff):
    ignore_gt = []
    ignore_pred = []
    dontcare = []

    n_gt = 0

    #clean ground truth
    for gt in gts:
        if cls == gt['class']:
            valid_class = 1
        else:
            valid_class = -1

        height = gt['box'][3] - gt['box'][1]

        # Filter using the new field names
        # Adjusted filtering logic to handle special values correctly
        occ_filter = False
        trunc_filter = False
        height_filter = False
        
        # Only apply the filter check when the value is valid
        if gt['occ'] >= 0:
            occ_filter = gt['occ'] > MAX_OCCLUSION[diff]
            
        if gt['trunc'] >= 0:
            trunc_filter = gt['trunc'] > MAX_TRUNCATION[diff]
            
        if height > 0:
            height_filter = height < MIN_HEIGHT[diff]
        
        if occ_filter or trunc_filter or height_filter:
            ignore = True
        else:
            ignore = False

        if valid_class == 1 and not ignore:
            n_gt += 1
            ignore_gt.append(0)
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignore_gt.append(1)
        else:
            ignore_gt.append(-1)

        #set Don't care
        if gt['class'] == DONTCARE_CLASS:
            dontcare.append(True)
        else:
            dontcare.append(False)

    #clean predictions
    for pred in preds:
        if pred['class'] == cls:
            valid_class = 1
        else:
            valid_class = -1
        height = pred['box'][3] - pred['box'][1]

        # Filter using the new field names
        # Adjusted filtering logic to handle special values correctly
        height_filter = False
        
        # Only apply the filter check when the height is valid
        if height > 0:
            height_filter = height < MIN_HEIGHT[diff]
        
        if height_filter:
            ignore_pred.append(1)
        elif valid_class == 1:
            ignore_pred.append(0)
        else:
            ignore_pred.append(-1)

    return ignore_gt, dontcare, ignore_pred, n_gt



def compute_statistics(gts, preds, dontcare, ignore_gt, ignore_pred, compute_fp, threshold, cls, diff):
    n_gt = len(gts)
    n_pred = len(preds)

    assigned_detection = [False for _ in range(n_pred)]
    TP, FP, FN = 0, 0, 0
    vs = []

    ignore_threshold = []
    if compute_fp:
        for pred in preds:
            if pred['score'] < threshold:
                ignore_threshold.append(True)
            else:
                ignore_threshold.append(False)
    else:
        for pred in preds:
            ignore_threshold.append(False)

    for i in range(n_gt):
        if ignore_gt[i] == -1:
            continue

        det_idx = -1
        valid_detection = -1
        max_iou = 0.
        assigned_ignored_det = False

        for j in range(n_pred):
            if ignore_pred[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignore_threshold[j]:
                continue

            iou = get_iou(gts[i], preds[j])

            if not compute_fp and iou > MIN_OVERLAP and preds[j]['score'] > threshold:
                det_idx = j
                valid_detection = preds[j]['score']
            elif compute_fp and iou > MIN_OVERLAP and (iou > max_iou or assigned_ignored_det) and ignore_pred[j] == 0:
                max_iou = iou
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif compute_fp and iou > MIN_OVERLAP and valid_detection == -1. and ignore_pred[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if valid_detection == -1 and ignore_gt[i] == 0:
            FN += 1
        elif valid_detection != -1 and (ignore_gt[i] == 1 or ignore_pred[det_idx]==1):
            assigned_detection[det_idx] = True
        elif valid_detection != -1:
            TP += 1
            vs.append(preds[det_idx]['score'])
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(n_pred):
            if not (assigned_detection[i] or ignore_pred[i]==-1 or ignore_pred[i]==1 or ignore_threshold[i]):
                FP += 1

        n_stuff = 0
        for i in range(n_gt):
            if not dontcare[i]:
                continue
            for j in range(n_pred):
                if assigned_detection[j]:
                    continue
                if ignore_pred[j] == -1 or ignore_pred[j] == 1:
                    continue
                if ignore_threshold[j]:
                    continue
                iou = get_iou(preds[j], gts[i], union=False)
                if iou > MIN_OVERLAP:
                    assigned_detection[j] = True
                    n_stuff += 1

        FP -= n_stuff

    return TP, FP, FN, vs


def eval_class(gt_list, pred_list, cls, diff):
    ignore_gt_list = []
    ignore_pred_list = []
    dontcare_list = []
    total_gt_num = 0

    #clean data
    vs = []
    for i in range(len(gt_list)):
        ignore_gt, dontcare, ignore_pred, n_gt_ = clean_data(gt_list[i], pred_list[i], cls, diff)
        ignore_gt_list.append(ignore_gt)
        ignore_pred_list.append(ignore_pred)
        dontcare_list.append(dontcare)
        total_gt_num += n_gt_

        _, _, _, vs_ = compute_statistics(gt_list[i], pred_list[i], dontcare, ignore_gt, ignore_pred, False, 0, cls, diff)
        vs = vs + vs_
    thresholds = get_thresholds(vs, total_gt_num)
    if total_gt_num <= 0 or len(thresholds) == 0:
        return [0.0] * N_SAMPLE_PTS, []
    len_th = len(thresholds)
    TPs = [0.] * len_th
    FPs = [0.] * len_th
    FNs = [0.] * len_th

    for i in range(len(gt_list)):
        for t, th in enumerate(thresholds):
            TP, FP, FN, _, = compute_statistics(gt_list[i], pred_list[i], dontcare_list[i], ignore_gt_list[i], ignore_pred_list[i], True, th, cls, diff)
            TPs[t] += TP
            FPs[t] += FP
            FNs[t] += FN

    precisions = [0.] * N_SAMPLE_PTS
    recalls = []

    for t, th in enumerate(thresholds):
        r = safe_div(TPs[t], TPs[t] + FNs[t])
        recalls.append(r)
        precisions[t] = safe_div(TPs[t], TPs[t] + FPs[t])

    for t, th in enumerate(thresholds):
        precisions[t] = np.max(precisions[t:])

    return  precisions, recalls


def plot_and_compute(precisions, cls, plot, plot_path='2d_result.png', terminal_label=None):
    if plot:
        import matplotlib.pyplot as plt

        Xs = np.arange(0., 1., 1./len(precisions[0]))

        l_easy = plt.plot(Xs, precisions[0], c='green')[0]
        l_moderate = plt.plot(Xs, precisions[1], c='blue')[0]
        l_hard = plt.plot(Xs, precisions[2], c='red')[0]

        labels = ['Easy','Moderate','Hard']
        plt.legend(handles=[l_easy,l_moderate,l_hard],labels=labels,loc='best')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(cls)
        plt.ylim((0,1.0))
        plt.grid()
        plt.savefig(plot_path)
        plt.close()

    ap11_easy = compute_ap11_from_precisions(precisions[0])
    ap11_moderate = compute_ap11_from_precisions(precisions[1])
    ap11_hard = compute_ap11_from_precisions(precisions[2])
    total_map11 = average_ap([ap11_easy, ap11_moderate, ap11_hard])

    ap41_easy = compute_ap41_from_precisions(precisions[0])
    ap41_moderate = compute_ap41_from_precisions(precisions[1])
    ap41_hard = compute_ap41_from_precisions(precisions[2])
    total_map41 = average_ap([ap41_easy, ap41_moderate, ap41_hard])

    return {
        'class': cls,
        'easy_ap': ap11_easy,
        'moderate_ap': ap11_moderate,
        'hard_ap': ap11_hard,
        'total_map': total_map11,
        'easy_ap11': ap11_easy,
        'moderate_ap11': ap11_moderate,
        'hard_ap11': ap11_hard,
        'total_map11': total_map11,
        'easy_ap41': ap41_easy,
        'moderate_ap41': ap41_moderate,
        'hard_ap41': ap41_hard,
        'total_map41': total_map41,
        'plot_path': plot_path if plot else None,
    }


def evaluate_single_frame(gts, preds, cls, sequence_name, frame_id):
    frame_precisions = []
    for diff in range(3):
        precisions, _ = eval_class([gts], [preds], cls, diff)
        frame_precisions.append(precisions)

    easy_ap11 = compute_ap11_from_precisions(frame_precisions[0])
    moderate_ap11 = compute_ap11_from_precisions(frame_precisions[1])
    hard_ap11 = compute_ap11_from_precisions(frame_precisions[2])
    total_map11 = average_ap([easy_ap11, moderate_ap11, hard_ap11])
    easy_ap41 = compute_ap41_from_precisions(frame_precisions[0])
    moderate_ap41 = compute_ap41_from_precisions(frame_precisions[1])
    hard_ap41 = compute_ap41_from_precisions(frame_precisions[2])
    total_map41 = average_ap([easy_ap41, moderate_ap41, hard_ap41])
    return {
        'sequence_name': sequence_name,
        'frame_id': frame_id,
        'easy_ap': easy_ap11,
        'moderate_ap': moderate_ap11,
        'hard_ap': hard_ap11,
        'total_map': total_map11,
        'easy_ap11': easy_ap11,
        'moderate_ap11': moderate_ap11,
        'hard_ap11': hard_ap11,
        'total_map11': total_map11,
        'easy_ap41': easy_ap41,
        'moderate_ap41': moderate_ap41,
        'hard_ap41': hard_ap41,
        'total_map41': total_map41,
        'gt_count': len(gts),
        'pred_count': len(preds),
    }


def write_frame_metrics_csv(frame_metrics, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'sequence_name',
                'frame_id',
                'gt_count',
                'pred_count',
                'easy_ap',
                'moderate_ap',
                'hard_ap',
                'total_map',
                'easy_ap11',
                'moderate_ap11',
                'hard_ap11',
                'total_map11',
                'easy_ap41',
                'moderate_ap41',
                'hard_ap41',
                'total_map41',
            ],
        )
        writer.writeheader()
        writer.writerows(frame_metrics)


def write_summary_json(summary, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(summary, json_file, indent=2)




def evaluate_directory(
    gt_dir,
    pred_dir,
    cls,
    seq_name=None,
    frame_align_mode='union',
    plot=True,
    plot_path='2d_result.png',
    frame_metrics_path=None,
    summary_path=None,
    verbose=False,
    terminal_label=None,
):
    cls = normalize_eval_class_name(cls)
    frame_align_mode = normalize_frame_align_mode(frame_align_mode)
    gt_list = []
    pred_list = []
    frame_metrics = []

    pred_files = {
        file_name
        for file_name in os.listdir(pred_dir)
        if os.path.isfile(os.path.join(pred_dir, file_name))
    }
    gt_files = {
        file_name
        for file_name in os.listdir(gt_dir)
        if os.path.isfile(os.path.join(gt_dir, file_name))
    }
    eval_files = sorted(pred_files | gt_files)

    frame_count = 0
    file_count = 0
    paired_file_count = 0
    common_frame_count = 0
    gt_only_frame_count = 0
    pred_only_frame_count = 0

    for f in eval_files:
        if seq_name is not None and seq_name not in f:
            continue

        pred_file_path = os.path.join(pred_dir, f)
        gt_file_path = os.path.join(gt_dir, f)

        pred_exists = os.path.exists(pred_file_path)
        gt_exists = os.path.exists(gt_file_path)
        if not pred_exists and not gt_exists:
            continue

        if pred_exists and gt_exists:
            paired_file_count += 1

        record_pred = load_pred(pred_file_path) if pred_exists else {}
        record_gt = load_gt(gt_file_path) if gt_exists else {}

        pred_frames = set(record_pred.keys())
        gt_frames = set(record_gt.keys())
        if frame_align_mode == 'union':
            all_frames = sorted(pred_frames | gt_frames)
        elif frame_align_mode == 'gt_present':
            all_frames = sorted(gt_frames)
        else:
            if gt_frames:
                all_frames = list(range(min(gt_frames), max(gt_frames) + 1))
            else:
                all_frames = []
        common_frames = pred_frames & gt_frames
        if not all_frames:
            if verbose and paired_file_count <= 3:
                print("File: {}".format(f))
                print("  Pred frames: {}, GT frames: {}".format(len(record_pred), len(record_gt)))
                print("  Common frames: 0")
                print("  Skip empty fine-grained file pair")
            continue

        common_frame_count += len(common_frames)
        gt_only_frame_count += len(gt_frames - pred_frames)
        pred_only_frame_count += len(pred_frames - gt_frames)

        if verbose and file_count < 3:
            print("File: {}".format(f))
            print("  Pred frames: {}, GT frames: {}".format(len(record_pred), len(record_gt)))
            print("  Common frames: {}".format(len(common_frames)))
            print("  GT-only frames: {}".format(len(gt_frames - pred_frames)))
            print("  Pred-only frames: {}".format(len(pred_frames - gt_frames)))
            first_frame = all_frames[0]
            print("  First eval frame {} - Pred data: {}, GT data: {}".format(
                first_frame, len(record_pred.get(first_frame, [])), len(record_gt.get(first_frame, []))))

        for frame in all_frames:
            pred_data = record_pred.get(frame, [])
            gt_data = record_gt.get(frame, [])
            pred_list.append(pred_data)
            gt_list.append(gt_data)
            frame_metrics.append(evaluate_single_frame(gt_data, pred_data, cls, os.path.splitext(f)[0], frame))
            frame_count += 1

        file_count += 1

    terminal_prefix = build_terminal_prefix(terminal_label)
    print(
        '{}Processed {} frames from {} files (paired files: {}, common frames: {}, gt-only frames: {}, pred-only frames: {})'.format(
            terminal_prefix,
            frame_count,
            file_count,
            paired_file_count,
            common_frame_count,
            gt_only_frame_count,
            pred_only_frame_count,
        )
    )

    if len(gt_list) == 0 or len(pred_list) == 0:
        print(f"{terminal_prefix}No data to evaluate!")
        return None
        
    # debug info
    if verbose:
        print('gt_list len: {}, pred_list len: {}'.format(len(gt_list), len(pred_list)))
        print("First few entries:")
        for i in range(min(3, len(gt_list))):
            print("  Frame {}: GT items: {}, Pred items: {}".format(i, len(gt_list[i]), len(pred_list[i])))
            if len(gt_list[i]) > 0:
                print("    GT example: {}".format(gt_list[i][0]))
            if len(pred_list[i]) > 0:
                print("    Pred example: {}".format(pred_list[i][0]))
                pred_scores = [pred['score'] for pred in pred_list[i]]
                if len(pred_scores) > 0:
                    print("    Pred scores range: {:.2f} - {:.2f}".format(min(pred_scores), max(pred_scores)))
        
    recall_all_diff = []
    precision_all_diff = []
    for diff in range(3):
        precisions, recalls = eval_class(gt_list, pred_list, cls, diff)
        precision_all_diff.append(precisions)
        recall_all_diff.append(recalls)

    summary = plot_and_compute(
        precision_all_diff,
        cls,
        plot=plot,
        plot_path=plot_path,
        terminal_label=terminal_label,
    )
    summary['frame_align_mode'] = frame_align_mode
    summary['terminal_label'] = terminal_label
    summary['frame_count'] = frame_count
    summary['file_count'] = file_count
    summary['paired_file_count'] = paired_file_count
    summary['common_frame_count'] = common_frame_count
    summary['gt_only_frame_count'] = gt_only_frame_count
    summary['pred_only_frame_count'] = pred_only_frame_count
    summary['frame_metrics'] = frame_metrics

    if frame_metrics_path is not None:
        write_frame_metrics_csv(frame_metrics, frame_metrics_path)
        summary['frame_metrics_path'] = frame_metrics_path
    if summary_path is not None:
        summary_without_frames = dict(summary)
        summary_without_frames.pop('frame_metrics', None)
        write_summary_json(summary_without_frames, summary_path)
        summary['summary_path'] = summary_path

    return summary


def eval(gt_dir, pred_dir, cls, seq_name=None):
    return evaluate_directory(
        gt_dir,
        pred_dir,
        cls,
        seq_name=seq_name,
        frame_align_mode='union',
        plot=True,
        plot_path='2d_result.png',
        verbose=False,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Fish 2D detection AP with KITTI-compatible inputs')
    parser.add_argument('--gt_dir', type=str, default='data/kitti_format/gt/label_02')
    parser.add_argument('--pred_dir', type=str, default='data/kitti_format/pred/DTTR/data')
    parser.add_argument('--cls', type=str, default=FISH_CLASS, help='Detection class. Only Fish is supported.')
    parser.add_argument('--seq_name', type=str, default=None)
    parser.add_argument('--no_plot', action='store_true', help='Disable PR curve plotting')
    parser.add_argument('--plot_path', type=str, default='2d_result.png')
    parser.add_argument('--frame_metrics_path', type=str, default=None)
    parser.add_argument('--summary_path', type=str, default=None)
    parser.add_argument(
        '--frame_align_mode',
        type=str,
        choices=FRAME_ALIGN_MODES,
        default='union',
        help='Frame alignment mode for detection AP evaluation.',
    )
    parser.add_argument('--verbose', action='store_true', help='Print per-file and sample-frame debug details')
    args = parser.parse_args()

    evaluate_directory(
        args.gt_dir,
        args.pred_dir,
        args.cls,
        seq_name=args.seq_name,
        frame_align_mode=args.frame_align_mode,
        plot=not args.no_plot,
        plot_path=args.plot_path,
        frame_metrics_path=args.frame_metrics_path,
        summary_path=args.summary_path,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
