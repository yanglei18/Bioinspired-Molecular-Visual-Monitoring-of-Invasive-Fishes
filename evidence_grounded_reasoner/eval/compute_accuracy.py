"""Compute multiple-choice accuracy from evaluation results (res.json).

Compares predictions against ground truth using option-letter or
species-name matching.  Also prints a per-species breakdown.
"""
import json
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict


def parse_mc_answer(text: str):
    """Parse '(X) species name' format. Returns (option_letter, species_name)."""
    text = text.lower().strip()
    match = re.search(r'\(([a-z])\)\s*(.*)', text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, text


def is_correct_mc(prediction: str, ground_truth: str) -> bool:
    gt_option, gt_species = parse_mc_answer(ground_truth)
    pred_option, pred_species = parse_mc_answer(prediction)
    if not gt_option or not gt_species:
        return False
    option_match = gt_option and pred_option and gt_option == pred_option
    species_match = gt_species and pred_species and gt_species == pred_species
    return option_match or species_match


def main():
    parser = ArgumentParser(description='Compute MC accuracy from eval results')
    parser.add_argument('--result-file', required=True, help='Path to res.json from eval_parallel_http.py')
    parser.add_argument('--benchmark-file', default='',
                        help='Benchmark JSON with ground_truth field (if not embedded in results)')
    parser.add_argument('--output', default='', help='Write accuracy summary JSON to this path')
    args = parser.parse_args()

    with open(args.result_file, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    results = payload.get('results', [])
    summary = payload.get('summary', {})

    # Build ground-truth lookup from benchmark file if provided
    gt_lookup = {}
    if args.benchmark_file:
        with open(args.benchmark_file, 'r', encoding='utf-8') as f:
            bench_data = json.load(f)
        for item in bench_data:
            vp = item.get('video_path', '')
            if 'ground_truth' in item:
                gt_lookup[vp] = item['ground_truth']
            elif 'answer' in item:
                gt_lookup[vp] = item['answer']

    # Compute accuracy
    total = len(results)
    correct = 0
    per_species = defaultdict(lambda: {'correct': 0, 'total': 0})

    for item in results:
        vp = item.get('video_path', '')
        pred = item.get('prediction', '')
        raw = item.get('model_raw_output', '')
        if not pred or pred == 'ERROR':
            continue

        # Try to find ground truth
        gt = gt_lookup.get(vp)
        if gt is None:
            # Check source_item
            src = item.get('source_item', {})
            gt = src.get('ground_truth') or src.get('answer')

        if not gt:
            continue

        if is_correct_mc(pred, gt):
            correct += 1
            per_species[gt.lower()]['correct'] += 1
        per_species[gt.lower()]['total'] += 1

    if total == 0:
        print('No results found.')
        return

    micro_acc = correct / total if total else 0
    macro_acc = 0
    species_count = 0
    for sp, stats in per_species.items():
        if stats['total'] > 0:
            macro_acc += stats['correct'] / stats['total']
            species_count += 1
    macro_acc = macro_acc / species_count if species_count else 0

    print(f'Total samples: {total}')
    print(f'Evaluated (with GT): {sum(s["total"] for s in per_species.values())}')
    print(f'Micro accuracy: {correct}/{total} = {micro_acc:.4f}')
    print(f'Macro accuracy: {macro_acc:.4f}')
    print()
    print('Per-species breakdown:')
    for sp in sorted(per_species.keys()):
        s = per_species[sp]
        acc = s['correct'] / s['total'] if s['total'] else 0
        print(f'  {sp}: {s["correct"]}/{s["total"]} = {acc:.4f}')

    if args.output:
        acc_summary = {
            'micro_accuracy': micro_acc,
            'macro_accuracy': macro_acc,
            'total_samples': total,
            'correct': correct,
            'per_species': {sp: {'correct': s['correct'], 'total': s['total'],
                                  'accuracy': s['correct'] / s['total'] if s['total'] else 0}
                           for sp, s in per_species.items()},
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(acc_summary, f, indent=2, ensure_ascii=False)
        print(f'\nAccuracy summary saved to: {args.output}')


if __name__ == '__main__':
    main()
