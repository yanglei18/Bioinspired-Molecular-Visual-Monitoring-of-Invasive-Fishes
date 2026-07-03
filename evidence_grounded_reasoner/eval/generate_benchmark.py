"""Generate a benchmark JSON file from a directory of video files.

Each entry contains a video_path and a multiple-choice question prompt.
Optionally includes a ground_truth answer field if --answer is provided.
"""
import argparse
import glob
import json
import os


def build_question_prompt(options: list[str]) -> str:
    lines = ['<video>', 'Describe what kind of fish is in this video.', 'Options:']
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        lines.append(f'({letter}) {opt}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark JSON from video directory')
    parser.add_argument('--video-dir', required=True, help='Directory containing .mp4 video files')
    parser.add_argument('--output', required=True, help='Output benchmark JSON path')
    parser.add_argument('--options', nargs='+', required=True,
                        help='Candidate species names (e.g., "Common carp" "Crucian carp")')
    parser.add_argument('--answer', default='',
                        help='If set, add a ground_truth answer field with this value to all entries')
    args = parser.parse_args()

    question_prompt = build_question_prompt(args.options)
    video_files = sorted(glob.glob(os.path.join(args.video_dir, '*.mp4')))
    if not video_files:
        print(f'No .mp4 files found in {args.video_dir}')
        return

    benchmark_data = []
    for video_path in video_files:
        entry = {'video_path': video_path, 'question': question_prompt}
        if args.answer:
            entry['ground_truth'] = args.answer
        benchmark_data.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f'Generated {len(benchmark_data)} benchmark entries -> {args.output}')


if __name__ == '__main__':
    main()
