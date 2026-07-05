"""Generate a benchmark JSON file from a directory of video files.

Each entry contains a video_path and a multiple-choice question prompt.
Optionally includes a ground_truth answer field if --answer is provided.
"""
import argparse
import glob
import json
import os


# Predefined species sets. `yanghu` are the 10 Yanghu-pond species used by the
# object_centric_extractor closed loop (they match its evaluation classes). Other
# experiments (e.g. the nine invasive species) pass their own list via --options.
SPECIES_SETS = {
    "yanghu": [
        "black carp", "chinese labeo", "chinese sucker", "redeye barbel", "serrated barb",
        "common carp", "chinese paddlefish", "mud carp", "schizothorax fish", "wuchang bream",
    ],
}


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
    parser.add_argument('--options', nargs='+', default=None,
                        help='Candidate species names (e.g., "Common carp" "Crucian carp"). '
                             'Provide this or --species-set.')
    parser.add_argument('--species-set', choices=sorted(SPECIES_SETS), default=None,
                        help='Use a predefined species set instead of --options '
                             '(e.g. "yanghu" = the 10 Yanghu-pond species).')
    parser.add_argument('--answer', default='',
                        help='If set, add a ground_truth answer field with this value to all entries')
    args = parser.parse_args()

    if args.species_set:
        options = SPECIES_SETS[args.species_set]
    elif args.options:
        options = args.options
    else:
        parser.error('provide either --options or --species-set')

    question_prompt = build_question_prompt(options)
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
