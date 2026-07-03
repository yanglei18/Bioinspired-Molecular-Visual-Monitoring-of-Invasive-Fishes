#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any


def _rewrite_video_paths(sample: dict[str, Any], video_root: str, original_video_root: str | None) -> dict[str, Any]:
    rewritten = dict(sample)
    videos = rewritten.get("videos")
    if not isinstance(videos, list):
        return rewritten

    new_videos = []
    for video in videos:
        if not isinstance(video, str):
            new_videos.append(video)
            continue
        if original_video_root and video.startswith(original_video_root.rstrip("/") + "/"):
            suffix = video[len(original_video_root.rstrip("/")) + 1:]
            new_videos.append(str(Path(video_root) / suffix))
        elif Path(video).is_absolute():
            new_videos.append(video)
        else:
            new_videos.append(str(Path(video_root) / video))
    rewritten["videos"] = new_videos
    return rewritten


def build_smoke_dataset(
    source_path: str | Path,
    output_path: str | Path,
    num_samples: int,
    video_root: str | None = None,
    original_video_root: str | None = None,
) -> int:
    source_path = Path(source_path)
    output_path = Path(output_path)
    samples = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("source dataset must be a JSON list")

    selected = samples[:num_samples]
    if video_root:
        selected = [_rewrite_video_paths(sample, video_root, original_video_root) for sample in selected]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small GRPO smoke-test dataset.")
    parser.add_argument("--source", required=True, help="Path to the full RL JSON dataset.")
    parser.add_argument("--output", required=True, help="Path for the smoke-test JSON dataset.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to keep.")
    parser.add_argument("--video-root", help="Optional new root for video paths.")
    parser.add_argument("--original-video-root", help="Original absolute video root to replace.")
    args = parser.parse_args()

    count = build_smoke_dataset(
        source_path=args.source,
        output_path=args.output,
        num_samples=args.num_samples,
        video_root=args.video_root,
        original_video_root=args.original_video_root,
    )
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
