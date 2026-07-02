"""Lightweight smoke tests (no torch / GPU / dataset required).

These verify that the repository imports cleanly, all sources compile, and the
config layer resolves for both modules. Heavier functional tests that exercise
the models require a torch + CUDA environment and are out of scope here.

Run with either:
    pytest tests/
    python tests/test_smoke.py
"""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PACKAGES = ("object_centric_extractor", "open_world_filter")


def test_all_sources_compile():
    ok = True
    for pkg in PACKAGES:
        ok &= compileall.compile_dir(str(REPO_ROOT / pkg), quiet=1, force=True)
    assert ok, "One or more source files failed to byte-compile."


def test_detector_config_loads():
    from object_centric_extractor.pipeline.config import load_detector_config

    cfg = load_detector_config(str(REPO_ROOT / "object_centric_extractor/configs/default.yaml"))
    assert cfg.pipeline.outputs.det_dir
    assert "extraction_output" in cfg.pipeline.outputs.det_dir
    assert "sam2_prediction" not in cfg.pipeline.outputs.det_dir


def test_classifier_configs_load():
    from open_world_filter.config import load_classifier_config

    configs = sorted((REPO_ROOT / "open_world_filter/configs").glob("*.yaml"))
    assert configs, "No classifier YAML configs found."
    for path in configs:
        cfg = load_classifier_config(str(path))
        assert cfg.data.train_dir and cfg.data.val_dir and cfg.data.reference_dir
        assert cfg.outputs.save_path
        # naming migrations must hold everywhere
        assert "coarse-classifier" not in (cfg.data.train_dir + cfg.outputs.save_path)
        assert "fish-recognition-dataset" in cfg.data.train_dir
        assert "open-world-filter-outputs" in cfg.outputs.save_path


def test_fish_label_map():
    from object_centric_extractor.utils import fish_label_map as flm

    assert len(flm.LABEL_ID_TO_NAME) == 10
    assert flm.get_fish_class_name("3") == "carp"
    assert flm.normalize_fish_class_name("Common Carp.") == "common_carp"
    assert flm.is_supported_fine_class("black_carp") is True
    assert flm.is_supported_fine_class("not_a_fish") is False


def test_input_discovery_video_detection(tmp_path=None):
    import tempfile
    from object_centric_extractor.pipeline.input_discovery import is_supported_video_file

    if tmp_path is not None:
        base = Path(tmp_path)
        cleanup = False
    else:
        base = Path(tempfile.mkdtemp(prefix="owf_smoke_"))
        cleanup = True
    mp4 = base / "clip.mp4"
    txt = base / "notes.txt"
    mp4.write_bytes(b"\x00")
    txt.write_text("x", encoding="utf-8")
    try:
        assert is_supported_video_file(str(mp4)) is True
        assert is_supported_video_file(str(txt)) is False
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(base, ignore_errors=True)


def test_legacy_perdiction_filename_loads(tmp_path=None):
    import json
    import tempfile
    from object_centric_extractor.utils.annotation_io import load_prediction_sequences

    if tmp_path is not None:
        base = Path(tmp_path)
        cleanup = False
    else:
        base = Path(tempfile.mkdtemp(prefix="pred_typo_smoke_"))
        cleanup = True

    payload = {
        "videos": {
            "demo_sequence": {
                "video_name": "demo_sequence",
                "frame_count": 1,
                "frames": {
                    "000000": {
                        "labels": {},
                    },
                },
            },
        },
        "sequence_count": 1,
    }
    (base / "perdiction.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        sequences = load_prediction_sequences(base)
        assert list(sequences) == ["demo_sequence"]
        assert list(sequences["demo_sequence"]) == ["000000"]
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    tests = [
        test_all_sources_compile,
        test_detector_config_loads,
        test_classifier_configs_load,
        test_fish_label_map,
        test_input_discovery_video_detection,
        test_legacy_perdiction_filename_loads,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
