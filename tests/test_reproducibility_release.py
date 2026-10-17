import json
from pathlib import Path

import pytest

from reproducibility.demo import run
from reproducibility.run import prepare_config
from reproducibility.verify_reference import verify
from reproducibility.video_overlay import command

ROOT = Path(__file__).resolve().parents[1]


def test_reference_reports_match_manifest():
    assert verify(ROOT / "reference_results") >= 15


def test_small_demo_recovers_independent_exact_neighbor(tmp_path):
    result = run(tmp_path / "demo.json", n=48)
    assert len(result["results"]) == 4
    assert all(row["exact"] for row in result["results"])


def test_demo_rejects_too_small_dataset(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / "invalid.json", n=4)


def test_portable_frontier_config_points_at_real_source_and_results(tmp_path):
    path = prepare_config("frontier-20261016.json", 0, 2, tmp_path)
    config = json.loads(path.read_text())
    source = json.loads(Path(config["synthetic_config"]).read_text())
    assert source["pin_core"] == 0
    assert source["build_threads"] == 2
    assert Path(config["reference_results_root"]) == ROOT / "results/fresh-synthetic-expanded-20261012"


def test_video_overlay_refuses_to_overwrite_input(tmp_path):
    video, curves = tmp_path / "video.mp4", tmp_path / "curves.gif"
    video.touch()
    curves.touch()
    with pytest.raises(ValueError, match="replace"):
        command(video, curves, video)
    cmd = command(video, curves, tmp_path / "overlay.gif")
    assert cmd[0] == "ffmpeg"
    assert "-n" in cmd
    assert "hstack" in cmd[cmd.index("-filter_complex") + 1]
