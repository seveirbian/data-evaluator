import json
from pathlib import Path

import numpy as np
import pytest

from src.episodes_extractor._extractor import _validate_inputs, extract_episodes


def _make_src(tmp_path: Path, total_episodes: int) -> Path:
    """Create a minimal dataset dir with just meta/info.json."""
    src = tmp_path / "src"
    (src / "meta").mkdir(parents=True)
    (src / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": total_episodes})
    )
    return src


def test_validate_rejects_empty_ids(tmp_path):
    src = _make_src(tmp_path, 5)
    with pytest.raises(ValueError, match="empty"):
        _validate_inputs(src, [], tmp_path / "out")


def test_validate_rejects_duplicate_ids(tmp_path):
    src = _make_src(tmp_path, 5)
    with pytest.raises(ValueError, match="duplicate"):
        _validate_inputs(src, [1, 1, 2], tmp_path / "out")


def test_validate_rejects_out_of_range_ids(tmp_path):
    src = _make_src(tmp_path, 5)
    with pytest.raises(ValueError, match="out of range"):
        _validate_inputs(src, [0, 5], tmp_path / "out")  # valid range is 0..4


def test_validate_rejects_missing_info_json(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        _validate_inputs(src, [0], tmp_path / "out")


def test_validate_rejects_nonempty_out_dir(tmp_path):
    src = _make_src(tmp_path, 5)
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        _validate_inputs(src, [0, 1], out)


def test_validate_passes_for_valid_inputs(tmp_path):
    src = _make_src(tmp_path, 5)
    total = _validate_inputs(src, [0, 2, 4], tmp_path / "out")
    assert total == 5


def _build_synthetic_dataset(root: Path, lengths: list[int]):
    """Create a small v3.0 dataset with len(lengths) episodes.

    Frame state/action encode (episode, frame) so content is checkable.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.images.cam": {
            "dtype": "video",
            "shape": (3, 64, 64),
            "names": ["channels", "height", "width"],
        },
    }
    ds = LeRobotDataset.create(
        "synthetic",
        fps=10,
        features=features,
        root=root,
        robot_type="test",
        use_videos=True,
        video_backend="pyav",
    )
    for ep, length in enumerate(lengths):
        for fi in range(length):
            ds.add_frame(
                {
                    "observation.state": np.array([ep, fi], dtype=np.float32),
                    "action": np.array([ep, fi], dtype=np.float32),
                    "observation.images.cam": np.zeros((3, 64, 64), dtype=np.float32),
                    "task": f"task{ep}",
                }
            )
        ds.save_episode()
    ds.finalize()


def test_extract_round_trip(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_dir = tmp_path / "src"
    _build_synthetic_dataset(src_dir, lengths=[3, 4, 2])  # episodes 0,1,2

    out_dir = tmp_path / "out"
    # Extract episodes 2 and 0, in that order → new ids 0 and 1 respectively
    mapping = extract_episodes(src_dir, [2, 0], out_dir, repo_id="extracted")

    assert mapping == {2: 0, 0: 1}

    out = LeRobotDataset("extracted", root=out_dir)
    assert out.num_episodes == 2
    assert out.num_frames == 2 + 3  # episode 2 (len 2) + episode 0 (len 3)

    # mapping file written
    mapping_file = out_dir / "extraction_mapping.json"
    assert mapping_file.exists()
    assert json.loads(mapping_file.read_text()) == {"2": 0, "0": 1}

    # new episode 0 came from original episode 2 → state[:,0] == 2
    first_frame = out[0]
    assert float(first_frame["observation.state"][0]) == pytest.approx(2.0)


def test_extract_invalid_id_raises_before_writing(tmp_path):
    src_dir = tmp_path / "src"
    _build_synthetic_dataset(src_dir, lengths=[2, 2])  # episodes 0,1
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="out of range"):
        extract_episodes(src_dir, [0, 9], out_dir)
