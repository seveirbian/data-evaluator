import json
from pathlib import Path

import pytest

from src.first_obs_extractor._extractor import _grid_dims, _validate_episode_ids


def _make_src(tmp_path: Path, total_episodes: int) -> Path:
    src = tmp_path / "src"
    (src / "meta").mkdir(parents=True)
    (src / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": total_episodes})
    )
    return src


@pytest.mark.parametrize(
    "n,expected",
    [(1, (1, 1)), (4, (2, 2)), (5, (2, 3)), (6, (2, 3)), (7, (3, 3)), (9, (3, 3))],
)
def test_grid_dims(n, expected):
    assert _grid_dims(n) == expected


def test_validate_rejects_empty_ids(tmp_path):
    src = _make_src(tmp_path, 5)
    with pytest.raises(ValueError, match="empty"):
        _validate_episode_ids(src, [])


def test_validate_rejects_duplicate_ids(tmp_path):
    src = _make_src(tmp_path, 5)
    with pytest.raises(ValueError, match="duplicate"):
        _validate_episode_ids(src, [1, 1, 2])


def test_validate_rejects_out_of_range_ids(tmp_path):
    src = _make_src(tmp_path, 5)
    with pytest.raises(ValueError, match="out of range"):
        _validate_episode_ids(src, [0, 5])


def test_validate_rejects_missing_info_json(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        _validate_episode_ids(src, [0])


def test_validate_passes_for_valid_inputs(tmp_path):
    src = _make_src(tmp_path, 5)
    assert _validate_episode_ids(src, [0, 2, 4]) == 5


import numpy as np

from src.first_obs_extractor._extractor import _render_montage


def test_render_montage_writes_png(tmp_path):
    # mix of HWC (numpy) and CHW (channel-first) float images
    hwc = np.zeros((8, 8, 3), dtype=np.float32)
    chw = np.ones((3, 8, 8), dtype=np.float32)
    cells = [("ep0 / front", hwc), ("ep1 / front", chw)]
    out = tmp_path / "sub" / "montage.png"

    result = _render_montage(cells, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


from src.first_obs_extractor._extractor import extract_first_obs


def _build_synthetic_dataset(root: Path, lengths: list[int]):
    """Create a small v3.0 dataset with len(lengths) episodes, one camera."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.images.cam": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
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
                    "observation.images.cam": np.zeros((64, 64, 3), dtype=np.float32),
                    "task": f"task{ep}",
                }
            )
        ds.save_episode()
    ds.finalize()


def test_extract_first_obs_writes_montage(tmp_path):
    src_dir = tmp_path / "src"
    _build_synthetic_dataset(src_dir, lengths=[2, 2, 2])  # episodes 0,1,2

    out_path = tmp_path / "out" / "first_obs.png"
    result = extract_first_obs(src_dir, [2, 0], out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_extract_first_obs_invalid_id_raises(tmp_path):
    src_dir = tmp_path / "src"
    _build_synthetic_dataset(src_dir, lengths=[2, 2])  # episodes 0,1
    out_path = tmp_path / "out" / "first_obs.png"
    with pytest.raises(ValueError, match="out of range"):
        extract_first_obs(src_dir, [0, 9], out_path)


from src.first_obs_extractor.__main__ import _read_episode_ids


def test_read_episode_ids_from_json(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"episode_ids": [3, 1, 4]}))
    assert _read_episode_ids(p) == [3, 1, 4]


def test_read_episode_ids_missing_key_raises(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"wrong_key": [1, 2]}))
    with pytest.raises(ValueError, match="episode_ids"):
        _read_episode_ids(p)
