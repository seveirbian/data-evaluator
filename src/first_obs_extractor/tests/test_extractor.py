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
