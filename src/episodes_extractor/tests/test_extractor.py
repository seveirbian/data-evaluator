import json
from pathlib import Path

import pytest

from src.episodes_extractor._extractor import _validate_inputs


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
