# episodes_extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `src/episodes_extractor` 模块,从一个 LeRobot v3.0 数据集中按 episode id 列表抽取子集,输出一个合法的 v3.0 数据集并记录 id 映射;提供 Python 函数与 CLI。

**Architecture:** 委托 lerobot 0.4.0 的写入器(`LeRobotDataset.create/add_frame/save_episode/finalize`)。先校验输入,再用 `episodes=` 只加载选定 episode,逐帧把源数据 `add_frame` 进新数据集,lerobot 自动完成重编号/视频重编码/meta/stats/info.json。

**Tech Stack:** Python 3.11, lerobot 0.4.0, numpy, pytest, json/argparse (标准库)

**Spec:** `docs/superpowers/specs/2026-06-10-episodes-extractor-design.md`

> **⚠️ 执行前置条件:磁盘当前 100% 满(0 字节可用)。** Task 2/3 的测试会用 lerobot 创建并写出小型数据集,需要可用磁盘。开始前必须先释放空间,否则测试会报 `OSError: Not enough disk space`。

---

### Task 1: 包骨架 + 输入校验

**Files:**
- Create: `src/episodes_extractor/__init__.py`
- Create: `src/episodes_extractor/_extractor.py`
- Create: `src/episodes_extractor/tests/__init__.py`
- Test: `src/episodes_extractor/tests/test_extractor.py`

本任务只实现**校验逻辑**(不依赖 lerobot、不需要大磁盘),把实际抽取留给 Task 2。

- [ ] **Step 1: Write the failing tests**

创建 `src/episodes_extractor/tests/__init__.py`(空文件)。

创建 `src/episodes_extractor/tests/test_extractor.py`:

```python
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
    # Should not raise; returns total_episodes
    total = _validate_inputs(src, [0, 2, 4], tmp_path / "out")
    assert total == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name '_validate_inputs'`

- [ ] **Step 3: Write minimal implementation**

创建 `src/episodes_extractor/__init__.py`:

```python
from ._extractor import extract_episodes

__all__ = ["extract_episodes"]
```

创建 `src/episodes_extractor/_extractor.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def _validate_inputs(
    src_dir: str | Path,
    episode_ids: list[int],
    out_dir: str | Path,
) -> int:
    """Validate extraction inputs. Returns source total_episodes.

    Raises ValueError / FileNotFoundError on invalid inputs.
    """
    src_dir = Path(src_dir)
    info_path = src_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Source dataset has no meta/info.json: {info_path}")
    total_episodes = int(json.loads(info_path.read_text())["total_episodes"])

    if not episode_ids:
        raise ValueError("episode_ids is empty.")

    duplicates = sorted({e for e in episode_ids if episode_ids.count(e) > 1})
    if duplicates:
        raise ValueError(f"episode_ids contains duplicate ids: {duplicates}")

    out_of_range = sorted(e for e in episode_ids if e < 0 or e >= total_episodes)
    if out_of_range:
        raise ValueError(
            f"episode_ids out of range [0, {total_episodes}): {out_of_range}"
        )

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"out_dir already exists and is not empty: {out_dir}")

    return total_episodes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/episodes_extractor/__init__.py src/episodes_extractor/_extractor.py src/episodes_extractor/tests/__init__.py src/episodes_extractor/tests/test_extractor.py
git commit -m "[feat] episodes_extractor: input validation"
```

---

### Task 2: 核心抽取(lerobot 写入)

**Files:**
- Modify: `src/episodes_extractor/_extractor.py`
- Test: `src/episodes_extractor/tests/test_extractor.py`

> 需要可用磁盘(见执行前置条件)。

- [ ] **Step 1: Write the failing round-trip test**

在 `src/episodes_extractor/tests/test_extractor.py` 末尾追加:

```python
import numpy as np

from src.episodes_extractor._extractor import extract_episodes


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
            "shape": (3, 16, 16),
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
                    "observation.images.cam": np.zeros((3, 16, 16), dtype=np.float32),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -k round_trip -v`
Expected: FAIL with `ImportError: cannot import name 'extract_episodes'` (function not defined yet)

- [ ] **Step 3: Write the implementation**

在 `src/episodes_extractor/_extractor.py` 末尾追加(顶部 `from __future__` 已存在):

```python
from collections import defaultdict

# Auto-managed by lerobot's writer; must NOT be placed into a frame dict.
_DEFAULT_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _tuple_shapes(features: dict) -> dict:
    """Return a copy of features with every 'shape' coerced to a tuple.

    lerobot's frame validator compares actual numpy shape (tuple) against the
    feature's stored shape with `!=`; a list shape never equals a tuple, so
    shapes loaded from info.json (lists) must be tuples for add_frame to pass.
    """
    out = {}
    for key, spec in features.items():
        spec = dict(spec)
        if "shape" in spec and spec["shape"] is not None:
            spec["shape"] = tuple(spec["shape"])
        out[key] = spec
    return out


def extract_episodes(
    src_dir: str | Path,
    episode_ids: list[int],
    out_dir: str | Path,
    repo_id: str = "extracted",
) -> dict[int, int]:
    """Extract selected episodes into a new LeRobot v3.0 dataset.

    Args:
        src_dir: source v3.0 dataset root.
        episode_ids: original episode ids to keep; list order defines new ids
            (new episode_index = position in this list).
        out_dir: output dataset root (must be empty or non-existent).
        repo_id: repo id for the output dataset.

    Returns:
        {original_episode_id: new_episode_id}.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    _validate_inputs(src_dir, episode_ids, out_dir)

    src = LeRobotDataset("source", root=src_dir, episodes=sorted(episode_ids))

    out = LeRobotDataset.create(
        repo_id,
        fps=src.fps,
        features=_tuple_shapes(src.features),
        root=out_dir,
        robot_type=src.meta.robot_type,
        use_videos=True,
    )

    feature_keys = [k for k in src.features if k not in _DEFAULT_FEATURES]

    # Group source global frame indices by original episode id (no video decode).
    groups: dict[int, list[int]] = defaultdict(list)
    for global_i, eid in enumerate(src.hf_dataset["episode_index"]):
        groups[int(eid)].append(global_i)

    mapping: dict[int, int] = {}
    for new_id, orig_id in enumerate(episode_ids):
        for global_i in groups[int(orig_id)]:
            item = src[global_i]
            frame = {k: item[k] for k in feature_keys}
            frame["task"] = item["task"]
            out.add_frame(frame)
        out.save_episode()
        mapping[int(orig_id)] = new_id

    out.finalize()

    (out_dir / "extraction_mapping.json").write_text(
        json.dumps({str(k): v for k, v in mapping.items()}, indent=2)
    )
    return mapping
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -v`
Expected: PASS (all Task 1 + Task 2 tests). The round-trip test may take some seconds (video encode).

- [ ] **Step 5: Commit**

```bash
git add src/episodes_extractor/_extractor.py src/episodes_extractor/tests/test_extractor.py
git commit -m "[feat] episodes_extractor: core extraction via lerobot writer"
```

---

### Task 3: CLI 入口

**Files:**
- Create: `src/episodes_extractor/__main__.py`
- Test: `src/episodes_extractor/tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

在 `src/episodes_extractor/tests/test_extractor.py` 末尾追加:

```python
from src.episodes_extractor.__main__ import _read_episode_ids


def test_read_episode_ids_from_json(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"episode_ids": [3, 1, 4]}))
    assert _read_episode_ids(p) == [3, 1, 4]


def test_read_episode_ids_missing_key_raises(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"wrong_key": [1, 2]}))
    with pytest.raises(ValueError, match="episode_ids"):
        _read_episode_ids(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -k read_episode_ids -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (`__main__` or `_read_episode_ids` missing)

- [ ] **Step 3: Write the implementation**

创建 `src/episodes_extractor/__main__.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._extractor import extract_episodes


def _read_episode_ids(json_path: str | Path) -> list[int]:
    """Read {"episode_ids": [...]} from a JSON file."""
    data = json.loads(Path(json_path).read_text())
    if not isinstance(data, dict) or "episode_ids" not in data:
        raise ValueError(
            f"JSON must be an object with an 'episode_ids' key: {json_path}"
        )
    return [int(e) for e in data["episode_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.episodes_extractor",
        description="Extract a subset of episodes from a LeRobot v3.0 dataset.",
    )
    parser.add_argument("--src", required=True, help="source v3.0 dataset dir")
    parser.add_argument(
        "--episodes", required=True, help='JSON file: {"episode_ids": [...]}'
    )
    parser.add_argument("--out", required=True, help="output dataset dir")
    parser.add_argument("--repo-id", default="extracted", help="output repo id")
    args = parser.parse_args()

    episode_ids = _read_episode_ids(args.episodes)
    mapping = extract_episodes(args.src, episode_ids, args.out, repo_id=args.repo_id)
    print(f"Extracted {len(mapping)} episodes to {args.out}")
    print(f"id mapping (original -> new): {mapping}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -k read_episode_ids -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full module test suite**

Run: `uv run pytest src/episodes_extractor/tests/test_extractor.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add src/episodes_extractor/__main__.py src/episodes_extractor/tests/test_extractor.py
git commit -m "[feat] episodes_extractor: CLI entry point"
```

---

## Self-Review

**Spec coverage:**
- 模块位置 `src/episodes_extractor` + 文件结构 → Task 1/2/3 创建 `__init__.py`/`_extractor.py`/`__main__.py`/`tests/` ✓
- `extract_episodes(src_dir, episode_ids, out_dir, repo_id) -> dict[int,int]` → Task 2 ✓
- 校验(空/重复/越界/缺 info.json/out_dir 非空) → Task 1 `_validate_inputs` + 测试 ✓
- 委托 lerobot 写入器 create/add_frame/save_episode/finalize → Task 2 ✓
- 只加载选定 episode(`episodes=sorted(...)`) → Task 2 ✓
- 新 id = `episode_ids` 列表下标(按用户顺序) → Task 2 循环 `enumerate(episode_ids)` + round-trip 测试断言 `{2:0,0:1}` ✓
- 帧构造剔除 `DEFAULT_FEATURES` + 加 task,图像原样传(lerobot 内部转置) → Task 2 ✓
- 映射文件 `out_dir/extraction_mapping.json` + 返回 dict → Task 2 + 测试 ✓
- CLI `python -m src.episodes_extractor --src --episodes --out --repo-id`,读 JSON 的 `episode_ids` → Task 3 ✓
- round-trip 主测 + 校验测试 → Task 1/2 测试 ✓
- 已验证风险(shape list→tuple)→ Task 2 `_tuple_shapes` 处理 ✓

**Placeholder scan:** 无 TBD/TODO,所有步骤含完整代码与命令。

**Type consistency:** `_validate_inputs(src_dir, episode_ids, out_dir) -> int`、`extract_episodes(...) -> dict[int,int]`、`_tuple_shapes(features) -> dict`、`_read_episode_ids(path) -> list[int]` 在定义与调用处签名一致;`_DEFAULT_FEATURES` 命名一致;映射文件名 `extraction_mapping.json` 在实现与测试中一致。
