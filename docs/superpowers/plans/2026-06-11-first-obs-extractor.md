# first_obs_extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `src/first_obs_extractor` 模块,把数据集中选定 episode 的第一帧观测拼成一张大图(PNG);并把 `LeRobotDatasetLoader` 提升为共享模块 `src/dataset_io.py`。

**Architecture:** 复用现成的 `LeRobotDatasetLoader.get_initial_observations()`(取每个 episode 的首帧观测),只做"加载 → 收集 (episode,相机) cell → matplotlib 近方形网格渲染 → 保存"。先把 loader 移到共享模块,消除跨 feature 私有耦合。

**Tech Stack:** Python 3.11, matplotlib (Agg), numpy, lerobot 0.4.0(仅造测试数据), pytest, json/argparse

**Spec:** `docs/superpowers/specs/2026-06-11-first-obs-extractor-design.md`

---

### Task 1: 把 LeRobotDatasetLoader 提升为共享模块

**Files:**
- Move: `src/scaling_curve/_dataset.py` → `src/dataset_io.py`
- Modify: `src/scaling_curve/scaling_curve.py:9`
- Modify: `main.py:10`

- [ ] **Step 1: Move the file with git**

Run:
```bash
git mv src/scaling_curve/_dataset.py src/dataset_io.py
```

- [ ] **Step 2: Update the import in `src/scaling_curve/scaling_curve.py`**

Change line 9 from:
```python
from ._dataset import LeRobotDatasetLoader
```
to:
```python
from src.dataset_io import LeRobotDatasetLoader
```

- [ ] **Step 3: Update the import in `main.py`**

Change line 10 from:
```python
from src.scaling_curve._dataset import LeRobotDatasetLoader
```
to:
```python
from src.dataset_io import LeRobotDatasetLoader
```

- [ ] **Step 4: Verify scaling_curve tests still pass (patch target unchanged)**

Run: `uv run pytest src/scaling_curve/tests/test_scaling_curve.py -q`
Expected: all pass (the `@patch("src.scaling_curve.scaling_curve.LeRobotDatasetLoader")` target still resolves because scaling_curve.py re-imports the name into its namespace).

- [ ] **Step 5: Verify main imports cleanly**

Run: `uv run python -c "import main"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "[refactor] promote LeRobotDatasetLoader to shared src/dataset_io.py"
```

---

### Task 2: 包骨架 + 校验 + `_grid_dims`

**Files:**
- Create: `src/first_obs_extractor/__init__.py`
- Create: `src/first_obs_extractor/_extractor.py`
- Create: `src/first_obs_extractor/tests/__init__.py`
- Test: `src/first_obs_extractor/tests/test_extractor.py`

不依赖 matplotlib/数据集,纯逻辑。

- [ ] **Step 1: Write the failing tests**

创建空文件 `src/first_obs_extractor/tests/__init__.py`。

创建 `src/first_obs_extractor/tests/test_extractor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -v`
Expected: FAIL with `ImportError` (module/functions not defined).

- [ ] **Step 3: Write minimal implementation**

创建 `src/first_obs_extractor/__init__.py`:
```python
from ._extractor import extract_first_obs

__all__ = ["extract_first_obs"]
```

创建 `src/first_obs_extractor/_extractor.py`:
```python
from __future__ import annotations

import json
import math
from pathlib import Path


def _grid_dims(n: int) -> tuple[int, int]:
    """Near-square grid: returns (rows, cols) for n cells."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _validate_episode_ids(src_dir: str | Path, episode_ids: list[int]) -> int:
    """Validate episode ids against the source dataset. Returns total_episodes."""
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

    return total_episodes


def extract_first_obs(src_dir, episode_ids, out_path):
    raise NotImplementedError("Implemented in a later task.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -v`
Expected: PASS (11 passed: 6 grid + 5 validation).

- [ ] **Step 5: Commit**

```bash
git add src/first_obs_extractor/__init__.py src/first_obs_extractor/_extractor.py src/first_obs_extractor/tests/__init__.py src/first_obs_extractor/tests/test_extractor.py
git commit -m "[feat] first_obs_extractor: grid dims + input validation"
```

---

### Task 3: `_render_montage` 渲染网格

**Files:**
- Modify: `src/first_obs_extractor/_extractor.py`
- Test: `src/first_obs_extractor/tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

在 `src/first_obs_extractor/tests/test_extractor.py` 末尾追加:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -k render_montage -v`
Expected: FAIL with `ImportError: cannot import name '_render_montage'`.

- [ ] **Step 3: Write implementation**

在 `src/first_obs_extractor/_extractor.py` 末尾追加(顶部 `from __future__` 已存在):
```python
import numpy as np


def _to_hwc_image(value) -> np.ndarray:
    """Convert a torch tensor / numpy array image to a HWC numpy array.

    Decoded video frames are channel-first [C,H,W]; matplotlib's imshow wants
    [H,W,C]. Float images are left in [0,1] (clipped) for imshow.
    """
    arr = value
    if hasattr(arr, "detach"):  # torch tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def _render_montage(cells: list[tuple[str, object]], out_path: str | Path) -> Path:
    """Render labelled first-frame cells into a near-square montage PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cells)
    rows, cols = _grid_dims(n)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), squeeze=False
    )
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx < n:
            label, image = cells[idx]
            ax.imshow(_to_hwc_image(image))
            ax.set_title(label, fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -k render_montage -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/first_obs_extractor/_extractor.py src/first_obs_extractor/tests/test_extractor.py
git commit -m "[feat] first_obs_extractor: montage renderer"
```

---

### Task 4: `extract_first_obs` 编排 + 合成数据集 round-trip

**Files:**
- Modify: `src/first_obs_extractor/_extractor.py`
- Test: `src/first_obs_extractor/tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

在 `src/first_obs_extractor/tests/test_extractor.py` 末尾追加:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -k first_obs_writes -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Replace the stub implementation**

在 `src/first_obs_extractor/_extractor.py` 中,把:
```python
def extract_first_obs(src_dir, episode_ids, out_path):
    raise NotImplementedError("Implemented in a later task.")
```
替换为:
```python
def extract_first_obs(
    src_dir: str | Path,
    episode_ids: list[int],
    out_path: str | Path,
) -> Path:
    """Tile the first-frame observations of selected episodes into one PNG.

    Args:
        src_dir: LeRobot v2.0/v3.0 dataset root.
        episode_ids: episode ids to include; output cells follow this order,
            then cameras in sorted key order.
        out_path: output PNG path (overwritten if it exists).

    Returns:
        Path(out_path).
    """
    from src.dataset_io import LeRobotDatasetLoader

    src_dir = Path(src_dir)
    _validate_episode_ids(src_dir, episode_ids)

    loader = LeRobotDatasetLoader(src_dir)
    observations = loader.get_initial_observations()
    cameras = sorted(loader.camera_keys)

    cells: list[tuple[str, object]] = []
    for ep_id in episode_ids:
        ep_obs = observations[ep_id]
        for cam in cameras:
            short = cam.split(".")[-1]
            cells.append((f"ep{ep_id} / {short}", ep_obs[cam]))

    return _render_montage(cells, out_path)
```

- [ ] **Step 4: Run the full module test suite**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -v`
Expected: all pass (grid + validation + render + round-trip + invalid-id).

- [ ] **Step 5: Commit**

```bash
git add src/first_obs_extractor/_extractor.py src/first_obs_extractor/tests/test_extractor.py
git commit -m "[feat] first_obs_extractor: extract_first_obs orchestrator"
```

---

### Task 5: CLI 入口

**Files:**
- Create: `src/first_obs_extractor/__main__.py`
- Test: `src/first_obs_extractor/tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

在 `src/first_obs_extractor/tests/test_extractor.py` 末尾追加:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -k read_episode_ids -v`
Expected: FAIL with `ImportError` (`__main__` / `_read_episode_ids` missing).

- [ ] **Step 3: Write implementation**

创建 `src/first_obs_extractor/__main__.py`:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._extractor import extract_first_obs


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
        prog="python -m src.first_obs_extractor",
        description="Tile first-frame observations of selected episodes into one image.",
    )
    parser.add_argument("--src", required=True, help="LeRobot v2.0/v3.0 dataset dir")
    parser.add_argument(
        "--episodes", required=True, help='JSON file: {"episode_ids": [...]}'
    )
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args()

    episode_ids = _read_episode_ids(args.episodes)
    out = extract_first_obs(args.src, episode_ids, args.out)
    print(f"Saved first-observation montage to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -k read_episode_ids -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify the full suite + CLI help**

Run: `uv run pytest src/first_obs_extractor/tests/test_extractor.py -q`
Expected: all pass.

Run: `uv run python -m src.first_obs_extractor --help`
Expected: usage with `--src/--episodes/--out`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/first_obs_extractor/__main__.py src/first_obs_extractor/tests/test_extractor.py
git commit -m "[feat] first_obs_extractor: CLI entry point"
```

---

## Self-Review

**Spec coverage:**
- 复用方案 1:移动 loader 到 `src/dataset_io.py` + 改 2 处 import,patch 不受影响 → Task 1 ✓
- 模块文件结构 → Task 2/3/4/5 ✓
- `extract_first_obs(src_dir, episode_ids, out_path) -> Path` → Task 4 ✓
- 校验(空/重复/越界/缺 info.json) → Task 2 `_validate_episode_ids` + 测试 ✓
- 收集 cell(给定顺序 × sorted 相机,标 `ep{id} / {短名}`) → Task 4 ✓
- 近方形网格 `cols=ceil(sqrt(N)), rows=ceil(N/cols)` → Task 2 `_grid_dims` + Task 3 渲染 ✓
- CHW→HWC、float clip、imshow、关坐标轴、多余格留白 → Task 3 `_to_hwc_image` / `_render_montage` ✓
- 覆盖已存在文件、自动建父目录 → Task 3 (`mkdir parents` + `savefig`) ✓
- CLI `python -m src.first_obs_extractor --src --episodes --out`,读 `episode_ids` → Task 5 ✓
- 测试:`_grid_dims`、`_render_montage`(HWC+CHW)、round-trip、校验 → Task 2/3/4 ✓

**Placeholder scan:** 无 TBD/TODO;每个代码步骤含完整代码与命令。

**Type consistency:** `_grid_dims(n)->(rows,cols)`、`_validate_episode_ids(src_dir,episode_ids)->int`、`_to_hwc_image(value)->ndarray`、`_render_montage(cells,out_path)->Path`、`extract_first_obs(src_dir,episode_ids,out_path)->Path`、`_read_episode_ids(path)->list[int]` 在定义与调用处一致;`cells` 元素结构 `(label, image)` 在 render/orchestrator/test 中一致。
