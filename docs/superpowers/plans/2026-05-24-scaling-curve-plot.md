# Scaling Curve Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `ScalingCurveGenerator` 类，通过对训练数据集按等比间隔采样，计算每个子集的 c̄_π，并绘制 Scaling Curve 折线图（支持保存 PNG 和弹窗展示）。

**Architecture:** 新建 `scaling_curve.py`，包含纯函数 `_compute_steps`（生成采样点）和 `ScalingCurveGenerator` 类。`__init__` 阶段一次性提取 eval embeddings 并缓存；`generate()` 阶段一次性提取全量 train embeddings 再按 N 切片，避免重复推理；`plot()` 使用 matplotlib 绘图。

**Tech Stack:** Python 3.11, PyTorch, NumPy, Matplotlib, pytest

---

## File Structure

| 操作 | 路径 | 职责 |
|------|------|------|
| 新建 | `src/scaling_curve_evaluator/scaling_curve.py` | `_compute_steps` + `ScalingCurveGenerator` |
| 新建 | `tests/__init__.py` | 测试包标记 |
| 新建 | `tests/test_scaling_curve.py` | 所有单元测试 |
| 修改 | `src/scaling_curve_evaluator/__init__.py` | 导出 `ScalingCurveGenerator` |

---

## Task 1: 实现并测试 `_compute_steps` 纯函数

**Files:**
- Create: `src/scaling_curve_evaluator/scaling_curve.py`
- Create: `tests/__init__.py`
- Create: `tests/test_scaling_curve.py`

- [ ] **Step 1: 创建测试文件，写两个失败测试**

`tests/__init__.py` — 空文件即可。

`tests/test_scaling_curve.py`:
```python
from src.scaling_curve_evaluator.scaling_curve import _compute_steps


def test_compute_steps_bounds():
    steps = _compute_steps(n_total=30, num_points=10)
    assert steps[0] == 1
    assert steps[-1] == 30
    assert all(isinstance(s, int) for s in steps)
    assert len(steps) <= 10


def test_compute_steps_deduplication():
    # 数据量少时去重后应不超过 n_total 个点
    steps = _compute_steps(n_total=3, num_points=20)
    assert steps == [1, 2, 3]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_scaling_curve.py -v
```
预期：`ImportError` 或 `ModuleNotFoundError`（文件尚未创建）

- [ ] **Step 3: 实现 `_compute_steps` 和文件骨架**

新建 `src/scaling_curve_evaluator/scaling_curve.py`：
```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .dataset import LeRobotDatasetLoader
from .embeddings import PolicyEmbeddingExtractor
from .similarity import policy_embedding_similarity


def _compute_steps(n_total: int, num_points: int) -> list[int]:
    """Return sorted unique episode counts for scaling curve x-axis.

    Uses geometric spacing so small-N region is sampled more densely.
    Always includes n_total as the last point.
    """
    raw = np.geomspace(1, n_total, num_points)
    return np.unique(raw.round().astype(int)).tolist()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_scaling_curve.py -v
```
预期：`2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scaling_curve_evaluator/scaling_curve.py tests/__init__.py tests/test_scaling_curve.py
git commit -m "feat: add _compute_steps and test"
```

---

## Task 2: 实现 `ScalingCurveGenerator.__init__` 和 `generate()`

**Files:**
- Modify: `src/scaling_curve_evaluator/scaling_curve.py`
- Modify: `tests/test_scaling_curve.py`

- [ ] **Step 1: 写 `generate()` 的测试（使用 mock）**

在 `tests/test_scaling_curve.py` 末尾追加：

```python
import torch
from unittest.mock import patch, MagicMock
from src.scaling_curve_evaluator.scaling_curve import ScalingCurveGenerator


def _make_obs(n: int) -> list[dict]:
    return [{"cam": torch.zeros(3, 4, 4)} for _ in range(n)]


@patch("src.scaling_curve_evaluator.scaling_curve.PolicyEmbeddingExtractor")
@patch("src.scaling_curve_evaluator.scaling_curve.LeRobotDatasetLoader")
def test_generate_returns_correct_structure(MockLoader, MockExtractor):
    # train: 5 episodes, eval: 2 episodes
    MockLoader.return_value.camera_keys = ["cam"]
    MockLoader.return_value.get_initial_observations.side_effect = [
        _make_obs(5),  # train
        _make_obs(2),  # eval
    ]
    MockExtractor.return_value.extract_per_camera.return_value = {
        "cam": torch.zeros(128)
    }

    gen = ScalingCurveGenerator("p", "t", "e", "m", num_points=3)
    results = gen.generate()

    assert isinstance(results, list)
    assert len(results) >= 1
    n, score = results[0]
    assert isinstance(n, int)
    assert isinstance(score, float)
    assert results[-1][0] == 5  # 最后一点必须是 n_total
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_scaling_curve.py::test_generate_returns_correct_structure -v
```
预期：`AttributeError` 或 `ImportError`（类尚未实现）

- [ ] **Step 3: 实现 `ScalingCurveGenerator.__init__` 和 `generate()`**

在 `scaling_curve.py` 中 `_compute_steps` 函数之后追加：

```python
class ScalingCurveGenerator:
    """Generate and plot a scaling curve of c̄_π vs training episodes.

    Eval embeddings are computed once at init and reused across all steps.
    All train embeddings are extracted once in generate(), then sliced per step.

    Usage::

        gen = ScalingCurveGenerator(
            policy_dir="policy/my_policy",
            train_data_dir="data/train/dataset",
            eval_data_dir="data/eval/dataset",
            hook_module="model.backbone",
        )
        gen.generate()
        gen.plot(save_path="curve.png", show=True)
    """

    def __init__(
        self,
        policy_dir: str,
        train_data_dir: str,
        eval_data_dir: str,
        hook_module: str,
        device: str = "auto",
        num_points: int = 20,
    ):
        self.num_points = num_points
        self.extractor = PolicyEmbeddingExtractor(policy_dir, hook_module, device)

        train_loader = LeRobotDatasetLoader(train_data_dir)
        eval_loader = LeRobotDatasetLoader(eval_data_dir)

        self.camera_keys = sorted(
            set(train_loader.camera_keys) & set(eval_loader.camera_keys)
        )
        if not self.camera_keys:
            raise ValueError(
                "No common camera keys between train and eval datasets.\n"
                f"  Train cameras: {train_loader.camera_keys}\n"
                f"  Eval  cameras: {eval_loader.camera_keys}"
            )

        self._train_obs = train_loader.get_initial_observations()
        eval_obs = eval_loader.get_initial_observations()

        print(f"Train episodes: {len(self._train_obs)}, Eval episodes: {len(eval_obs)}")
        print(f"Cameras: {self.camera_keys}")

        # Pre-compute eval embeddings once — reused for every scaling step
        self._eval_embs = self._extract_embeddings(eval_obs, "Eval embeddings")
        self._results: list[tuple[int, float]] | None = None

    def _extract_embeddings(
        self, observations: list[dict], desc: str
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings for all observations. Returns {camera_key: [N, D]}."""
        per_camera: dict[str, list[torch.Tensor]] = {k: [] for k in self.camera_keys}
        for obs in tqdm(observations, desc=desc, unit="ep"):
            embs = self.extractor.extract_per_camera(obs, self.camera_keys)
            for k in self.camera_keys:
                per_camera[k].append(embs[k])
        return {k: torch.stack(vs, dim=0) for k, vs in per_camera.items()}

    def generate(self) -> list[tuple[int, float]]:
        """Compute c̄_π for each training subset. Returns [(n_episodes, score), ...]."""
        n_total = len(self._train_obs)
        steps = _compute_steps(n_total, self.num_points)

        # Extract all train embeddings once, then slice per step
        all_train_embs = self._extract_embeddings(self._train_obs, "Train embeddings")

        self._results = []
        for n in tqdm(steps, desc="Scaling curve", unit="step"):
            train_embs_n = {k: v[:n] for k, v in all_train_embs.items()}
            score = policy_embedding_similarity(train_embs_n, self._eval_embs)
            self._results.append((n, score))

        return self._results
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_scaling_curve.py -v
```
预期：`3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scaling_curve_evaluator/scaling_curve.py tests/test_scaling_curve.py
git commit -m "feat: implement ScalingCurveGenerator init and generate()"
```

---

## Task 3: 实现并测试 `plot()`

**Files:**
- Modify: `src/scaling_curve_evaluator/scaling_curve.py`
- Modify: `tests/test_scaling_curve.py`

- [ ] **Step 1: 写 `plot()` 的两个测试**

在 `tests/test_scaling_curve.py` 末尾追加：

```python
import pytest
from src.scaling_curve_evaluator.scaling_curve import ScalingCurveGenerator


def _generator_with_results(results):
    """Bypass __init__ and inject results directly."""
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = results
    return gen


def test_plot_raises_if_generate_not_called():
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = None
    with pytest.raises(RuntimeError, match="generate"):
        gen.plot()


def test_plot_saves_file(tmp_path):
    gen = _generator_with_results([(1, 0.4), (5, 0.7), (10, 1.0)])
    save_path = tmp_path / "subdir" / "curve.png"
    gen.plot(save_path=str(save_path))
    assert save_path.exists()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_scaling_curve.py::test_plot_raises_if_generate_not_called tests/test_scaling_curve.py::test_plot_saves_file -v
```
预期：`AttributeError: 'ScalingCurveGenerator' object has no attribute 'plot'`

- [ ] **Step 3: 实现 `plot()` 方法**

在 `ScalingCurveGenerator` 类的 `generate()` 方法之后追加：

```python
    def plot(self, save_path: str | None = None, show: bool = False) -> None:
        """Plot the scaling curve.

        Args:
            save_path: If given, save the figure to this path (PNG). Parent
                directories are created automatically.
            show: If True, display an interactive matplotlib window.
        """
        if self._results is None:
            raise RuntimeError(
                "请先调用 generate() 再调用 plot()。"
            )

        import matplotlib.pyplot as plt

        xs = [r[0] for r in self._results]
        ys = [r[1] for r in self._results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("c̄_π")
        ax.set_title("Policy Embedding Similarity — Scaling Curve")
        ax.grid(True, linestyle="--", alpha=0.5)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        plt.close(fig)
```

- [ ] **Step 4: 运行所有测试，确认通过**

```bash
uv run pytest tests/test_scaling_curve.py -v
```
预期：`5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scaling_curve_evaluator/scaling_curve.py tests/test_scaling_curve.py
git commit -m "feat: implement ScalingCurveGenerator.plot()"
```

---

## Task 4: 导出 `ScalingCurveGenerator` 并验证

**Files:**
- Modify: `src/scaling_curve_evaluator/__init__.py`

- [ ] **Step 1: 更新 `__init__.py`**

将 `src/scaling_curve_evaluator/__init__.py` 改为：

```python
from .evaluator import PolicyEmbeddingSimilarityEvaluator
from .scaling_curve import ScalingCurveGenerator

__all__ = ["PolicyEmbeddingSimilarityEvaluator", "ScalingCurveGenerator"]
```

- [ ] **Step 2: 验证导入正常**

```bash
uv run python -c "from src.scaling_curve_evaluator import ScalingCurveGenerator; print('OK')"
```
预期：`OK`

- [ ] **Step 3: 运行全部测试，确认无回归**

```bash
uv run pytest tests/ -v
```
预期：`5 passed`

- [ ] **Step 4: Commit**

```bash
git add src/scaling_curve_evaluator/__init__.py
git commit -m "feat: export ScalingCurveGenerator from package"
```

---

## Task 5: 端到端烟测

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 更新 `main.py`，加入 ScalingCurveGenerator 示例**

将 `main.py` 改为：

```python
from src.scaling_curve_evaluator import ScalingCurveGenerator


def main():
    gen = ScalingCurveGenerator(
        policy_dir="example/policy/act_policy_grabcuicuishaplacematting-30",
        train_data_dir="example/data/train/Task-GrabCuicuishaPlaceMatting-30",
        eval_data_dir="example/data/eval/Task-GrabCuicuishaPlaceMatting-30",
        hook_module="model.backbone",
        device="auto",
        num_points=20,
    )
    gen.generate()
    gen.plot(save_path="scaling_curve.png", show=False)
    print("Done. Saved to scaling_curve.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行烟测**

```bash
uv run main.py
```
预期：打印进度条，最终输出 `Done. Saved to scaling_curve.png`，且 `scaling_curve.png` 文件存在。

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: update main.py with ScalingCurveGenerator end-to-end example"
```
