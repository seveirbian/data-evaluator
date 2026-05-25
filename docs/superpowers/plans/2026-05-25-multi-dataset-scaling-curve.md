# 多数据集 Scaling Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `MultiScalingCurvePlotter` 类，支持将多个（policy, train_data_dir）组合的 scaling curve 叠加绘制在同一张图上。

**Architecture:** 在现有 `scaling_curve.py` 中新增 `_infer_labels` 纯函数和 `MultiScalingCurvePlotter` 类；类内部为每条曲线创建一个 `ScalingCurveGenerator` 实例，`generate_all()` 顺序调用各实例的 `generate()`，`plot()` 把所有结果叠加在同一 axes 上并绘制 legend。

**Tech Stack:** Python 3.11, PyTorch, Matplotlib, pytest

---

## File Structure

| 操作 | 路径 | 变更内容 |
|------|------|---------|
| 修改 | `src/scaling_curve_evaluator/scaling_curve.py` | 新增 `_infer_labels` + `MultiScalingCurvePlotter` |
| 修改 | `src/scaling_curve_evaluator/__init__.py` | 导出 `MultiScalingCurvePlotter` |
| 修改 | `tests/test_scaling_curve.py` | 新增对应测试 |
| 修改 | `main.py` | 端到端烟测示例 |

---

## Task 1: 实现并测试 `_infer_labels` 纯函数

**Files:**
- Modify: `src/scaling_curve_evaluator/scaling_curve.py`
- Modify: `tests/test_scaling_curve.py`

- [ ] **Step 1: 写两个失败测试**

在 `tests/test_scaling_curve.py` 末尾追加：

```python
from src.scaling_curve_evaluator.scaling_curve import _infer_labels


def test_infer_labels_no_collision():
    curves = [
        {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "m"},
        {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch2", "hook_module": "m"},
    ]
    labels = _infer_labels(curves)
    assert labels == ["batch1", "batch2"]


def test_infer_labels_collision():
    curves = [
        {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "m"},
        {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch1", "hook_module": "m"},
    ]
    labels = _infer_labels(curves)
    assert labels == ["batch1/act", "batch1/pi0"]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_scaling_curve.py::test_infer_labels_no_collision tests/test_scaling_curve.py::test_infer_labels_collision -v
```
预期：`ImportError: cannot import name '_infer_labels'`

- [ ] **Step 3: 实现 `_infer_labels`**

在 `src/scaling_curve_evaluator/scaling_curve.py` 中，`_compute_steps` 函数之后追加：

```python
def _infer_labels(curves: list[dict]) -> list[str]:
    """Infer display labels from curve configs.

    Uses train_data_dir basename by default.
    If duplicates exist, appends policy_dir basename to disambiguate.
    """
    labels = [Path(c["train_data_dir"]).name for c in curves]
    if len(set(labels)) < len(labels):
        labels = [
            f"{Path(c['train_data_dir']).name}/{Path(c['policy_dir']).name}"
            for c in curves
        ]
    return labels
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_scaling_curve.py::test_infer_labels_no_collision tests/test_scaling_curve.py::test_infer_labels_collision -v
```
预期：`2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scaling_curve_evaluator/scaling_curve.py tests/test_scaling_curve.py
git commit -m "feat: add _infer_labels and tests"
```

---

## Task 2: 实现 `MultiScalingCurvePlotter.__init__` 和 `generate_all()`

**Files:**
- Modify: `src/scaling_curve_evaluator/scaling_curve.py`
- Modify: `tests/test_scaling_curve.py`

- [ ] **Step 1: 写测试**

在 `tests/test_scaling_curve.py` 末尾追加：

```python
from src.scaling_curve_evaluator.scaling_curve import MultiScalingCurvePlotter


def test_multi_plotter_empty_curves_raises():
    with pytest.raises(ValueError, match="curves"):
        MultiScalingCurvePlotter(eval_data_dir="e", curves=[])


@patch("src.scaling_curve_evaluator.scaling_curve.ScalingCurveGenerator")
def test_generate_all_calls_each_generator(MockGen):
    MockGen.return_value.generate.return_value = [(1, 0.5), (5, 0.9)]
    plotter = MultiScalingCurvePlotter(
        eval_data_dir="e",
        curves=[
            {"policy_dir": "p1", "train_data_dir": "t1", "hook_module": "m"},
            {"policy_dir": "p2", "train_data_dir": "t2", "hook_module": "m"},
        ],
    )
    plotter.generate_all()
    assert MockGen.return_value.generate.call_count == 2
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_scaling_curve.py::test_multi_plotter_empty_curves_raises tests/test_scaling_curve.py::test_generate_all_calls_each_generator -v
```
预期：`ImportError: cannot import name 'MultiScalingCurvePlotter'`

- [ ] **Step 3: 实现 `MultiScalingCurvePlotter` 类（不含 `plot()`）**

在 `scaling_curve.py` 末尾追加：

```python
class MultiScalingCurvePlotter:
    """Plot multiple scaling curves on the same figure.

    Each curve corresponds to one (policy, train_data_dir) combination.
    A shared eval_data_dir is used for all curves.

    Usage::

        plotter = MultiScalingCurvePlotter(
            eval_data_dir="data/eval/dataset",
            curves=[
                {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "model.backbone"},
                {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch2", "hook_module": "model.vision_tower"},
            ],
        )
        plotter.generate_all()
        plotter.plot(save_path="multi_curve.png", show=True)
    """

    def __init__(
        self,
        eval_data_dir: str,
        curves: list[dict],
        device: str = "auto",
        num_points: int = 20,
    ):
        if not curves:
            raise ValueError("curves 列表不能为空。")

        self._labels = _infer_labels(curves)
        self._generators = [
            ScalingCurveGenerator(
                policy_dir=c["policy_dir"],
                train_data_dir=c["train_data_dir"],
                eval_data_dir=eval_data_dir,
                hook_module=c["hook_module"],
                device=device,
                num_points=num_points,
            )
            for c in curves
        ]

    def generate_all(self) -> None:
        """Run generate() for each curve."""
        for gen in self._generators:
            gen.generate()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_scaling_curve.py::test_multi_plotter_empty_curves_raises tests/test_scaling_curve.py::test_generate_all_calls_each_generator -v
```
预期：`2 passed`

- [ ] **Step 5: 运行全量测试，确认无回归**

```bash
uv run pytest tests/ -v
```
预期：`9 passed`

- [ ] **Step 6: Commit**

```bash
git add src/scaling_curve_evaluator/scaling_curve.py tests/test_scaling_curve.py
git commit -m "feat: implement MultiScalingCurvePlotter init and generate_all()"
```

---

## Task 3: 实现并测试 `MultiScalingCurvePlotter.plot()`

**Files:**
- Modify: `src/scaling_curve_evaluator/scaling_curve.py`
- Modify: `tests/test_scaling_curve.py`

- [ ] **Step 1: 写测试**

在 `tests/test_scaling_curve.py` 末尾追加：

```python
def _multi_plotter_with_results(results_list, labels):
    """Bypass __init__, inject generators with results directly."""
    plotter = object.__new__(MultiScalingCurvePlotter)
    plotter._labels = labels
    plotter._generators = []
    for results in results_list:
        gen = object.__new__(ScalingCurveGenerator)
        gen._results = results
        plotter._generators.append(gen)
    return plotter


def test_multi_plot_raises_if_generate_not_called():
    plotter = object.__new__(MultiScalingCurvePlotter)
    plotter._labels = ["a"]
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = None
    plotter._generators = [gen]
    with pytest.raises(RuntimeError, match="generate"):
        plotter.plot()


def test_multi_plot_saves_file(tmp_path):
    plotter = _multi_plotter_with_results(
        results_list=[
            [(1, 0.4), (5, 0.7), (10, 1.0)],
            [(1, 0.3), (5, 0.6), (10, 0.9)],
        ],
        labels=["batch1", "batch2"],
    )
    save_path = tmp_path / "multi_curve.png"
    plotter.plot(save_path=str(save_path))
    assert save_path.exists()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_scaling_curve.py::test_multi_plot_raises_if_generate_not_called tests/test_scaling_curve.py::test_multi_plot_saves_file -v
```
预期：`AttributeError: 'MultiScalingCurvePlotter' object has no attribute 'plot'`

- [ ] **Step 3: 实现 `plot()` 方法**

在 `MultiScalingCurvePlotter` 类的 `generate_all()` 方法之后追加：

```python
    def plot(self, save_path: str | None = None, show: bool = False) -> None:
        """Plot all scaling curves on the same figure.

        Args:
            save_path: If given, save the figure to this path (PNG). Parent
                directories are created automatically.
            show: If True, display an interactive matplotlib window.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))

        for gen, label in zip(self._generators, self._labels):
            if gen._results is None:
                raise RuntimeError(
                    f"Curve '{label}' 未生成数据，请先调用 generate_all()。"
                )
            xs = [r[0] for r in gen._results]
            ys = [r[1] for r in gen._results]
            (line,) = ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=label)
            for x, y in zip(xs, ys):
                ax.annotate(
                    f"{y:.3f}",
                    xy=(x, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=line.get_color(),
                )

        ax.set_xlabel("Training episodes")
        ax.set_ylabel("c̄_π")
        ax.set_title("Policy Embedding Similarity — Scaling Curves")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        plt.close(fig)
```

- [ ] **Step 4: 运行全量测试，确认通过**

```bash
uv run pytest tests/ -v
```
预期：`11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scaling_curve_evaluator/scaling_curve.py tests/test_scaling_curve.py
git commit -m "feat: implement MultiScalingCurvePlotter.plot()"
```

---

## Task 4: 导出并验证

**Files:**
- Modify: `src/scaling_curve_evaluator/__init__.py`

- [ ] **Step 1: 更新 `__init__.py`**

将 `src/scaling_curve_evaluator/__init__.py` 改为：

```python
from .evaluator import PolicyEmbeddingSimilarityEvaluator
from .scaling_curve import MultiScalingCurvePlotter, ScalingCurveGenerator

__all__ = [
    "PolicyEmbeddingSimilarityEvaluator",
    "ScalingCurveGenerator",
    "MultiScalingCurvePlotter",
]
```

- [ ] **Step 2: 验证导入正常**

```bash
uv run python -c "from src.scaling_curve_evaluator import MultiScalingCurvePlotter; print('OK')"
```
预期：`OK`

- [ ] **Step 3: 运行全量测试**

```bash
uv run pytest tests/ -v
```
预期：`11 passed`

- [ ] **Step 4: Commit**

```bash
git add src/scaling_curve_evaluator/__init__.py
git commit -m "feat: export MultiScalingCurvePlotter from package"
```

---

## Task 5: 端到端烟测

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 更新 `main.py`**

将 `main.py` 改为：

```python
from src.scaling_curve_evaluator import MultiScalingCurvePlotter


def main():
    plotter = MultiScalingCurvePlotter(
        eval_data_dir="example/data/eval/Task-GrabCuicuishaPlaceMatting-30",
        curves=[
            {
                "policy_dir": "example/policy/act_policy_grabcuicuishaplacematting-30",
                "train_data_dir": "example/data/train/Task-GrabCuicuishaPlaceMatting-30",
                "hook_module": "model.backbone",
            },
        ],
        device="auto",
        num_points=10,
    )
    plotter.generate_all()
    plotter.plot(save_path="scaling_curve.png", show=False)
    print("Done. Saved to scaling_curve.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行烟测**

```bash
uv run main.py
```
预期：进度条正常，最终输出 `Done. Saved to scaling_curve.png`，文件存在。

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: update main.py to use MultiScalingCurvePlotter"
```
