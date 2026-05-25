# 多数据集 Scaling Curve 设计文档

**日期**：2026-05-25
**状态**：待实现

---

## 背景

现有 `ScalingCurveGenerator` 支持单个训练数据集生成一条 scaling curve。用户需要在同一张图上叠加多条曲线，以对比不同数据集或不同 policy 的覆盖效果。

---

## 目标

- 支持任意数量的（policy, train_data_dir）组合，各生成一条 scaling curve
- 共用同一个 eval 数据集
- 所有曲线叠加在同一张 matplotlib 图上，带 legend
- Label 自动从目录名推断，冲突时追加 policy 名区分
- 输出支持保存 PNG + 可选弹窗（与 `ScalingCurveGenerator.plot()` 一致）

---

## 方案

新建 `MultiScalingCurvePlotter` 类，放在现有 `scaling_curve.py` 中。内部为每条曲线创建一个 `ScalingCurveGenerator` 实例，复用其全部逻辑。不修改 `ScalingCurveGenerator`。

---

## 架构

### 文件变更

| 操作 | 路径 | 变更内容 |
|------|------|---------|
| 修改 | `src/scaling_curve_evaluator/scaling_curve.py` | 新增 `MultiScalingCurvePlotter` 类 |
| 修改 | `src/scaling_curve_evaluator/__init__.py` | 导出 `MultiScalingCurvePlotter` |

### 类设计

```python
class MultiScalingCurvePlotter:
    def __init__(
        self,
        eval_data_dir: str,
        curves: list[dict],   # [{"policy_dir": ..., "train_data_dir": ..., "hook_module": ...}, ...]
        device: str = "auto",
        num_points: int = 20,
    ): ...

    def generate_all(self) -> None:
        """为每条曲线调用 generator.generate()。"""
        ...

    def plot(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """将所有曲线叠加绘制在同一张图上。"""
        ...
```

---

## 数据流

```
__init__
  ├── 校验 curves 非空，否则 ValueError
  ├── 推断每条曲线的 label（见下方规则）
  └── for each curve config:
        ScalingCurveGenerator(
            policy_dir, train_data_dir, eval_data_dir,
            hook_module, device, num_points
        )
        # 注：eval embeddings 在此阶段按 policy 各自计算，不可跨 policy 共享

generate_all()
  └── for each generator:
        generator.generate()

plot()
  ├── assert 每个 generator._results is not None，否则 RuntimeError
  ├── for each (generator, label):
  │     ax.plot(xs, ys, marker="o", label=label)
  │     for each (x, y): ax.annotate(f"{y:.3f}", ...)
  ├── ax.legend()
  └── save_path → mkdir -p + savefig；show → plt.show()
```

---

## Label 推断规则

```python
# 第一步：取各 train_data_dir 的目录名
labels = [Path(c["train_data_dir"]).name for c in curves]

# 第二步：若有重复，追加 policy 目录名
if len(set(labels)) < len(labels):
    labels = [
        f"{Path(c['train_data_dir']).name}/{Path(c['policy_dir']).name}"
        for c in curves
    ]
```

---

## 接口示例

```python
from src.scaling_curve_evaluator import MultiScalingCurvePlotter

plotter = MultiScalingCurvePlotter(
    eval_data_dir="data/eval/Task-Foo",
    curves=[
        {"policy_dir": "policy/act",  "train_data_dir": "data/train/batch1", "hook_module": "model.backbone"},
        {"policy_dir": "policy/pi0",  "train_data_dir": "data/train/batch2", "hook_module": "model.vision_tower"},
    ],
    device="auto",
    num_points=20,
)
plotter.generate_all()
plotter.plot(save_path="multi_curve.png", show=True)
```

---

## 错误处理

| 情况 | 处理 |
|------|------|
| `curves` 为空列表 | `ValueError("curves 列表不能为空")` |
| `plot()` 在 `generate_all()` 前调用 | 各 generator 的 `RuntimeError` 自然冒泡 |
| label 冲突 | 自动追加 policy 目录名，无需用户干预 |
| 各曲线 camera_keys 不一致 | 各 generator 独立处理，互不干扰 |

---

## 依赖

无新增外部依赖。
