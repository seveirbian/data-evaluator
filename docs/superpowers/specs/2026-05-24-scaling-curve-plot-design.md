# Scaling Curve 可视化设计文档

**日期**：2026-05-24
**状态**：待实现

---

## 背景

现有项目已实现论文 Eq. 9-11 的 Policy Embedding Similarity 计算（`PolicyEmbeddingSimilarityEvaluator`），可以输出单一标量 c̄_π。

论文中的 Scaling Curve 展示了随训练数据量增加，c̄_π 如何变化，直观反映数据扩展对覆盖率的提升效果。本方案为项目新增该可视化能力。

---

## 目标

- x 轴：训练 episodes 数量（从 1 到 N_total，等比间隔采样）
- y 轴：对应训练子集的 c̄_π
- 输出：保存 PNG 图片 + 可选弹出交互窗口

---

## 方案选择

选择**方案 B：新建 `ScalingCurveGenerator` 类**。

理由：
- eval embeddings 在整个曲线绘制过程中不变，在类内一次性计算并缓存，避免重复开销
- 单一职责，不污染现有 `PolicyEmbeddingSimilarityEvaluator`
- 符合现有代码风格（每个类一个明确职责）

---

## 架构

### 新增文件

```
src/scaling_curve_evaluator/
└── scaling_curve.py    # ScalingCurveGenerator 类
```

`__init__.py` 新增导出 `ScalingCurveGenerator`。

### 类设计

```python
class ScalingCurveGenerator:
    def __init__(
        self,
        policy_dir: str,
        train_data_dir: str,
        eval_data_dir: str,
        hook_module: str,
        device: str = "auto",
        num_points: int = 20,
    ): ...

    def generate(self) -> list[tuple[int, float]]:
        """计算 scaling curve 数据点，返回 [(n_episodes, c̄_π), ...]"""
        ...

    def plot(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """绘制并输出 scaling curve 图"""
        ...
```

---

## 数据流

```
__init__
  ├── PolicyEmbeddingExtractor(policy_dir, hook_module, device)
  ├── LeRobotDatasetLoader(train_data_dir)  → train_obs (全量，缓存)
  ├── LeRobotDatasetLoader(eval_data_dir)   → eval_obs
  └── 提取 eval_embs（缓存，整个过程不重复）

generate()
  ├── adaptive_steps = np.unique(np.geomspace(1, N_total, num_points).round().astype(int))
  ├── for N in adaptive_steps:
  │     train_embs = extract(train_obs[:N])
  │     score = policy_embedding_similarity(train_embs, eval_embs)
  │     results.append((N, score))
  └── return results

plot()
  ├── assert generate() 已调用，否则 RuntimeError
  ├── matplotlib 折线图（x=episodes, y=c̄_π）
  ├── 若 save_path 不为 None：mkdir -p + savefig
  └── 若 show=True：plt.show()
```

---

## 接口示例

```python
from src.scaling_curve_evaluator import ScalingCurveGenerator

generator = ScalingCurveGenerator(
    policy_dir="policy/my_policy",
    train_data_dir="data/train/dataset",
    eval_data_dir="data/eval/dataset",
    hook_module="model.backbone",
    device="auto",
    num_points=20,
)

data = generator.generate()
# [(1, 0.42), (3, 0.61), (8, 0.78), ..., (30, 1.0)]

generator.plot(save_path="scaling_curve.png", show=True)
```

---

## 错误处理

| 情况 | 处理方式 |
|------|---------|
| `plot()` 在 `generate()` 前调用 | `RuntimeError("请先调用 generate()")` |
| `num_points > N_total` | `np.unique` 去重后自动缩减，不报错 |
| `num_points == 1` | 退化为单点，正常运行 |
| `save_path` 目录不存在 | 自动 `mkdir -p`，不要求用户手动建目录 |

---

## 依赖

无新增外部依赖，`matplotlib` 已通过 `lerobot[all]` 间接引入。
