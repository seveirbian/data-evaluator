# Refactor: Clean Public API + Limitations Documentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the package to expose only `ScalingCurveGenerator` and `MultiScalingCurveGenerator` as the public API, mark internal modules with underscore prefix, delete the unused `PolicyEmbeddingSimilarityEvaluator`, rename `MultiScalingCurvePlotter` → `MultiScalingCurveGenerator`, and document usage limitations in both README and class docstrings.

**Architecture:** Three-layer structure — public API layer (`scaling_curve.py`), internal implementation layer (`_dataset.py`, `_embeddings.py`, `_similarity.py`), and package entry (`__init__.py`). The public layer depends on internals; internals are unaware of scaling curve logic. The similarity module is a pure stateless function.

**Tech Stack:** Python 3.11, PyTorch, Matplotlib, pytest

---

## Architecture

```
公开 API 层       scaling_curve.py
                  ScalingCurveGenerator
                  MultiScalingCurveGenerator   (renamed from MultiScalingCurvePlotter)

内部实现层        _dataset.py     → LeRobotDatasetLoader
                  _embeddings.py  → PolicyEmbeddingExtractor
                  _similarity.py  → policy_embedding_similarity

包入口            __init__.py     → 只导出两个公开类
```

---

## File Changes

| 操作 | 路径 | 变更内容 |
|------|------|---------|
| 删除 | `src/scaling_curve_evaluator/evaluator.py` | 删除 PolicyEmbeddingSimilarityEvaluator |
| 重命名 | `dataset.py` → `_dataset.py` | 加下划线前缀，标识内部模块 |
| 重命名 | `embeddings.py` → `_embeddings.py` | 同上 |
| 重命名 | `similarity.py` → `_similarity.py` | 同上 |
| 修改 | `scaling_curve.py` | 更新 import 路径；`MultiScalingCurvePlotter` → `MultiScalingCurveGenerator` |
| 修改 | `__init__.py` | 只导出 `ScalingCurveGenerator`, `MultiScalingCurveGenerator` |
| 修改 | `tests/test_scaling_curve.py` | 更新所有 `MultiScalingCurvePlotter` 引用 |
| 修改 | `tests/test_similarity.py` 等 | 更新 import 路径 |
| 修改 | `main.py` | 更新 import |
| 修改 | `README.md` | 重写：加"使用前提与限制"章节，移除 Evaluator 用法 |

---

## Limitations

以下限制写入 README.md 的"使用前提与限制"章节，以及 `ScalingCurveGenerator` 和 `MultiScalingCurveGenerator` 的 docstring。

1. **LeRobot v3.0 格式**：数据集必须是 LeRobot v3.0 格式（Parquet + MP4 + `meta/info.json`）。不支持其他格式。

2. **Policy 与数据必须同构**：Policy 的 `input_shapes`（机器人关节数、状态维度）必须与数据集匹配。用 6-DOF 机器人训练的 policy 无法处理 14-DOF 机器人的数据，会在 forward pass 时报 shape 错误。

3. **相机视角必须有交集**：Train 和 eval 数据集至少共享一个相机视角（物理视角相同）。不同命名可通过 `camera_key_map` 参数处理，但不能替代真实的视角对应。

4. **只使用初始帧**：每个 episode 只取 `frame_index == 0` 的帧，不反映轨迹中间的视觉分布。适合评估"数据集对目标场景的初始状态覆盖程度"。

5. **多曲线比较须用同一 Policy**：`MultiScalingCurveGenerator` 支持多条曲线对比，但如果不同曲线使用不同 policy，embedding 空间不同，分数不可直接横向比较。建议多条曲线共用同一 policy，只变化训练数据集。

6. **支持的 Policy 类型**：仅支持 ACT、DiffusionPolicy、Pi0、Pi0Fast、TDMPC、VQBeT（`lerobot==0.4.0`）。其他 policy 需自行注册到 `_POLICY_REGISTRY`。

7. **`hook_module` 必须指向 vision encoder**：填写错误的模块路径不会报错，但会提取语义错误的 embedding，导致分数无意义。推荐参考 README 中的 hook_module 对照表。
