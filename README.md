# data-evaluator

无需在真机上运行 policy，通过 **Policy Embedding Similarity** 评估训练数据集对目标环境的覆盖程度，从而指导数据采集策略。

## 背景

本项目实现了论文中的 Factored Scaling Curve 方法，使用 vision encoder 的 embedding 相似度作为真实成功率的代理指标：

**Eq. 9** — 两个观测之间的余弦相似度：

$$c_\pi(x_i, x_j) = \frac{\phi_\pi(x_i) \cdot \phi_\pi(x_j)}{||\phi_\pi(x_i)|| \cdot ||\phi_\pi(x_j)||}$$

**Eq. 10** — eval 样本与训练集的相似度（取最大值）：

$$c_\pi(x_i, D_{\text{train}}) = \max_{x_j \in D_{\text{train}}} c_\pi(x_i, x_j)$$

**Eq. 11** — 对所有 eval 样本取均值，得到最终指标：

$$\bar{c}_\pi = \sum_{x_i \in D_{\text{eval}}} \frac{c(x_i, D_{\text{train}})}{|D_{\text{eval}}|}$$

$\bar{c}_\pi$ 越高，说明训练数据对 eval 环境的覆盖越好，policy 在目标环境中的成功率也越高。

---

## 使用前提与限制

使用本工具前，请确认以下条件均满足：

| 限制 | 说明 |
|------|------|
| **LeRobot v3.0 格式** | 数据集必须是 LeRobot v3.0 格式（Parquet + MP4 + `meta/info.json`）。 |
| **Policy 与数据同构** | Policy 的 `input_shapes`（关节数、状态维度）必须与数据集匹配。用 6-DOF 机器人训练的 policy 无法处理 14-DOF 机器人的数据，forward pass 会报 shape 错误。 |
| **相机视角有交集** | Train 和 eval 数据集至少共享一个相机视角（物理视角相同）。不同命名可通过 `camera_key_map` 处理，但不能替代真实的视角对应。 |
| **只使用初始帧** | 每个 episode 只取 `frame_index == 0` 的帧。指标反映的是初始状态的视觉覆盖，不涵盖轨迹中间帧的分布。 |
| **多曲线须用同一 Policy** | `MultiScalingCurveGenerator` 支持多条曲线对比，但不同曲线若使用不同 policy，embedding 空间不同，分数不可直接横向比较。 |
| **支持的 Policy 类型** | 仅支持 ACT、DiffusionPolicy、Pi0、Pi0Fast、TDMPC、VQBeT（`lerobot==0.4.0`）。 |
| **hook_module 必须正确** | 填写错误的 `hook_module` 路径不会报错，但会提取语义无意义的 embedding。请参考下方对照表。 |

---

## 模块结构

```
src/scaling_curve_evaluator/
├── __init__.py       # 公开 API：ScalingCurveGenerator, MultiScalingCurveGenerator
├── scaling_curve.py  # 公开类实现
├── _dataset.py       # 内部：LeRobot v3.0 数据加载
├── _embeddings.py    # 内部：Policy embedding 提取（forward hook）
└── _similarity.py    # 内部：Eq. 9-11 余弦相似度计算
```

---

## 快速开始

### 环境准备

```bash
uv sync
```

### 目录结构

```
project/
├── policy/
│   └── my_policy/          # LeRobot policy 本地目录（含 config.json + 权重）
├── data/
│   ├── train/
│   │   └── dataset_name/   # LeRobot v3.0 训练集
│   └── eval/
│       └── dataset_name/   # LeRobot v3.0 eval 集
└── main.py
```

### 单条 Scaling Curve

```python
from src.scaling_curve_evaluator import ScalingCurveGenerator

gen = ScalingCurveGenerator(
    policy_dir="policy/my_policy",
    train_data_dir="data/train/dataset_name",
    eval_data_dir="data/eval/dataset_name",
    hook_module="model.backbone",  # 根据 policy 类型选择，见下表
    device="auto",                 # 自动选择 cuda / mps / cpu
    num_points=20,                 # scaling curve 采样点数
)
gen.generate()
gen.plot(save_path="curve.png", show=True)
```

### 多数据集对比（同一 Policy）

```python
from src.scaling_curve_evaluator import MultiScalingCurveGenerator

gen = MultiScalingCurveGenerator(
    eval_data_dir="data/eval/dataset_name",
    curves=[
        {
            "policy_dir": "policy/my_policy",
            "train_data_dir": "data/train/batch1",
            "hook_module": "model.backbone",
        },
        {
            "policy_dir": "policy/my_policy",
            "train_data_dir": "data/train/batch2",
            "hook_module": "model.backbone",
        },
    ],
    device="auto",
    num_points=20,
)
gen.generate_all()
gen.plot(save_path="multi_curve.png", show=True)
```

### 相机键映射（train/eval 命名不一致时）

```python
curves=[
    {
        "policy_dir": "policy/my_policy",
        "train_data_dir": "data/train/batch1",
        "hook_module": "model.backbone",
        "camera_key_map": {
            "observation.images.right_wrist": "observation.images.front",
        },
    },
]
```

---

## 确定 hook_module

| Policy | `hook_module` |
|--------|--------------|
| ACT | `model.backbone` |
| DiffusionPolicy | `model.obs_encoder` |
| Pi0 | `model.paligemma_with_expert.paligemma.vision_tower` |
| Pi0Fast | `model.paligemma_with_expert.paligemma.vision_tower` |
| TDMPC | `model.encoder` |
| VQBeT | `model.obs_encoder` |

如果不确定，运行以下命令查看 policy 结构：

```bash
uv run python -c "
from src.scaling_curve_evaluator._embeddings import _POLICY_REGISTRY
import importlib, json
config = json.load(open('policy/my_policy/config.json'))
policy_type = config.get('type','').lower().replace('config','').strip('_-')
module_path, class_name = _POLICY_REGISTRY[policy_type]
cls = getattr(importlib.import_module(module_path), class_name)
p = cls.from_pretrained('policy/my_policy')
print(p)
"
```

选择 vision encoder 对应的模块路径（一般是处理图像输入、输出特征图的那层）。

---

## 依赖

- `lerobot[all]==0.4.0`
- `torch`
- `av`（视频解码）
- `pandas`（parquet 读取）
- `tqdm`
- `matplotlib`
