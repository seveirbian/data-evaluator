# data-evaluator

无需在真机上运行 policy，通过 **Policy Embedding Similarity** 评估训练数据集对目标环境的覆盖程度，从而指导数据采集策略。

## 背景

本项目实现了论文中的 Factored Scaling Curve 方法，使用 vision encoder 的 embedding 相似度作为真实成功率的代理指标：

**Eq. 9** — 两个观测之间的余弦相似度：

$$c_\pi(x_i, x_j) = \frac{\phi_\pi(x_i) \cdot \phi_\pi(x_j)}{||\phi_\pi(x_i)|| \cdot ||\phi_\pi(x_j)||}$$

**Eq. 10** — eval 样本与训练集的相似度（取最大值）：

$$c_\pi(x_i, D_{\text{train}}) = \max_{x_j \in D_{\text{train}}} c_\pi(x_i, x_j)$$

**Eq. 11** — 归一化后对所有 eval 样本取均值，得到最终指标：

$$\bar{c}_\pi = \sum_{x_i \in D_{\text{eval}}} \frac{c(x_i, D_{\text{train}})}{|D_{\text{eval}}|}$$

$\bar{c}_\pi$ 越高，说明训练数据对 eval 环境的覆盖越好，policy 在目标环境中的成功率也越高。

---

## 模块结构

```
src/scaling_curve_evaluator/
├── __init__.py
├── dataset.py      # LeRobot v3.0 数据集加载
├── embeddings.py   # Policy embedding 提取（forward hook）
├── similarity.py   # Eq. 9-11 计算
└── evaluator.py    # 主入口，串联全流程
```

### `dataset.py` — LeRobotDatasetLoader

从 LeRobot v3.0 格式数据集中提取每个 episode 的**初始观测**（`frame_index == 0`）。

论文说明只需初始帧，不需要完整轨迹，因此不必在真机上 roll out。

**LeRobot v3.0 数据格式：**
```
{dataset}/
  meta/info.json                              # 数据集元信息（fps、features 等）
  data/chunk-000/file-000.parquet             # state/action 数据，行号 = 帧号
  videos/{cam_key}/chunk-000/file-000.mp4    # 对应视频，帧顺序与 parquet 一致
```

图像通过 `PyAV` seek 到目标时间戳后读取第一帧，避免逐帧遍历。

### `embeddings.py` — PolicyEmbeddingExtractor

通过 **PyTorch forward hook** 拦截 policy 的 vision encoder 输出，提取每个观测的 embedding。

支持的 LeRobot policy（`lerobot==0.4.0`）：

| Policy | `hook_module` |
|--------|--------------|
| ACT | `model.backbone` |
| DiffusionPolicy | `model.obs_encoder` |
| Pi0 | `model.paligemma_with_expert.paligemma.vision_tower` |
| Pi0Fast | `model.paligemma_with_expert.paligemma.vision_tower` |
| TDMPC | `model.encoder` |
| VQBeT | `model.obs_encoder` |

**多相机 embedding 解析逻辑：**
- Hook 触发 N 次（每个相机一次）→ 按顺序映射到 camera_key
- Hook 触发 1 次，输出 shape[0] == N_cams → 拆分
- Hook 触发 1 次，单一 embedding → 复制给所有相机（产生相同分数）

**支持的 hook 输出类型：**
- `torch.Tensor` → 直接使用
- `dict`（如 `IntermediateLayerGetter` 的 `OrderedDict`）→ 取最后一个值
- `tuple/list` → 取第一个元素

任意维度的 tensor 通过 global average pooling 压缩为 `[N, D]`。

### `similarity.py` — 相似度计算

实现论文 Eq. 9-11，多相机扩展策略：

```
对每个 eval 样本 x_i：
  for 每个相机 k：
    score_k = max_{x_j ∈ D_train} cosine_sim(φ^k(x_i), φ^k(x_j))   # Eq. 10
  score_i = max_k score_k    # 任意相机被覆盖即算匹配

归一化 score_i 到 [0, 1]
c̄_π = mean(score_i)         # Eq. 11
```

### `evaluator.py` — PolicyEmbeddingSimilarityEvaluator

串联以上三个模块的主类，只暴露一个 `evaluate()` 方法。

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

### 运行

```python
from src.scaling_curve_evaluator import PolicyEmbeddingSimilarityEvaluator

evaluator = PolicyEmbeddingSimilarityEvaluator(
    policy_dir="policy/my_policy",
    train_data_dir="data/train/dataset_name",
    eval_data_dir="data/eval/dataset_name",
    hook_module="model.backbone",  # 根据 policy 类型选择，见上表
    device="auto",                 # 自动选择 cuda / mps / cpu
)
score = evaluator.evaluate()
print(f"c̄_π = {score:.4f}")
```

```bash
uv run main.py
```

---

## 确定 hook_module

如果不确定用哪个模块，运行以下命令查看 policy 结构：

```bash
uv run python -c "
from src.scaling_curve_evaluator.embeddings import _POLICY_REGISTRY
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
