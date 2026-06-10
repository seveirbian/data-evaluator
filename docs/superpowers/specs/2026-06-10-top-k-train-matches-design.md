# 测试集视角的 Top-K 训练集匹配 — 设计文档

日期:2026-06-10

## 目标

新增功能:从测试集(eval)的视角,为**每个 eval episode** 记录与之相似度最高的 5 个训练集(train)episode。

## 背景与核心洞察

现有 `compute_sim_matrix()`(`src/scaling_curve/_similarity.py`)已经产出一个
`[N_eval, N_train]` 的余弦相似度矩阵,且已对多相机做了 max 融合。

"每个 eval episode 最相似的 5 个 train episode" 本质上就是对该矩阵的**每一行取 top-k**。
因此本功能无需重新计算 embedding 或相似度,只是在已有矩阵上做一次 `topk`。

Episode 身份用其在 `LeRobotDatasetLoader.get_initial_observations()` 返回的有序列表中的
**0-based 索引**表示。数据集本身没有显式 episode_id,这与现有 per-eval 图中的 `eval_id`
口径一致(`main.py` 的 `test_openpi_jax`,`eval_ids = list(range(len(eval_scores)))`)。

## 设计

### 1. 纯函数(`src/scaling_curve/_similarity.py`)

```python
def top_k_train_matches(
    sim_matrix: torch.Tensor,   # [N_eval, N_train]
    c_min: float,
    c_max: float,
    k: int = 5,
) -> list[dict]:
    """对每个 eval episode 返回归一化分数最高的前 k 个 train episode。"""
```

行为:

- 用 `torch.topk(sim_matrix, k, dim=1)` 取每行 top-k。`k` 自动 clamp 到 `N_train`
  (`k = min(k, N_train)`),避免越界。
- 分数采用与 scaling curve **一致**的归一化:`(raw - c_min) / (c_max - c_min)`,
  落在 `[0, 1]`。当 `denom = c_max - c_min < 1e-8` 时回退为 `1.0`(与
  `policy_embedding_similarity` / `per_sample_scores` 的退化分支一致)。
- topk 的排序在**原始 cosine** 上进行;归一化是单调变换,不改变排序。
- 返回 **JSON 可序列化** 的纯数据结构,文件 I/O 交给调用方,保持本模块为纯数学。

返回结构:

```json
[
  {"eval_id": 0, "top_k": [{"train_id": 12, "score": 0.987}, {"train_id": 5, "score": 0.954}]},
  {"eval_id": 1, "top_k": [{"train_id": 3, "score": 0.991}]}
]
```

- 外层 list 按 `eval_id` 升序(即矩阵行顺序)。
- 每个 `top_k` 列表按 `score` 降序,长度为 `min(k, N_train)`。

### 2. 接线(`main.py` 的 `test_openpi_jax`)

在已经算出 `sim_matrix, c_min, c_max` 之后(Step 4 区域,per-eval 图附近)调用该函数,
用 `json.dump` 写到项目根目录 `openpi_jax_top5_matches.json`,并打印保存路径——
与现有图片输出(`openpi_jax_per_eval.png` 等)风格一致。

### 3. 测试(TDD,先写测试)

在 `src/scaling_curve/tests/test_scaling_curve.py` 增加单测,用构造好的小
`sim_matrix` 验证:

- 选出的 `train_id` 正确,且每行 `top_k` 按分数**降序**排列。
- `k > N_train` 时正确 clamp,返回长度为 `N_train`。
- 归一化分数计算正确(给定 `c_min/c_max` 手算对比)。
- 退化矩阵(全相等,`denom < 1e-8`)走 `1.0` 分支。

## 不做(YAGNI)

- 不画图、不写 CSV(仅 JSON)。
- 不改 `ScalingCurveGenerator` / `MultiScalingCurveGenerator` 类(本轮 OOP 路径未使用)。
- 不引入显式 episode_id 体系(沿用索引)。
