# episodes_extractor 模块 — 设计文档

日期:2026-06-10

## 目标

新增模块 `src/episodes_extractor`:接收一个 LeRobot v3.0 格式的数据集和一个
`{"episode_ids": [...]}` 的 JSON 文件,输出一个**只包含选定 episode** 的、合法的
LeRobot v3.0 数据集,并记录原始 episode id → 新 episode id 的映射。

## 背景

LeRobot v3.0 是**打包格式**:多个 episode 共享同一个 `data/*.parquet`,也共享同一个
`videos/<cam>/*.mp4`(每个 episode 在视频里对应一段 `from_timestamp`/`to_timestamp`)。
因此"抽取子集"不是文件拷贝,而是要重写 data parquet、逐 episode 的 meta+stats、视频片段、
tasks、splits、`info.json`,并把所有索引重新连续编号。

环境中已安装 `lerobot 0.4.0`,它提供了规范的写入器
(`LeRobotDataset.create` / `add_frame` / `save_episode` / `finalize`)。本设计**委托
lerobot 写入器**完成所有重编号/视频重编码/meta/stats/info.json 的生成,而非手工改文件。

## 已验证的 lerobot 0.4.0 API 行为(源码确认)

- `LeRobotDataset.__init__(..., episodes: list[int] | None)`:可只加载指定 episode。
- `validate_frame(frame, features)`:`frame` 须含 `task`,以及 `features - DEFAULT_FEATURES`
  的全部键。`DEFAULT_FEATURES = ["timestamp","frame_index","episode_index","index","task_index"]`
  由写入器自动管理,**不要**放进 `frame`。
- `add_frame(frame)`:自动追加 `frame_index`/`timestamp`,从 `frame` 弹出 `task`;其余特征
  逐键写入。
- 图像:`validate_feature_image_or_video` 同时接受 `(C,H,W)` 与 `(H,W,C)`;写盘的
  `image_array_to_pil_image` 在 `shape[0]==3` 时**自动**转置 `(C,H,W)→(H,W,C)`,并把
  float `[0,1]` 转 `uint8`。**因此源 `__getitem__` 返回的 `[C,H,W]` float 张量可直接传入
  `add_frame`,无需手动转置**(本数据集图像为 1080×1920×3,无 H/W 等于 3 的歧义)。

## 文件结构

```
src/episodes_extractor/
├── __init__.py          # 导出 extract_episodes
├── _extractor.py        # 核心逻辑
├── __main__.py          # CLI 入口
└── tests/
    ├── __init__.py
    └── test_extractor.py
```

## 核心函数

```python
def extract_episodes(
    src_dir: str | Path,
    episode_ids: list[int],   # 来自 JSON 的 "episode_ids",按此顺序决定新 id
    out_dir: str | Path,
    repo_id: str = "extracted",
) -> dict[int, int]:          # 返回 {原始id: 新id}
```

### 流程

1. **校验**:读源 `meta/info.json` 取 `total_episodes`;要求 `episode_ids` 非空、无重复、
   全部落在 `[0, total_episodes)`,否则 `raise ValueError`。`out_dir` 已存在且非空时
   `raise ValueError`(避免覆盖)。
2. **加载源**:`src = LeRobotDataset(repo_id="source", root=src_dir, episodes=sorted(episode_ids))`,
   只加载选定 episodes。
3. **创建输出**:
   `out = LeRobotDataset.create(repo_id, fps=src.fps, features=src.features, root=out_dir,
   robot_type=src.meta.robot_type, use_videos=True)`。
4. **逐 episode 写入**(按用户给定的 `episode_ids` 顺序,新 id = 列表下标):
   取该原始 episode 的所有帧(按 `frame_index` 升序);对每帧构造
   `frame = {k: src_item[k] for k in src.features if k not in DEFAULT_FEATURES}`,
   再设 `frame["task"] = src_item["task"]`,图像原样传入;`out.add_frame(frame)`。
   该 episode 全部帧写完后 `out.save_episode()`。
5. `out.finalize()`。
6. **写映射**:把 `{原始id: 新id}` 以 JSON 写到 `out_dir/extraction_mapping.json`,
   函数同时返回该 dict。

### 帧的获取方式

用源数据集的 `__getitem__`(已正确解码视频帧)。由于第 2 步用 `episodes=` 只加载了选定
episodes,源数据集的全局帧索引覆盖这些 episode 的所有帧;通过每帧的
`episode_index`(原始 id)分组,并按用户 `episode_ids` 顺序逐个 episode 写出。

## 错误处理

- 源缺 `meta/info.json` → `FileNotFoundError`/`ValueError`。
- `episode_ids` 为空、含重复、含越界 id → `ValueError`,信息中列出问题 id。
- `out_dir` 已存在且非空 → `ValueError`。

## CLI

```
python -m src.episodes_extractor --src <dataset_dir> --episodes <ids.json> --out <out_dir> [--repo-id NAME]
```

读取 `<ids.json>` 的 `episode_ids` 字段后调用 `extract_episodes`,完成后打印输出路径与映射。

## 测试(TDD)

- **Round-trip(主测)**:用 `LeRobotDataset.create` 造一个 3-episode 的小合成数据集
  (小图、少量帧,`use_videos` 可关或开),抽取其中 2 个 episode,重新加载输出集,断言:
  - `num_episodes == 2`;
  - 总帧数 = 两个被选 episode 的长度之和;
  - 返回的映射 = `{原始id: 0/1}` 且与 `episode_ids` 顺序一致;
  - 抽取后某帧的非图像特征(如 `observation.state`/`action`)与源一致;
  - `out_dir/extraction_mapping.json` 存在且内容等于返回的映射。
- **校验测试**:越界 id、重复 id、空列表分别 `raise ValueError`;`out_dir` 非空 `raise ValueError`。

## 不做(YAGNI)

- 不做 `push_to_hub`。
- 不支持按 task 名筛选(只按 episode id)。
- 不把原始 id 保留为输出 `episode_index`(lerobot 写入器强制 0..K-1;映射文件已满足追溯需求)。
- 不手工拼接 parquet/视频(全交给 lerobot 写入器)。
