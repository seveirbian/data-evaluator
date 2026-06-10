# episodes_extractor

从一个 **LeRobot v3.0** 数据集中,按 episode id 列表抽取子集,输出一个**只包含选定 episode** 的、合法的 v3.0 数据集,并记录原始 id → 新 id 的映射。

常见用途:把 [`scaling_curve`](../scaling_curve) 算出的"与某些 eval episode 最相似的 train episode"挑出来,单独组成一个小数据集做可视化、复跑或二次采集分析。

---

## 工作原理

LeRobot v3.0 是**打包格式**:多个 episode 共享同一个 `data/*.parquet`,也共享同一个 `videos/<cam>/*.mp4`(每个 episode 在视频里对应一段时间戳)。因此"抽子集"不是文件拷贝,而要重写 data parquet、逐 episode 的 meta+stats、视频片段、tasks、`info.json`,并把所有索引重新连续编号。

本模块**委托 lerobot 0.4.0 的写入器**(`LeRobotDataset.create / add_frame / save_episode / finalize`)完成上述全部工作:只加载选定 episode → 逐帧重新写入 → lerobot 自动重编号、重编码视频、重算 stats、生成 `info.json`。输出保证是合法 v3.0 数据集。

---

## 命令行用法

```bash
python -m src.episodes_extractor \
    --src  <源数据集目录> \
    --episodes <episode_ids.json> \
    --out  <输出目录> \
    [--repo-id NAME] \
    [--image-writer-threads N] \
    [--image-writer-processes M]
```

`episode_ids.json` 的格式:

```json
{ "episode_ids": [5, 0, 12] }
```

示例:

```bash
python -m src.episodes_extractor \
    --src example/cuicuisha/data/train/Task-GrabCuicuishaPlaceMatting-30 \
    --episodes ids.json \
    --out /tmp/subset \
    --image-writer-threads 8
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--src` | 必填 | 源 v3.0 数据集根目录(含 `meta/info.json`)。 |
| `--episodes` | 必填 | JSON 文件,形如 `{"episode_ids": [...]}`。 |
| `--out` | 必填 | 输出数据集目录,必须为空或不存在。 |
| `--repo-id` | `extracted` | 输出数据集的 repo id。 |
| `--image-writer-threads` | `4` | 后台写 PNG 帧的线程数。`0` 为同步写入,高分辨率下会显著拖慢;`>0` 并行化。 |
| `--image-writer-processes` | `0` | 图像写入进程数(在线程之外另开进程)。 |

---

## Python API

```python
from src.episodes_extractor import extract_episodes

mapping = extract_episodes(
    src_dir="path/to/v3_dataset",
    episode_ids=[5, 0, 12],     # 列表顺序决定新 id:5→0, 0→1, 12→2
    out_dir="path/to/output",
    repo_id="extracted",
    image_writer_threads=4,
)
# mapping == {5: 0, 0: 1, 12: 2}
```

**新 episode_index = 该原始 id 在 `episode_ids` 列表中的下标**(保留用户给定顺序,而非排序顺序)。

---

## 输出

输出目录是一个标准 v3.0 数据集,额外多一个映射文件:

```
out/
├── data/ …                       # 重新打包、重新连续编号的 parquet
├── videos/ …                     # 只含选定 episode 的视频
├── meta/info.json …              # total_episodes / total_frames 等已更新
└── extraction_mapping.json       # {"5": 0, "0": 1, "12": 2}  原始id→新id
```

`extract_episodes` 的返回值与 `extraction_mapping.json` 内容一致(返回值用 int 作 key,JSON 中 key 为字符串)。

---

## 校验与错误处理

以下情况在**写任何文件之前**就 `raise`:

| 情况 | 异常 |
|------|------|
| 源缺 `meta/info.json` | `FileNotFoundError` |
| `episode_ids` 为空 | `ValueError` |
| `episode_ids` 含重复 | `ValueError` |
| 含越界 id(不在 `[0, total_episodes)`) | `ValueError` |
| `--out` 已存在且非空 | `ValueError` |

抽取中途若失败,已写入的 `out_dir` 会被清理,保证调用可重入。

---

## 性能说明

该方案需要**解码源视频 + 重新编码 AV1**,这是固有成本,无法靠参数消除:

- 源视频解码约 **30 ms/帧**;
- 1080p AV1 重编码约 **2 分钟/episode**(595 帧),整套大数据集可能要一小时量级。

`--image-writer-threads` 把同步的 PNG 写盘并行化,是当前唯一干净有效的提速杠杆(`svtav1 preset` 几乎无效;lerobot 0.4.0 的 `batch_encoding_size>1` 有 bug,故未暴露)。若需要数量级的提速,只能改用"视频流复制不重编码"的方案(本模块未实现)。

---

## 限制

- 仅支持 **LeRobot v3.0** 格式。
- 不保留原始 id 作为输出 `episode_index`(lerobot 写入器强制 `0..K-1`,用 `extraction_mapping.json` 追溯)。
- 不支持按 task 名筛选,只按 episode id。
- 不做 `push_to_hub`。
