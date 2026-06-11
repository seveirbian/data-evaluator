# first_obs_extractor

输入一个 **LeRobot**(v2.0/v3.0)数据集 + 一个 `{"episode_ids": [...]}` 的 JSON,输出**一张大图**(PNG):把选定 episode 的**第一帧观测**(每个相机一张)拼成一个近方形网格,每个 cell 标注 `ep{id} / {相机名}`。

常见用途:快速目检某批 episode 的初始状态是否符合预期(比如 [`episodes_extractor`](../episodes_extractor) 抽出来的子集,或 [`scaling_curve`](../scaling_curve) 里相似度高/低的那些 episode),一张图看全。

---

## 工作原理

复用共享的 `src/dataset_io.py` 里的 `LeRobotDatasetLoader.get_initial_observations()` —— 它已经能取每个 episode 的 `frame_index == 0` 观测(相机 → `[0,1]` 图像张量),并正确处理 v2.0/v3.0 的共享视频帧定位。本模块只做三件事:**加载首帧 → 收集 (episode, 相机) cell → matplotlib 渲染近方形网格并保存**。不涉及视频重编码,很快。

网格布局:设 cell 总数为 N(= 选中 episode 数 × 相机数),则 `cols = ceil(sqrt(N))`、`rows = ceil(N / cols)`,多出来的格子留白。

---

## 命令行用法

```bash
python -m src.first_obs_extractor \
    --src <数据集目录> \
    --episodes <episode_ids.json> \
    --out  <输出图片.png>
```

`episode_ids.json` 的格式:

```json
{ "episode_ids": [5, 0, 12] }
```

示例:

```bash
python -m src.first_obs_extractor \
    --src example/cuicuisha/data/train/Task-GrabCuicuishaPlaceMatting-30 \
    --episodes ids.json \
    --out first_obs.png
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--src` | 是 | LeRobot v2.0/v3.0 数据集根目录(含 `meta/info.json`)。 |
| `--episodes` | 是 | JSON 文件,形如 `{"episode_ids": [...]}`。 |
| `--out` | 是 | 输出 PNG 路径(已存在则覆盖,父目录自动创建)。 |

---

## Python API

```python
from src.first_obs_extractor import extract_first_obs

out = extract_first_obs(
    src_dir="path/to/dataset",
    episode_ids=[5, 0, 12],
    out_path="first_obs.png",
)
# out == Path("first_obs.png")
```

**cell 排列顺序**:按 `episode_ids` 的**给定顺序**逐个 episode,episode 内按相机 key 排序(`sorted(camera_keys)`)。单相机时就是 episodes 的方形拼图。

---

## 输出

一张 PNG,网格中每个 cell = 一个 (episode, 相机) 的首帧,标题为 `ep{原始id} / {相机短名}`(相机短名取 key 末段,如 `observation.images.front` → `front`)。多相机数据集会把每个相机各占一个 cell。

---

## 校验与错误处理

| 情况 | 异常 |
|------|------|
| 缺 `meta/info.json` | `FileNotFoundError` |
| `episode_ids` 为空 | `ValueError` |
| `episode_ids` 含重复 | `ValueError` |
| 含越界 id(不在 `[0, total_episodes)`) | `ValueError` |

`out_path` 已存在则覆盖;父目录自动创建。

---

## 限制

- 仅取每个 episode 的**第一帧**(`frame_index == 0`),不做轨迹中间帧采样。
- 默认画**全部相机**,不支持按相机筛选。
- `get_initial_observations()` 会解码**全数据集**每个 episode 的首帧(含未选中的),大数据集略有浪费——需要时可给共享 loader 加按需加载。
