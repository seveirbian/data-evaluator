# first_obs_extractor 模块 — 设计文档

日期:2026-06-11

## 目标

新增模块 `src/first_obs_extractor`:输入一个 LeRobot 数据集(v2.0/v3.0)和一个
`{"episode_ids": [...]}` 的 JSON,输出**一张大图**(PNG)——把选定 episode 的**第一帧
观测**(每个相机一张图)拼成一个近方形网格,每个 cell 标注 `ep{id} / {相机名}`。

## 背景与复用决策

`src/scaling_curve/_dataset.py` 里的 `LeRobotDatasetLoader.get_initial_observations()`
已经能取每个 episode 的 `frame_index == 0` 观测(相机 → CHW `[0,1]` 张量),并正确处理
v2.0 / v3.0 的共享视频帧定位。本模块复用它,不碰 lerobot 写入器。

**复用方式(已选方案 1):把 `LeRobotDatasetLoader` 提升为共享模块。**
将其从 `src/scaling_curve/_dataset.py` 移动到新的 **`src/dataset_io.py`**,让
`scaling_curve` 与 `first_obs_extractor` 都从公开的共享模块 import,消除"伸手进另一个
feature 包私有模块"的耦合。它本就是通用的数据集 IO 工具,不专属于 scaling curve。

涉及改动(经 grep 确认,真实引用仅两处):
- 移动文件:`src/scaling_curve/_dataset.py` → `src/dataset_io.py`(内容不变,仅 `LeRobotDatasetLoader` 一个类)。
- `src/scaling_curve/scaling_curve.py:9`:`from ._dataset import LeRobotDatasetLoader` → `from src.dataset_io import LeRobotDatasetLoader`。
- `main.py:10`:`from src.scaling_curve._dataset import LeRobotDatasetLoader` → `from src.dataset_io import LeRobotDatasetLoader`。
- 测试 `test_scaling_curve.py:28` 的 `@patch("src.scaling_curve.scaling_curve.LeRobotDatasetLoader")` **不受影响**:scaling_curve.py 仍把该名字 import 进自身命名空间,patch 目标依旧有效。

## 文件结构

```
src/dataset_io.py                 # 移动来的 LeRobotDatasetLoader(共享)
src/first_obs_extractor/
├── __init__.py                   # 导出 extract_first_obs
├── _extractor.py                 # 核心:校验 + 收集 cell + 渲染网格
├── __main__.py                   # CLI
└── tests/
    ├── __init__.py
    └── test_extractor.py
```

## 核心函数

```python
def extract_first_obs(
    src_dir: str | Path,
    episode_ids: list[int],
    out_path: str | Path,     # 输出 PNG 路径
) -> Path:                    # 返回 out_path
```

### 流程

1. **校验**:读 `meta/info.json` 取 `total_episodes`;要求 `episode_ids` 非空、无重复、
   全部在 `[0, total_episodes)` 内,否则 `raise ValueError`(缺 `info.json` →
   `FileNotFoundError`)。
2. `loader = LeRobotDatasetLoader(src_dir)`;`obs = loader.get_initial_observations()`
   (按 episode 顺序返回**全部** episode 的首帧观测)。
3. **收集 cell**:按 `episode_ids` 的**给定顺序**遍历;每个 episode 取 `obs[ep_id]`,
   对 `sorted(loader.camera_keys)` 的每个相机生成一个 cell
   `(label=f"ep{ep_id} / {相机短名}", image=obs[ep_id][cam])`。相机短名取 key 末段
   (如 `observation.images.front` → `front`)。
4. **渲染网格**:`cols = ceil(sqrt(N))`,`rows = ceil(N / cols)`(N = cell 总数);
   每个 cell `imshow`(CHW → HWC),标题 = label,关坐标轴;多出的格子留白。用 matplotlib
   (Agg,不弹窗)。
5. `savefig(out_path)`,自动创建父目录;返回 `Path(out_path)`。

### 内部分解(便于独立测试)

- `_grid_dims(n) -> tuple[int, int]`:返回 `(rows, cols)` 近方形布局,纯函数。
- `_render_montage(cells: list[tuple[str, ndarray | Tensor]], out_path) -> Path`:
  纯渲染,不依赖数据集;图像若为 CHW(`shape[0] == 3`)则转 HWC 再 `imshow`。
- `_validate_episode_ids(src_dir, episode_ids) -> int`:返回 `total_episodes`。

## 错误处理

| 情况 | 异常 |
|------|------|
| 缺 `meta/info.json` | `FileNotFoundError` |
| `episode_ids` 为空 | `ValueError` |
| `episode_ids` 含重复 | `ValueError` |
| 含越界 id(不在 `[0, total_episodes)`) | `ValueError` |

`out_path` 已存在则**覆盖**(与 `scaling_curve` 出图一致);父目录自动创建。

## CLI

```
python -m src.first_obs_extractor --src <数据集目录> --episodes <ids.json> --out <out.png>
```

读取 `<ids.json>` 的 `episode_ids` 后调用 `extract_first_obs`,完成后打印输出路径。

## 测试(TDD)

- `_grid_dims`:`1→(1,1)`、`4→(2,2)`、`5→(2,3)`、`6→(2,3)`、`7→(3,3)`、`9→(3,3)`。
- `_render_montage`:给若干 dummy numpy 小图(HWC 与 CHW 各一)→ 断言输出 PNG 存在且文件非空。
- `extract_first_obs`:用小合成 v3.0 数据集(沿用 `episodes_extractor` 测试的造数据方式,
  小图、少量帧)抽 2 个 episode → 断言 PNG 生成且非空。
- 校验:越界 id、重复 id、空列表分别 `raise ValueError`。

## 不做(YAGNI)

- 不做按相机筛选(默认画全部相机)。
- 不做单帧以外的轨迹采样。
- 不返回图像数组(只存文件并返回路径)。
- `get_initial_observations()` 会解码**全数据集**每个 episode 的首帧(含未选中的),大数据集
  略有浪费——本轮接受;需要时再给共享 loader 加按需加载。
