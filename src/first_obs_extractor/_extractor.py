from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _grid_dims(n: int) -> tuple[int, int]:
    """Near-square grid: returns (rows, cols) for n cells."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _validate_episode_ids(src_dir: str | Path, episode_ids: list[int]) -> int:
    """Validate episode ids against the source dataset. Returns total_episodes."""
    src_dir = Path(src_dir)
    info_path = src_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Source dataset has no meta/info.json: {info_path}")
    total_episodes = int(json.loads(info_path.read_text())["total_episodes"])

    if not episode_ids:
        raise ValueError("episode_ids is empty.")

    duplicates = sorted({e for e in episode_ids if episode_ids.count(e) > 1})
    if duplicates:
        raise ValueError(f"episode_ids contains duplicate ids: {duplicates}")

    out_of_range = sorted(e for e in episode_ids if e < 0 or e >= total_episodes)
    if out_of_range:
        raise ValueError(
            f"episode_ids out of range [0, {total_episodes}): {out_of_range}"
        )

    return total_episodes


def extract_first_obs(
    src_dir: str | Path,
    episode_ids: list[int],
    out_path: str | Path,
) -> Path:
    """Tile the first-frame observations of selected episodes into one PNG.

    Args:
        src_dir: LeRobot v2.0/v3.0 dataset root.
        episode_ids: episode ids to include; output cells follow this order,
            then cameras in sorted key order.
        out_path: output PNG path (overwritten if it exists).

    Returns:
        Path(out_path).
    """
    from src.dataset_io import LeRobotDatasetLoader

    src_dir = Path(src_dir)
    _validate_episode_ids(src_dir, episode_ids)

    loader = LeRobotDatasetLoader(src_dir)
    observations = loader.get_initial_observations()
    cameras = sorted(loader.camera_keys)

    cells: list[tuple[str, object]] = []
    for ep_id in episode_ids:
        ep_obs = observations[ep_id]
        for cam in cameras:
            short = cam.split(".")[-1]
            cells.append((f"ep{ep_id} / {short}", ep_obs[cam]))

    return _render_montage(cells, out_path)


def _to_hwc_image(value) -> np.ndarray:
    """Convert a torch tensor / numpy array image to a HWC numpy array.

    Decoded video frames are channel-first [C,H,W]; matplotlib's imshow wants
    [H,W,C]. Float images are left in [0,1] (clipped) for imshow.
    """
    arr = value
    if hasattr(arr, "detach"):  # torch tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def _render_montage(cells: list[tuple[str, object]], out_path: str | Path) -> Path:
    """Render labelled first-frame cells into a near-square montage PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cells)
    rows, cols = _grid_dims(n)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), squeeze=False
    )
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx < n:
            label, image = cells[idx]
            ax.imshow(_to_hwc_image(image))
            ax.set_title(label, fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
