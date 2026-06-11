from __future__ import annotations

import json
import math
from pathlib import Path


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


def extract_first_obs(src_dir, episode_ids, out_path):
    raise NotImplementedError("Implemented in a later task.")
