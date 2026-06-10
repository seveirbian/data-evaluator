from __future__ import annotations

import json
from pathlib import Path


def _validate_inputs(
    src_dir: str | Path,
    episode_ids: list[int],
    out_dir: str | Path,
) -> int:
    """Validate extraction inputs. Returns source total_episodes.

    Raises ValueError / FileNotFoundError on invalid inputs.
    """
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

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"out_dir already exists and is not empty: {out_dir}")

    return total_episodes


def extract_episodes(src_dir, episode_ids, out_dir, repo_id="extracted"):
    raise NotImplementedError("Implemented in a later task.")
