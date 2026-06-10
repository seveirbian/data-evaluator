from __future__ import annotations

import json
import shutil
from collections import defaultdict
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


# Auto-managed by lerobot's writer; must NOT be placed into a frame dict.
_DEFAULT_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _tuple_shapes(features: dict) -> dict:
    """Return a copy of features with every 'shape' coerced to a tuple.

    lerobot's frame validator compares actual numpy shape (tuple) against the
    feature's stored shape with `!=`; a list shape never equals a tuple, so
    shapes loaded from info.json (lists) must be tuples for add_frame to pass.
    """
    out = {}
    for key, spec in features.items():
        spec = dict(spec)
        if "shape" in spec and spec["shape"] is not None:
            spec["shape"] = tuple(spec["shape"])
        out[key] = spec
    return out


def extract_episodes(
    src_dir: str | Path,
    episode_ids: list[int],
    out_dir: str | Path,
    repo_id: str = "extracted",
) -> dict[int, int]:
    """Extract selected episodes into a new LeRobot v3.0 dataset.

    Args:
        src_dir: source v3.0 dataset root.
        episode_ids: original episode ids to keep; list order defines new ids
            (new episode_index = position in this list).
        out_dir: output dataset root (must be empty or non-existent).
        repo_id: repo id for the output dataset.

    Returns:
        {original_episode_id: new_episode_id}.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    _validate_inputs(src_dir, episode_ids, out_dir)

    src = LeRobotDataset("source", root=src_dir, episodes=sorted(episode_ids))

    try:
        out = LeRobotDataset.create(
            repo_id,
            fps=src.fps,
            features=_tuple_shapes(src.features),
            root=out_dir,
            robot_type=src.meta.robot_type,
            use_videos=True,
        )

        feature_keys = [k for k in src.features if k not in _DEFAULT_FEATURES]

        # Group source global frame indices by original episode id.
        # Reading the episode_index column does not decode any video.
        groups: dict[int, list[int]] = defaultdict(list)
        for global_i, eid in enumerate(src.hf_dataset["episode_index"]):
            groups[int(eid)].append(global_i)

        mapping: dict[int, int] = {}
        for new_id, orig_id in enumerate(episode_ids):
            for global_i in groups[int(orig_id)]:
                item = src[global_i]
                # "task" is not a stored feature; it is derived per-frame from
                # task_index by __getitem__, so it is added separately here.
                frame = {k: item[k] for k in feature_keys}
                frame["task"] = item["task"]
                out.add_frame(frame)
            out.save_episode()
            mapping[int(orig_id)] = new_id

        out.finalize()

        (out_dir / "extraction_mapping.json").write_text(
            json.dumps({str(k): v for k, v in mapping.items()}, indent=2)
        )
    except BaseException:
        # _validate_inputs guaranteed out_dir was empty/absent beforehand, so
        # removing what we created here is safe; makes the call re-entrant.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    return mapping
