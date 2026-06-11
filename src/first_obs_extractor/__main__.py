from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._extractor import extract_first_obs


def _read_episode_ids(json_path: str | Path) -> list[int]:
    """Read {"episode_ids": [...]} from a JSON file."""
    data = json.loads(Path(json_path).read_text())
    if not isinstance(data, dict) or "episode_ids" not in data:
        raise ValueError(
            f"JSON must be an object with an 'episode_ids' key: {json_path}"
        )
    return [int(e) for e in data["episode_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.first_obs_extractor",
        description="Tile first-frame observations of selected episodes into one image.",
    )
    parser.add_argument("--src", required=True, help="LeRobot v2.0/v3.0 dataset dir")
    parser.add_argument(
        "--episodes", required=True, help='JSON file: {"episode_ids": [...]}'
    )
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args()

    episode_ids = _read_episode_ids(args.episodes)
    out = extract_first_obs(args.src, episode_ids, args.out)
    print(f"Saved first-observation montage to {out}")


if __name__ == "__main__":
    main()
