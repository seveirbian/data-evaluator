from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._extractor import extract_episodes


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
        prog="python -m src.episodes_extractor",
        description="Extract a subset of episodes from a LeRobot v3.0 dataset.",
    )
    parser.add_argument("--src", required=True, help="source v3.0 dataset dir")
    parser.add_argument(
        "--episodes", required=True, help='JSON file: {"episode_ids": [...]}'
    )
    parser.add_argument("--out", required=True, help="output dataset dir")
    parser.add_argument("--repo-id", default="extracted", help="output repo id")
    args = parser.parse_args()

    episode_ids = _read_episode_ids(args.episodes)
    mapping = extract_episodes(args.src, episode_ids, args.out, repo_id=args.repo_id)
    print(f"Extracted {len(mapping)} episodes to {args.out}")
    print(f"id mapping (original -> new): {mapping}")


if __name__ == "__main__":
    main()
