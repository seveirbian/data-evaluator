import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_io import LeRobotDatasetLoader


def _build_gray_dataset(root: Path, lengths: list[int]):
    """Build a v3.0 dataset, one camera; each frame is solid gray = global_index*20.

    The gray level lets a test identify exactly which video frame was returned.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["s"]},
        "observation.images.cam": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
    }
    ds = LeRobotDataset.create(
        "gray",
        fps=10,
        features=features,
        root=root,
        robot_type="t",
        use_videos=True,
        video_backend="pyav",
    )
    g = 0
    for _ep, length in enumerate(lengths):
        for _ in range(length):
            img = np.full((64, 64, 3), (g * 20) / 255.0, dtype=np.float32)
            ds.add_frame(
                {
                    "observation.state": np.array([g], dtype=np.float32),
                    "observation.images.cam": img,
                    "task": "t",
                }
            )
            g += 1
        ds.save_episode()
    ds.finalize()


def _first_frame_grays(root: Path) -> list[int]:
    obs = LeRobotDatasetLoader(root).get_initial_observations()
    return [round(float(o["observation.images.cam"].mean()) * 255) for o in obs]


def test_v3_first_frame_follows_meta_not_data_row_position(tmp_path):
    root = tmp_path / "ds"
    _build_gray_dataset(root, lengths=[4, 4, 4])  # global frames 0..11; ep starts 0,4,8

    # Aligned dataset: first frames read correctly.
    assert _first_frame_grays(root) == pytest.approx([0, 80, 160], abs=8)

    # Misalign: delete two interior frames of episode 0 from the DATA parquet only
    # (video untouched). Now the data-row position of later episodes no longer equals
    # their true video frame number; meta's from_timestamp remains authoritative.
    dpath = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))[0]
    df = pd.read_parquet(dpath)
    drop = df[(df["episode_index"] == 0) & (df["frame_index"].isin([1, 2]))].index
    df.drop(drop).reset_index(drop=True).to_parquet(dpath)

    # Episode first frames are still global 0,4,8 -> gray 0,80,160. The buggy loader
    # used data-row position and returned 0,40,120 instead.
    assert _first_frame_grays(root) == pytest.approx([0, 80, 160], abs=8)
