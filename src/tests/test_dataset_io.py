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


def _reencode_single_keyframe(video_path: Path):
    """Re-encode a video so only frame 0 is a keyframe (every other frame mid-GOP).

    Decodes the existing frames and re-encodes them as fresh ndarray frames with a
    huge GOP and no B-frames, so later episode-start frames are NOT on a keyframe.
    """
    import av

    inp = av.open(str(video_path))
    ins = inp.streams.video[0]
    arrays = [fr.to_ndarray(format="rgb24") for fr in inp.decode(ins)]
    w, h = ins.width, ins.height
    inp.close()

    tmp = str(video_path) + ".tmp.mp4"
    out = av.open(tmp, "w")
    stream = out.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.codec_context.gop_size = 1000
    stream.codec_context.max_b_frames = 0
    for arr in arrays:
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(frame):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()
    import shutil

    shutil.move(tmp, str(video_path))


def test_v3_first_frame_correct_when_episode_start_not_keyframe(tmp_path):
    """Universality guard: episode-start frames that are NOT keyframes must still
    decode to the exact frame (seek to keyframe <= from_timestamp, decode forward)."""
    import av

    root = tmp_path / "ds"
    _build_gray_dataset(root, lengths=[4, 4, 4])  # ep starts at global frames 0,4,8
    vf = sorted(glob.glob(str(root / "videos/**/*.mp4"), recursive=True))[0]
    _reencode_single_keyframe(Path(vf))

    # sanity: only frame 0 is a keyframe, so episode starts 4 and 8 are mid-GOP
    c = av.open(vf)
    kf = [i for i, fr in enumerate(c.decode(c.streams.video[0])) if fr.key_frame]
    c.close()
    assert kf == [0], f"fixture not mid-GOP: keyframes={kf}"

    # Despite starts 4,8 being non-keyframes, the loader returns the exact frames.
    assert _first_frame_grays(root) == pytest.approx([0, 80, 160], abs=8)


def test_progress_flag_does_not_change_result(tmp_path):
    import torch

    root = tmp_path / "ds"
    _build_gray_dataset(root, lengths=[2, 2, 2])
    ld = LeRobotDatasetLoader(root)

    with_bar = ld.get_initial_observations(progress=True)
    without_bar = ld.get_initial_observations(progress=False)

    assert len(with_bar) == len(without_bar) == 3
    for a, b in zip(with_bar, without_bar):
        assert torch.equal(a["observation.images.cam"], b["observation.images.cam"])


def test_parallel_decode_matches_sequential(tmp_path):
    import torch

    root = tmp_path / "ds"
    _build_gray_dataset(root, lengths=[3, 2, 4, 1, 2])  # 5 episodes
    ld = LeRobotDatasetLoader(root)

    sequential = ld.get_initial_observations(num_workers=1, progress=False)
    parallel = ld.get_initial_observations(num_workers=4, progress=False)

    assert len(sequential) == len(parallel) == 5
    # Episode order and frame content must be identical regardless of worker count.
    for a, b in zip(sequential, parallel):
        assert torch.equal(a["observation.images.cam"], b["observation.images.cam"])
        assert torch.equal(a["observation.state"], b["observation.state"])
