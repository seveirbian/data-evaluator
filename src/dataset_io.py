import json
from pathlib import Path

import torch


class LeRobotDatasetLoader:
    """Load LeRobot datasets (v2.0 and v3.0) and extract per-episode initial observations.

    v3.0 layout::
        data/chunk-000/file-000.parquet          # multiple episodes per file
        videos/{cam_key}/chunk-000/file-000.mp4  # shared video per chunk

    v2.0 layout::
        data/chunk-000/episode_000000.parquet    # one episode per file
        videos/chunk-000/{cam_key}/episode_000000.mp4
    """

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.version = self.info.get("codebase_version", "v3.0")
        features = self.info["features"]
        self.camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        self.state_keys = [
            k for k, v in features.items()
            if k.startswith("observation.") and v.get("dtype") != "video"
        ]

    def get_initial_observations(self) -> list[dict[str, torch.Tensor]]:
        """Return one observation dict per episode (frame_index == 0), in episode order."""
        if self.version == "v2.0":
            return self._initial_observations_v2()
        return self._initial_observations_v3()

    def _first_frame_states(self) -> dict[int, "pd.Series"]:
        """Map episode_index -> the frame_index==0 data row (for state features)."""
        import pandas as pd

        states: dict[int, pd.Series] = {}
        for parquet_path in sorted(self.root.glob("data/**/*.parquet")):
            df = pd.read_parquet(parquet_path)
            fi = df["frame_index"].apply(
                lambda x: int(x[0]) if hasattr(x, "__len__") else int(x)
            )
            for _, row in df[fi == 0].iterrows():
                ei = row["episode_index"]
                ei = int(ei[0]) if hasattr(ei, "__len__") else int(ei)
                states[ei] = row
        return states

    def _initial_observations_v3(self) -> list[dict[str, torch.Tensor]]:
        """v3.0: locate each camera's first frame via meta/episodes.

        Multiple episodes share a video file, and each camera is partitioned and
        timestamped independently. The authoritative location of an episode's first
        frame in a given camera's video is ``meta/episodes`` -> ``videos/<cam>/
        {chunk_index,file_index}`` (which file) + ``from_timestamp`` (where in it).
        Deriving it from the data-parquet row position instead is wrong whenever the
        video partitioning differs from the data partitioning or between cameras.
        """
        import pandas as pd

        ep_meta = pd.concat(
            [pd.read_parquet(p) for p in sorted(self.root.glob("meta/episodes/**/*.parquet"))],
            ignore_index=True,
        ).sort_values("episode_index")

        video_template = self.info["video_path"]
        states = self._first_frame_states()

        ep_order = [int(em["episode_index"]) for _, em in ep_meta.iterrows()]
        obs_by_ep: dict[int, dict[str, torch.Tensor]] = {ei: {} for ei in ep_order}

        # Group episodes by (camera, video file) so each shared video file is opened
        # ONCE — re-opening a large shared file per episode dominates wall-time for
        # many-episode datasets.
        for cam in self.camera_keys:
            by_file: dict[str, list[tuple[float, int]]] = {}
            for _, em in ep_meta.iterrows():
                ei = int(em["episode_index"])
                video_path = self.root / video_template.format(
                    video_key=cam,
                    chunk_index=int(em[f"videos/{cam}/chunk_index"]),
                    file_index=int(em[f"videos/{cam}/file_index"]),
                )
                ts = float(em[f"videos/{cam}/from_timestamp"])
                by_file.setdefault(str(video_path), []).append((ts, ei))

            for video_path, items in by_file.items():
                items.sort()  # ascending timestamp
                frames = self._decode_frames_at_timestamps(
                    Path(video_path), [ts for ts, _ in items]
                )
                for (_, ei), frame in zip(items, frames):
                    obs_by_ep[ei][cam] = frame

        observations = []
        for ei in ep_order:
            obs = obs_by_ep[ei]
            row = states[ei]
            for key in self.state_keys:
                obs[key] = torch.tensor(row[key], dtype=torch.float32)
            observations.append(obs)

        return observations

    def _initial_observations_v2(self) -> list[dict[str, torch.Tensor]]:
        """v2.0: one video per episode; the initial frame is always video frame 0."""
        import pandas as pd

        observations = []
        for parquet_path in sorted(self.root.glob("data/**/*.parquet")):
            df = pd.read_parquet(parquet_path)
            fi = df["frame_index"].apply(
                lambda x: int(x[0]) if hasattr(x, "__len__") else int(x)
            )
            for frame_pos in df.index[fi == 0]:
                obs = {
                    cam: self._load_video_frame_at(self._video_path_v2(parquet_path, cam), 0)
                    for cam in self.camera_keys
                }
                row = df.loc[frame_pos]
                for key in self.state_keys:
                    obs[key] = torch.tensor(row[key], dtype=torch.float32)
                observations.append(obs)

        return observations

    def _video_path_v2(self, parquet_path: Path, cam_key: str) -> Path:
        """v2.0: videos/chunk-000/{cam_key}/episode_000000.mp4"""
        rel = parquet_path.relative_to(self.root)
        chunk_dir = rel.parts[1]       # e.g. "chunk-000"
        episode_name = parquet_path.stem  # e.g. "episode_000000"
        return self.root / "videos" / chunk_dir / cam_key / f"{episode_name}.mp4"

    def _load_video_frame_at(self, video_path: Path, frame_idx: int) -> torch.Tensor:
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if frame_idx > 0:
                container.seek(int(frame_idx / self.info["fps"] * 1_000_000))
            for frame in container.decode(stream):
                return torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255.0

        raise ValueError(f"Frame {frame_idx} not found in {video_path}")

    def _decode_frames_at_timestamps(
        self, video_path: Path, timestamps: list[float]
    ) -> list[torch.Tensor]:
        """Decode the exact frame at each timestamp, opening the file only once.

        For every timestamp: seek to the keyframe at or before it, then decode
        forward to the first frame whose presentation time reaches the timestamp.
        Seeking directly to the target (rather than an arbitrary margin before it)
        decodes the minimum number of frames; the returned frame is identical
        either way since the decode-forward endpoint is fixed by the timestamp.
        """
        import av

        frames: list[torch.Tensor] = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for timestamp in timestamps:
                container.seek(int(max(0.0, timestamp) * 1_000_000))
                last = None
                for frame in container.decode(stream):
                    last = frame
                    if frame.time is not None and frame.time + 1e-4 >= timestamp:
                        break
                if last is None:
                    raise ValueError(f"No frame at t={timestamp}s in {video_path}")
                frames.append(
                    torch.from_numpy(last.to_ndarray(format="rgb24"))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
        return frames
