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
        """Return one observation dict per episode (frame_index == 0)."""
        import pandas as pd

        observations = []
        for parquet_path in sorted(self.root.glob("data/**/*.parquet")):
            df = pd.read_parquet(parquet_path)

            # frame_index may be stored as scalar (v3.0) or shape-[1] array (v2.0)
            fi_series = df["frame_index"].apply(
                lambda x: int(x[0]) if hasattr(x, "__len__") else int(x)
            )

            for frame_pos in df.index[fi_series == 0]:
                if self.version == "v2.0":
                    # Each episode has its own video; initial frame is always video frame 0
                    video_frame_idx = 0
                    video_path_fn = lambda cam, p=parquet_path: self._video_path_v2(p, cam)
                else:
                    # Multiple episodes share a video; iloc position = video frame number
                    video_frame_idx = df.index.get_loc(frame_pos)
                    video_path_fn = lambda cam, p=parquet_path: self._video_path_v3(p, cam)

                obs = {
                    cam: self._load_video_frame_at(video_path_fn(cam), video_frame_idx)
                    for cam in self.camera_keys
                }
                row = df.loc[frame_pos]
                for key in self.state_keys:
                    obs[key] = torch.tensor(row[key], dtype=torch.float32)
                observations.append(obs)

        return observations

    def _video_path_v3(self, parquet_path: Path, cam_key: str) -> Path:
        """v3.0: videos/{cam_key}/chunk-000/file-000.mp4"""
        rel = parquet_path.relative_to(self.root)
        return self.root / "videos" / cam_key / rel.parts[1] / rel.name.replace(".parquet", ".mp4")

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
