import json
from pathlib import Path

import numpy as np
import torch


class LeRobotDatasetLoader:
    """Load LeRobot v3.0 format datasets and extract initial observations.

    In v3.0, images are stored as video files (not in parquet). Parquet files
    contain only state/action data. The parquet row position (iloc) maps 1:1
    to the video frame number in the corresponding video file.

    Directory layout::

        {root}/
          meta/info.json
          data/chunk-000/file-000.parquet   <- state/action rows
          videos/{cam_key}/chunk-000/file-000.mp4  <- frames (same order as parquet)
    """

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.camera_keys = [
            k
            for k, v in self.info["features"].items()
            if v.get("dtype") == "video"
        ]

    def get_initial_observations(self) -> list[dict[str, torch.Tensor]]:
        """Return one observation dict per episode (frame_index==0).

        Each dict contains image tensors keyed by camera name, plus any
        non-video features (e.g. observation.state) as tensors.
        """
        import pandas as pd

        state_keys = [
            k
            for k, v in self.info["features"].items()
            if k.startswith("observation.") and v.get("dtype") != "video" and k in self._parquet_columns()
        ]

        observations = []

        for parquet_path in sorted(self.root.glob("data/**/*.parquet")):
            df = pd.read_parquet(parquet_path)

            # iloc positions of initial frames = video frame numbers
            initial_positions = np.where(df["frame_index"].values == 0)[0]
            if len(initial_positions) == 0:
                continue

            for frame_pos in initial_positions:
                obs = {}

                # Images from video files
                for cam_key in self.camera_keys:
                    video_path = self._video_path(parquet_path, cam_key)
                    obs[cam_key] = self._load_video_frame_at(video_path, int(frame_pos))

                # State from parquet
                row = df.iloc[frame_pos]
                for key in state_keys:
                    obs[key] = torch.tensor(row[key], dtype=torch.float32)

                observations.append(obs)

        return observations

    def _parquet_columns(self) -> set[str]:
        """Return column names from the first parquet file."""
        import pandas as pd

        for parquet_path in self.root.glob("data/**/*.parquet"):
            return set(pd.read_parquet(parquet_path, columns=None).columns)
        return set()

    def _video_path(self, parquet_path: Path, cam_key: str) -> Path:
        """Convert data/chunk-000/file-000.parquet -> videos/{cam_key}/chunk-000/file-000.mp4"""
        rel = parquet_path.relative_to(self.root)
        # rel.parts: ('data', 'chunk-000', 'file-000.parquet')
        video_rel = Path("videos") / cam_key / rel.parts[1] / rel.name.replace(".parquet", ".mp4")
        return self.root / video_rel

    def _load_video_frame_at(self, video_path: Path, frame_idx: int) -> torch.Tensor:
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            if frame_idx > 0:
                fps = self.info["fps"]
                seek_us = int(frame_idx / fps * 1_000_000)
                container.seek(seek_us)

            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        raise ValueError(f"Frame {frame_idx} not found in {video_path}")
