import json
from pathlib import Path

import torch


class LeRobotDatasetLoader:
    """Load LeRobot v3.0 datasets and extract per-episode initial observations.

    In v3.0, images are stored as video files; parquet files contain state/action.
    Parquet row position (iloc) maps 1:1 to video frame number.

    Layout::

        {root}/
          meta/info.json
          data/chunk-000/file-000.parquet
          videos/{cam_key}/chunk-000/file-000.mp4
    """

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
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
            for frame_pos in df.index[df["frame_index"] == 0]:
                # frame_pos is the label index; get_loc converts it to the
                # 0-based position within this file, which equals the video frame number.
                video_frame_idx = df.index.get_loc(frame_pos)
                obs = {
                    cam: self._load_video_frame_at(self._video_path(parquet_path, cam), video_frame_idx)
                    for cam in self.camera_keys
                }
                row = df.loc[frame_pos]
                for key in self.state_keys:
                    obs[key] = torch.tensor(row[key], dtype=torch.float32)
                observations.append(obs)

        return observations

    def _video_path(self, parquet_path: Path, cam_key: str) -> Path:
        rel = parquet_path.relative_to(self.root)
        return self.root / "videos" / cam_key / rel.parts[1] / rel.name.replace(".parquet", ".mp4")

    def _load_video_frame_at(self, video_path: Path, frame_idx: int) -> torch.Tensor:
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if frame_idx > 0:
                container.seek(int(frame_idx / self.info["fps"] * 1_000_000))
            for frame in container.decode(stream):
                return torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255.0

        raise ValueError(f"Frame {frame_idx} not found in {video_path}")
