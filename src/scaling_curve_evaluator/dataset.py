import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class LeRobotDatasetLoader:
    """Load LeRobot 3.0 format datasets and extract initial observations."""

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.camera_keys = [
            k
            for k, v in self.info["features"].items()
            if k.startswith("observation.images.")
            and v.get("dtype") in ("image", "video", None)
        ]

    def get_initial_observations(self) -> list[dict[str, torch.Tensor]]:
        """Return one observation per episode (frame_index == 0).

        Returns list of {camera_key: image_tensor [C, H, W]} dicts.
        """
        import pandas as pd

        observations = []
        for parquet_path in sorted(self.root.glob("data/**/*.parquet")):
            df = pd.read_parquet(parquet_path)
            initial_frames = df[df["frame_index"] == 0]

            for _, row in initial_frames.iterrows():
                obs = {}
                for cam_key in self.camera_keys:
                    if cam_key in row.index and row[cam_key] is not None:
                        obs[cam_key] = self._load_image(row[cam_key])
                if obs:
                    observations.append(obs)

        return observations

    def _load_image(self, img_data) -> torch.Tensor:
        if isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        elif isinstance(img_data, dict) and "path" in img_data:
            return self._load_video_first_frame(img_data["path"])
        raise ValueError(f"Unknown image format: {type(img_data)}")

    def _load_video_first_frame(self, rel_path: str) -> torch.Tensor:
        import av

        video_path = self.root / rel_path
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        raise ValueError(f"Could not decode frame from {video_path}")
