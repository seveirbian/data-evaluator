from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .dataset import LeRobotDatasetLoader
from .embeddings import PolicyEmbeddingExtractor
from .similarity import policy_embedding_similarity


def _compute_steps(n_total: int, num_points: int) -> list[int]:
    """Return sorted unique episode counts for scaling curve x-axis.

    Uses geometric spacing so small-N region is sampled more densely.
    Always includes n_total as the last point.
    """
    raw = np.geomspace(1, n_total, num_points)
    return np.unique(raw.round().astype(int)).tolist()


class ScalingCurveGenerator:
    """Generate and plot a scaling curve of c̄_π vs training episodes.

    Eval embeddings are computed once at init and reused across all steps.
    All train embeddings are extracted once in generate(), then sliced per step.

    Usage::

        gen = ScalingCurveGenerator(
            policy_dir="policy/my_policy",
            train_data_dir="data/train/dataset",
            eval_data_dir="data/eval/dataset",
            hook_module="model.backbone",
        )
        gen.generate()
        gen.plot(save_path="curve.png", show=True)
    """

    def __init__(
        self,
        policy_dir: str,
        train_data_dir: str,
        eval_data_dir: str,
        hook_module: str,
        device: str = "auto",
        num_points: int = 20,
    ):
        self.num_points = num_points
        self.extractor = PolicyEmbeddingExtractor(policy_dir, hook_module, device)

        train_loader = LeRobotDatasetLoader(train_data_dir)
        eval_loader = LeRobotDatasetLoader(eval_data_dir)

        self.camera_keys = sorted(
            set(train_loader.camera_keys) & set(eval_loader.camera_keys)
        )
        if not self.camera_keys:
            raise ValueError(
                "No common camera keys between train and eval datasets.\n"
                f"  Train cameras: {train_loader.camera_keys}\n"
                f"  Eval  cameras: {eval_loader.camera_keys}"
            )

        self._train_obs = train_loader.get_initial_observations()
        eval_obs = eval_loader.get_initial_observations()

        print(f"Train episodes: {len(self._train_obs)}, Eval episodes: {len(eval_obs)}")
        print(f"Cameras: {self.camera_keys}")

        # Pre-compute eval embeddings once — reused for every scaling step
        self._eval_embs = self._extract_embeddings(eval_obs, "Eval embeddings")
        self._results: list[tuple[int, float]] | None = None

    def _extract_embeddings(
        self, observations: list[dict], desc: str
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings for all observations. Returns {camera_key: [N, D]}."""
        per_camera: dict[str, list[torch.Tensor]] = {k: [] for k in self.camera_keys}
        for obs in tqdm(observations, desc=desc, unit="ep"):
            embs = self.extractor.extract_per_camera(obs, self.camera_keys)
            for k in self.camera_keys:
                per_camera[k].append(embs[k])
        return {k: torch.stack(vs, dim=0) for k, vs in per_camera.items()}

    def generate(self) -> list[tuple[int, float]]:
        """Compute c̄_π for each training subset. Returns [(n_episodes, score), ...]."""
        n_total = len(self._train_obs)
        steps = _compute_steps(n_total, self.num_points)

        # Extract all train embeddings once, then slice per step
        all_train_embs = self._extract_embeddings(self._train_obs, "Train embeddings")

        self._results = []
        for n in tqdm(steps, desc="Scaling curve", unit="step"):
            train_embs_n = {k: v[:n] for k, v in all_train_embs.items()}
            score = policy_embedding_similarity(train_embs_n, self._eval_embs)
            self._results.append((n, score))

        return self._results

    def plot(self, save_path: str | None = None, show: bool = False) -> None:
        """Plot the scaling curve.

        Args:
            save_path: If given, save the figure to this path (PNG). Parent
                directories are created automatically.
            show: If True, display an interactive matplotlib window.
        """
        if self._results is None:
            raise RuntimeError(
                "请先调用 generate() 再调用 plot()。"
            )

        import matplotlib.pyplot as plt

        xs = [r[0] for r in self._results]
        ys = [r[1] for r in self._results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("c̄_π")
        ax.set_title("Policy Embedding Similarity — Scaling Curve")
        ax.grid(True, linestyle="--", alpha=0.5)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        plt.close(fig)
