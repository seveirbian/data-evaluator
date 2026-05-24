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
