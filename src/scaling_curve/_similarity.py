"""Equations 9-11 from the paper.

Eq. 9:  c_π(x_i, x_j) = cosine_similarity(φ(x_i), φ(x_j))
Eq. 10: c_π(x_i, D_train) = max_{x_j ∈ D_train} c_π(x_i, x_j)
        Multi-camera extension: max over cameras after computing per-camera scores.
Eq. 11: c̄_π = mean_{x_i ∈ D_eval} c(x_i, D_train)   [after normalizing to [0,1]]
"""

import torch
import torch.nn.functional as F


def _cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Eq. 9 applied to all pairs.

    Args:
        a: [N, D]
        b: [M, D]

    Returns:
        [N, M] cosine similarity matrix.
    """
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return a @ b.T


def _per_sample_max_similarity(
    eval_embs: torch.Tensor, train_embs: torch.Tensor
) -> torch.Tensor:
    """Eq. 10: for each eval sample, take max cosine similarity over all train samples.

    Args:
        eval_embs:  [N_eval, D]
        train_embs: [N_train, D]

    Returns:
        [N_eval] scores in [-1, 1].
    """
    sim = _cosine_similarity_matrix(eval_embs, train_embs)  # [N_eval, N_train]
    return sim.max(dim=1).values


def per_sample_scores(
    train_embeddings: dict[str, torch.Tensor],
    eval_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return per-eval-episode similarity scores (before taking the mean).

    Returns:
        [N_eval] scores normalized to [0, 1].
    """
    camera_keys = list(train_embeddings.keys())
    if not camera_keys:
        raise ValueError("No camera embeddings provided.")

    per_camera = torch.stack(
        [
            _per_sample_max_similarity(eval_embeddings[k], train_embeddings[k])
            for k in camera_keys
        ],
        dim=1,
    )
    return per_camera.max(dim=1).values  # [N_eval], raw cosine similarity in [-1, 1]


def policy_embedding_similarity(
    train_embeddings: dict[str, torch.Tensor],
    eval_embeddings: dict[str, torch.Tensor],
) -> float:
    """Compute c̄_π (Eq. 11) with per-camera independent similarity + max fusion.

    For each eval sample x_i:
        1. Per camera k: score_k = max_{x_j ∈ D_train} cosine_sim(φ^k(x_i), φ^k(x_j))
        2. Fuse cameras:  score_i = max_k score_k
    Then normalize all score_i to [0, 1] and return the mean.

    Args:
        train_embeddings: {camera_key: [N_train, D]}
        eval_embeddings:  {camera_key: [N_eval, D]}

    Returns:
        Scalar c̄_π ∈ [-1, 1].
    """
    camera_keys = list(train_embeddings.keys())
    if not camera_keys:
        raise ValueError("No camera embeddings provided.")

    # [N_eval, N_cams] per-camera max-similarity scores
    per_camera_scores = torch.stack(
        [
            _per_sample_max_similarity(eval_embeddings[k], train_embeddings[k])
            for k in camera_keys
        ],
        dim=1,
    )

    # Eq. 10 multi-camera: max over cameras
    per_sample_scores = per_camera_scores.max(dim=1).values  # [N_eval]

    # Eq. 11: mean over eval set
    return per_sample_scores.mean().item()
