"""Equations 9-11 from the paper.

Eq. 9:  c_π(x_i, x_j) = cosine_similarity(φ(x_i), φ(x_j))
Eq. 10: c_π(x_i, D_train) = max_{x_j ∈ D_train} c_π(x_i, x_j)
        Multi-camera extension: max over cameras after computing per-camera scores.
Eq. 11: c̄_π = mean_{x_i ∈ D_eval} c(x_i, D_train)   [after normalizing to [0,1]]

Correct usage (paper-aligned):
    sim = compute_sim_matrix(train_embs, eval_embs)        # [N_eval, N_train], once
    c_min, c_max = sim_norm_range(sim)                     # global range from full set
    scores = per_sample_scores(sim, c_min, c_max)          # bar chart
    for n in steps:
        c_bar = policy_embedding_similarity(sim, n, c_min, c_max)  # curve point
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


def compute_sim_matrix(
    train_embeddings: dict[str, torch.Tensor],
    eval_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute [N_eval, N_train] similarity matrix with max-over-cameras fusion.

    This is the core computation that should be done ONCE against the full
    training set. The resulting matrix is then sliced per-n for the scaling curve.

    Args:
        train_embeddings: {camera_key: [N_train, D]}
        eval_embeddings:  {camera_key: [N_eval, D]}

    Returns:
        [N_eval, N_train] cosine similarity matrix (max over cameras).
    """
    camera_keys = list(train_embeddings.keys())
    if not camera_keys:
        raise ValueError("No camera embeddings provided.")

    per_camera = torch.stack(
        [
            _cosine_similarity_matrix(eval_embeddings[k], train_embeddings[k])
            for k in camera_keys
        ],
        dim=0,
    )  # [N_cams, N_eval, N_train]
    return per_camera.max(dim=0).values  # [N_eval, N_train]


def sim_norm_range(sim_matrix: torch.Tensor) -> tuple[float, float]:
    """Derive global normalization range from the full-set similarity matrix.

    Must be called with the FULL training set matrix (all N_train columns) so
    the range is stable and consistent across all scaling-curve subset calls.

    Returns:
        (c_min, c_max) scalars derived from per-eval max similarities.
    """
    full_scores = sim_matrix.max(dim=1).values  # [N_eval]
    return full_scores.min().item(), full_scores.max().item()


def per_sample_scores(
    sim_matrix: torch.Tensor,
    c_min: float,
    c_max: float,
) -> torch.Tensor:
    """Eq. 10 + normalize: per-eval max similarity normalized to [0, 1].

    Args:
        sim_matrix: [N_eval, N_train] (full training set).
        c_min, c_max: from sim_norm_range(sim_matrix).

    Returns:
        [N_eval] scores in [0, 1].
    """
    raw = sim_matrix.max(dim=1).values  # [N_eval]
    denom = c_max - c_min
    if denom < 1e-8:
        return torch.ones_like(raw)
    return (raw - c_min) / denom


def policy_embedding_similarity(
    sim_matrix: torch.Tensor,
    n: int,
    c_min: float,
    c_max: float,
) -> float:
    """Eq. 11: c̄_π for first n training episodes, normalized by global range.

    Args:
        sim_matrix: [N_eval, N_train] full matrix from compute_sim_matrix().
        n: number of training episodes to consider (uses first n columns).
        c_min, c_max: global normalization range from sim_norm_range().

    Returns:
        Scalar c̄_π ∈ [0, 1].
    """
    scores_n = sim_matrix[:, :n].max(dim=1).values  # [N_eval]
    denom = c_max - c_min
    if denom < 1e-8:
        return 1.0
    normalized = (scores_n - c_min) / denom
    return normalized.mean().item()


def top_k_train_matches(
    sim_matrix: torch.Tensor,
    c_min: float,
    c_max: float,
    k: int = 5,
) -> list[dict]:
    """Return the top-k most similar train episodes for each eval episode.

    Args:
        sim_matrix: [N_eval, N_train] from compute_sim_matrix().
        c_min, c_max: from sim_norm_range(); used to normalize scores to [0, 1].
        k: number of train matches to keep per eval episode; auto-clamped to N_train.

    Returns:
        list[dict], each item shaped
        {"eval_id": int, "top_k": [{"train_id": int, "score": float}, ...]}.
        Outer list is sorted by eval_id ascending; inner top_k is sorted by score
        descending with length min(k, N_train). Scores are clipped to [0, 1]
        (ranks below the per-eval max may otherwise fall below c_min).
    """
    n_train = sim_matrix.shape[1]
    k = min(k, n_train)
    denom = c_max - c_min

    top_vals, top_idx = torch.topk(sim_matrix, k, dim=1)  # 各 [N_eval, k]

    result: list[dict] = []
    for eval_id in range(sim_matrix.shape[0]):
        matches = []
        for rank in range(k):
            raw = top_vals[eval_id, rank].item()
            score = (
                1.0 if denom < 1e-8 else min(1.0, max(0.0, (raw - c_min) / denom))
            )
            matches.append(
                {"train_id": int(top_idx[eval_id, rank].item()), "score": score}
            )
        result.append({"eval_id": eval_id, "top_k": matches})
    return result
