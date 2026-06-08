import sys
from pathlib import Path
import torch
from src.scaling_curve import (
    MultiScalingCurveGenerator,
    OpenPIEmbeddingExtractor,
    OpenPIEmbeddingExtractorJAX,
)
from src.scaling_curve._dataset import LeRobotDatasetLoader
from src.scaling_curve._similarity import policy_embedding_similarity


def main():
    plotter = MultiScalingCurveGenerator(
        eval_data_dir="example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30",
        curves=[
            {
                "policy_dir": "example/cuicuisha/policy/act_policy_grabcuicuishaplacematting-30",
                "train_data_dir": "example/cuicuisha/data/train/Task-GrabCuicuishaPlaceMatting-30",
                "hook_module": "model.backbone",
            },
        ],
        device="auto",
        num_points=5,
    )
    plotter.generate_all()
    plotter.plot(save_path="cuicuisha-30.png", show=True)
    print("Done. Saved to cuicuisha-30.png")


def test_openpi(checkpoint_path: str = "~/.cache/openpi/pi05_pytorch"):
    """Test OpenPIEmbeddingExtractor (PyTorch) with scaling curve."""
    from src.scaling_curve._embeddings_openpi import OpenPIEmbeddingExtractor as Extractor

    checkpoint_path = Path(checkpoint_path).expanduser()

    # Manually create scaling curve with OpenPI
    train_data_dir = "example/cuicuisha/data/train/Task-GrabCuicuishaPlaceMatting-30"
    eval_data_dir = "example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30"

    # Initialize OpenPI extractor
    extractor = Extractor(
        checkpoint_path=str(checkpoint_path),
        model_type="pi05",
        hook_module="paligemma_with_expert.paligemma.vision_tower",
        device="auto",
    )

    # Load datasets
    train_loader = LeRobotDatasetLoader(train_data_dir)
    eval_loader = LeRobotDatasetLoader(eval_data_dir)

    train_obs = train_loader.get_initial_observations()
    eval_obs = eval_loader.get_initial_observations()

    camera_keys = train_loader.camera_keys

    # Extract embeddings
    print(f"Extracting embeddings for {len(train_obs)} train episodes...")
    train_embs = {k: [] for k in camera_keys}
    for obs in train_obs:
        embs = extractor.extract_per_camera(obs, camera_keys)
        for k, v in embs.items():
            train_embs[k].append(v)

    print(f"Extracting embeddings for {len(eval_obs)} eval episodes...")
    eval_embs = {k: [] for k in camera_keys}
    for obs in eval_obs:
        embs = extractor.extract_per_camera(obs, camera_keys)
        for k, v in embs.items():
            eval_embs[k].append(v)

    # Stack embeddings
    train_embs_stacked = {k: torch.stack(v, dim=0) for k, v in train_embs.items()}
    eval_embs_stacked = {k: torch.stack(v, dim=0) for k, v in eval_embs.items()}

    # Generate scaling curve
    import numpy as np
    import matplotlib.pyplot as plt

    n_total = len(train_obs)
    num_points = 5
    steps = np.unique(np.geomspace(1, n_total, num_points).round().astype(int))

    results = []
    for n in steps:
        train_subset = {k: v[:n] for k, v in train_embs_stacked.items()}
        score = policy_embedding_similarity(train_subset, eval_embs_stacked)
        results.append((n, score))
        print(f"Steps: {n}, Score: {score:.4f}")

    # Plot
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=5)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("c̄_π")
    ax.set_title("OpenPI Scaling Curve (PyTorch)")
    ax.grid(True, linestyle="--", alpha=0.5)

    save_path = "openpi_pytorch_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Done. Saved to {save_path}")
    plt.show()


def test_openpi_jax(checkpoint_path: str = "/root/codes/openpi/pi05_base/pi05_base"):
    """Test OpenPIEmbeddingExtractorJAX with scaling curve."""
    from src.scaling_curve._embeddings_openpi import OpenPIEmbeddingExtractorJAX as Extractor

    checkpoint_path = Path(checkpoint_path).expanduser()

    # Same as PyTorch version but using JAX extractor
    train_data_dir = "example/cuicuisha/data/train/Task-GrabCuicuishaPlaceMatting-30"
    eval_data_dir = "example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30"

    # Initialize JAX extractor
    extractor = Extractor(
        checkpoint_path=str(checkpoint_path),
        model_type="pi05",
        hook_module="PaliGemma.img",
    )

    # Load datasets
    train_loader = LeRobotDatasetLoader(train_data_dir)
    eval_loader = LeRobotDatasetLoader(eval_data_dir)

    train_obs = train_loader.get_initial_observations()
    eval_obs = eval_loader.get_initial_observations()

    camera_keys = train_loader.camera_keys

    # Extract embeddings
    print(f"Extracting embeddings for {len(train_obs)} train episodes...")
    train_embs = {k: [] for k in camera_keys}
    for obs in train_obs:
        embs = extractor.extract_per_camera(obs, camera_keys)
        for k, v in embs.items():
            train_embs[k].append(v)

    print(f"Extracting embeddings for {len(eval_obs)} eval episodes...")
    eval_embs = {k: [] for k in camera_keys}
    for obs in eval_obs:
        embs = extractor.extract_per_camera(obs, camera_keys)
        for k, v in embs.items():
            eval_embs[k].append(v)

    # Stack embeddings
    train_embs_stacked = {k: torch.stack(v, dim=0) for k, v in train_embs.items()}
    eval_embs_stacked = {k: torch.stack(v, dim=0) for k, v in eval_embs.items()}

    # Intermediate plot: per-eval-episode similarity to full train set
    import numpy as np
    import matplotlib.pyplot as plt
    from src.scaling_curve._similarity import per_sample_scores

    eval_scores = per_sample_scores(train_embs_stacked, eval_embs_stacked).numpy()
    eval_ids = list(range(len(eval_scores)))

    margin = max((eval_scores.max() - eval_scores.min()) * 0.5, 0.002)
    ylo = max(0.0, eval_scores.min() - margin)
    yhi = min(1.0, eval_scores.max() + margin)

    fig0, ax0 = plt.subplots(figsize=(max(6, len(eval_scores) * 0.5), 4))
    bars = ax0.bar(eval_ids, eval_scores - ylo, color="steelblue", alpha=0.8, bottom=ylo)
    ax0.axhline(eval_scores.mean(), color="crimson", linestyle="--", linewidth=1.2,
                label=f"mean={eval_scores.mean():.4f}")
    for bar, score in zip(bars, eval_scores):
        ax0.text(bar.get_x() + bar.get_width() / 2, score + margin * 0.1,
                 f"{score:.4f}", ha="center", va="bottom", fontsize=7)
    ax0.set_xlabel("Eval episode ID")
    ax0.set_ylabel("Cosine similarity (raw)")
    ax0.set_title("Per-eval-episode similarity to full train set")
    ax0.set_xticks(eval_ids)
    ax0.set_ylim(ylo, yhi + margin)
    ax0.legend()
    ax0.grid(axis="y", linestyle="--", alpha=0.4)

    per_eval_path = "openpi_jax_per_eval.png"
    fig0.savefig(per_eval_path, dpi=150, bbox_inches="tight")
    print(f"Per-eval plot saved to {per_eval_path}")
    plt.close(fig0)

    # Generate scaling curve
    n_total = len(train_obs)
    num_points = 5
    steps = np.unique(np.geomspace(1, n_total, num_points).round().astype(int))

    results = []
    for n in steps:
        train_subset = {k: v[:n] for k, v in train_embs_stacked.items()}
        score = policy_embedding_similarity(train_subset, eval_embs_stacked)
        results.append((n, score))
        print(f"Steps: {n}, Score: {score:.4f}")

    # Plot
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=5)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("c̄_π")
    ax.set_title("OpenPI Scaling Curve (JAX)")
    ax.grid(True, linestyle="--", alpha=0.5)

    save_path = "openpi_jax_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Done. Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--openpi":
            test_openpi(sys.argv[2] if len(sys.argv) > 2 else None)
        elif sys.argv[1] == "--openpi-jax":
            test_openpi_jax(sys.argv[2] if len(sys.argv) > 2 else "/root/codes/openpi/pi05_base/pi05_base")
        else:
            print("Usage: python main.py [--openpi <checkpoint_path>] [--openpi-jax <checkpoint_path>]")
    else:
        main()
