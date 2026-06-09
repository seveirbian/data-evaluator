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
    """Test OpenPIEmbeddingExtractorJAX with scaling curve.

    If train and eval datasets use different camera key names, set camera_map below:
        camera_map = {
            "<train_camera_key>": "<eval_camera_key>",
            ...
        }
    Leave as {} to auto-detect common keys (works when both sides share the same names).
    """
    from src.scaling_curve._embeddings_openpi import OpenPIEmbeddingExtractorJAX as Extractor
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    from src.scaling_curve._similarity import per_sample_scores

    # ── Camera key mapping (train key → eval key) ──────────────────────────────
    # Edit this dict when train/eval datasets use different camera names.
    camera_map: dict[str, str] = {}

    checkpoint_path = Path(checkpoint_path).expanduser()

    train_data_dir = "example/cuicuisha/data/train/Task-GrabCuicuishaPlaceMatting-30"
    eval_data_dir = "example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30"

    # --- Step 1: load datasets and validate camera mapping (fast, fail-early) ---
    print("[1/5] Loading dataset metadata ...")
    train_loader = LeRobotDatasetLoader(train_data_dir)
    eval_loader = LeRobotDatasetLoader(eval_data_dir)
    train_obs = train_loader.get_initial_observations()
    eval_obs = eval_loader.get_initial_observations()
    train_camera_keys = train_loader.camera_keys
    eval_camera_keys  = eval_loader.camera_keys
    print(f"      Train: {len(train_obs)} episodes | Eval: {len(eval_obs)} episodes")
    print(f"      Train cameras : {train_camera_keys}")
    print(f"      Eval  cameras : {eval_camera_keys}")

    # Resolve final camera_keys and validate/complete camera_map.
    if camera_map:
        missing = [k for k in camera_map if k not in train_camera_keys]
        if missing:
            raise ValueError(f"camera_map keys not found in train dataset: {missing}")
        bad_vals = [v for v in camera_map.values() if v not in eval_camera_keys]
        if bad_vals:
            raise ValueError(f"camera_map values not found in eval dataset: {bad_vals}")
        camera_keys = list(camera_map.keys())
    else:
        camera_keys = [k for k in train_camera_keys if k in eval_camera_keys]

    if not camera_keys:
        n = min(len(train_camera_keys), len(eval_camera_keys))
        suggested = {train_camera_keys[i]: eval_camera_keys[i] for i in range(n)}
        lines = [f'    "{tk}": "{ek}",' for tk, ek in suggested.items()]
        print("\n  [!] No common camera keys found between train and eval.")
        print("      Edit the camera_map dict in test_openpi_jax (suggested positional mapping):")
        print("      camera_map = {")
        for line in lines:
            print(f"      {line}")
        print("      }")
        raise SystemExit(1)

    print(f"      Active cameras: {camera_keys}")

    # --- Step 1b: visualize sample observations (verify camera mapping) ---
    print("      Saving observation preview ...")
    _N_VIS = 4  # episodes to show per dataset

    def _draw_obs_grid(sf, obs_list, keys, title, remap_fn=None):
        """Fill a SubFigure with a cameras × episodes image grid."""
        n_ep  = min(_N_VIS, len(obs_list))
        n_cam = len(keys)
        axs   = sf.subplots(n_cam, n_ep, squeeze=False)
        sf.suptitle(title, fontsize=10)
        # Spread sample indices evenly across the dataset.
        indices = [int(i * (len(obs_list) - 1) / max(n_ep - 1, 1)) for i in range(n_ep)]
        for ci, key in enumerate(keys):
            for col, ep_i in enumerate(indices):
                ax  = axs[ci][col]
                obs = remap_fn(obs_list[ep_i]) if remap_fn else obs_list[ep_i]
                if key in obs:
                    ax.imshow(obs[key].permute(1, 2, 0).numpy().clip(0, 1))
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="red")
                ax.axis("off")
                if ci == 0:
                    ax.set_title(f"ep {ep_i}", fontsize=8)
            axs[ci][0].set_ylabel(key.split(".")[-1], fontsize=8,
                                  rotation=0, ha="right", va="center", labelpad=4)

    n_cam = len(camera_keys)
    n_ep  = min(_N_VIS, len(train_obs), len(eval_obs)) or _N_VIS
    fig_out = plt.figure(figsize=(n_ep * 2.6, n_cam * 5.2))
    sf_train, sf_eval = fig_out.subfigures(2, 1, hspace=0.3)

    _draw_obs_grid(sf_train, train_obs, camera_keys, f"Train ({len(train_obs)} eps)")
    _draw_obs_grid(sf_eval,  eval_obs,  camera_keys,
                   f"Eval ({len(eval_obs)} eps) — after camera_map",
                   remap_fn=_remap_obs if camera_map else None)

    obs_preview_path = "openpi_jax_obs_preview.png"
    fig_out.savefig(obs_preview_path, dpi=120, bbox_inches="tight")
    plt.close(fig_out)
    print(f"      Saved → {obs_preview_path}  (verify camera mapping is semantically correct)")
    print("      Dataset OK.")

    # --- Step 2: load model weights (slow, only after data validation passes) ---
    print(f"[2/5] Loading model from {checkpoint_path} ...")
    extractor = Extractor(
        checkpoint_path=str(checkpoint_path),
        model_type="pi05",
        hook_module="PaliGemma.img",
        batch_size=32,
    )
    print("      Done.")

    # --- Step 3: extract embeddings (batched) ---
    bs = extractor._batch_size

    # Remap eval obs keys to match train keys when camera_map is set.
    # e.g. obs["eval_cam"] → obs["train_cam"] so extract_batch sees unified keys.
    def _remap_obs(obs: dict) -> dict:
        if not camera_map:
            return obs
        remapped = dict(obs)
        for train_key, eval_key in camera_map.items():
            if eval_key in remapped:
                remapped[train_key] = remapped.pop(eval_key)
        return remapped

    def _extract_all(obs_list, label, remap=False):
        embs = {k: [] for k in camera_keys}
        n_batches = (len(obs_list) + bs - 1) // bs
        for i in tqdm(range(0, len(obs_list), bs), desc=f"  {label}", unit="batch",
                      total=n_batches):
            batch = obs_list[i : i + bs]
            if remap:
                batch = [_remap_obs(o) for o in batch]
            for result in extractor.extract_batch(batch, camera_keys):
                for k, v in result.items():
                    embs[k].append(v)
        return embs

    print(f"[3/5] Extracting train embeddings ({len(train_obs)} eps, batch_size={bs}) ...")
    train_embs = _extract_all(train_obs, "train", remap=False)

    print(f"[3/5] Extracting eval  embeddings ({len(eval_obs)} eps, batch_size={bs}) ...")
    eval_embs = _extract_all(eval_obs, "eval ", remap=True)

    # --- Step 4: per-eval intermediate plot ---
    print("[4/5] Computing per-eval similarity and generating intermediate plot ...")
    train_embs_stacked = {k: torch.stack(v, dim=0) for k, v in train_embs.items()}
    eval_embs_stacked = {k: torch.stack(v, dim=0) for k, v in eval_embs.items()}

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
    print(f"      Per-eval plot saved to {per_eval_path}")
    plt.close(fig0)

    # --- Step 5: scaling curve ---
    print("[5/5] Computing scaling curve ...")
    n_total = len(train_obs)
    num_points = 5
    steps = np.unique(np.geomspace(1, n_total, num_points).round().astype(int))

    results = []
    for n in tqdm(steps, desc="  curve", unit="pt"):
        train_subset = {k: v[:n] for k, v in train_embs_stacked.items()}
        score = policy_embedding_similarity(train_subset, eval_embs_stacked)
        results.append((n, score))
        tqdm.write(f"      n={n:4d}  score={score:.4f}")

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
    print(f"      Scaling curve saved to {save_path}")
    print("All done.")
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
