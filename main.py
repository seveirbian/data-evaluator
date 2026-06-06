import sys
from src.scaling_curve import MultiScalingCurveGenerator, OpenPIEmbeddingExtractor
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


def test_openpi(
    checkpoint_path: str,
    eval_data_dir: str = "example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30",
    model_type: str = "pi05",
):
    """Test OpenPIEmbeddingExtractor with example dataset.

    Args:
        checkpoint_path: Path to openpi checkpoint directory (e.g., downloaded from openpi)
        eval_data_dir: Path to evaluation dataset
        model_type: Model type, one of "pi05", "pi0"
    """
    print(f"Testing OpenPI with {model_type} model...")

    # Initialize the openpi extractor
    extractor = OpenPIEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        hook_module="paligemma_with_expert.paligemma.vision_tower",
        device="auto",
    )

    # Load eval dataset
    eval_loader = LeRobotDatasetLoader(eval_data_dir)
    eval_obs = eval_loader.get_initial_observations()

    print(f"Loaded {len(eval_obs)} eval episodes")
    print(f"Camera keys: {eval_loader.camera_keys}")

    # Extract embeddings for first 5 observations as a test
    embeddings_list = []
    for i, obs in enumerate(eval_obs[:5]):
        print(f"Extracting embedding for episode {i+1}/5...")
        embeddings = extractor.extract_per_camera(obs, eval_loader.camera_keys)
        embeddings_list.append(embeddings)

    print(f"\nSuccessfully extracted embeddings!")
    print(f"Embedding shape for first camera: {list(embeddings_list[0].values())[0].shape}")

    # Compute similarity between first two observations as a demo
    if len(embeddings_list) >= 2:
        emb1 = {k: v.unsqueeze(0) for k, v in embeddings_list[0].items()}
        emb2 = {k: v.unsqueeze(0) for k, v in embeddings_list[1].items()}
        sim = policy_embedding_similarity(emb1, emb2)
        print(f"Similarity between first two episodes: {sim:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--openpi":
        if len(sys.argv) < 3:
            print("Usage: python main.py --openpi <checkpoint_path> [eval_data_dir] [model_type]")
            print("Example: python main.py --openpi ~/.cache/openpi/pi05 example/cuicuisha/data/eval/... pi05")
            sys.exit(1)
        checkpoint_path = sys.argv[2]
        eval_data_dir = sys.argv[3] if len(sys.argv) > 3 else "example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30"
        model_type = sys.argv[4] if len(sys.argv) > 4 else "pi05"
        test_openpi(checkpoint_path, eval_data_dir, model_type)
    else:
        main()
