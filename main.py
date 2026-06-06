import sys
from src.scaling_curve import (
    MultiScalingCurveGenerator,
    OpenPIEmbeddingExtractor,
    OpenPIEmbeddingExtractorJAX,
)
from src.scaling_curve._dataset import LeRobotDatasetLoader


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
    """Test OpenPIEmbeddingExtractor (PyTorch) with example dataset."""
    extractor = OpenPIEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        model_type="pi05",
        hook_module="paligemma_with_expert.paligemma.vision_tower",
        device="auto",
    )

    eval_loader = LeRobotDatasetLoader("example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30")
    eval_obs = eval_loader.get_initial_observations()

    for obs in eval_obs[:5]:
        embeddings = extractor.extract_per_camera(obs, eval_loader.camera_keys)
        print(f"Extracted: {list(embeddings.keys())}, shape: {list(embeddings.values())[0].shape}")


def test_openpi_jax(checkpoint_path: str = "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid"):
    """Test OpenPIEmbeddingExtractorJAX with example dataset."""
    extractor = OpenPIEmbeddingExtractorJAX(
        checkpoint_path=checkpoint_path,
        model_type="pi05",
        hook_module="PaliGemma.img",
    )

    eval_loader = LeRobotDatasetLoader("example/cuicuisha/data/eval/eval_act_Task-GrabCuicuishaPlaceMatting-30")
    eval_obs = eval_loader.get_initial_observations()

    for obs in eval_obs[:5]:
        embeddings = extractor.extract_per_camera(obs, eval_loader.camera_keys)
        print(f"Extracted: {list(embeddings.keys())}, shape: {list(embeddings.values())[0].shape}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--openpi":
            test_openpi(sys.argv[2] if len(sys.argv) > 2 else None)
        elif sys.argv[1] == "--openpi-jax":
            test_openpi_jax(sys.argv[2] if len(sys.argv) > 2 else None)
        else:
            print("Usage: python main.py [--openpi <checkpoint_path>] [--openpi-jax <checkpoint_path>]")
    else:
        main()
