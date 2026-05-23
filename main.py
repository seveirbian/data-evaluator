from scaling_curve_evaluator import PolicyEmbeddingSimilarityEvaluator


def main():
    evaluator = PolicyEmbeddingSimilarityEvaluator(
        policy_dir="example/policy/act_policy_grabcuicuishaplacematting-30",
        train_data_dir="example/data/train/Task-GrabCuicuishaPlaceMatting-30",
        eval_data_dir="example/data/eval/Task-GrabCuicuishaPlaceMatting-30",
        hook_module="model.backbone",  # ACT: "model.backbone", Diffusion: "model.obs_encoder"
        device="auto",
    )
    score = evaluator.evaluate()
    print(f"Policy Embedding Similarity c̄_π = {score:.4f}")


if __name__ == "__main__":
    main()
