from scaling_curve_evaluator import PolicyEmbeddingSimilarityEvaluator


def main():
    evaluator = PolicyEmbeddingSimilarityEvaluator(
        policy_dir="policy/",
        train_data_dir="data/train/",
        eval_data_dir="data/eval/",
        hook_module="model.backbone",  # ACT: "model.backbone", Diffusion: "model.obs_encoder"
        device="auto",
    )
    score = evaluator.evaluate()
    print(f"Policy Embedding Similarity c̄_π = {score:.4f}")


if __name__ == "__main__":
    main()
