from src.scaling_curve_evaluator import ScalingCurveGenerator


def main():
    gen = ScalingCurveGenerator(
        policy_dir="example/policy/act_policy_grabcuicuishaplacematting-30",
        train_data_dir="example/data/train/Task-GrabCuicuishaPlaceMatting-30",
        eval_data_dir="example/data/eval/Task-GrabCuicuishaPlaceMatting-30",
        hook_module="model.backbone",
        device="auto",
        num_points=20,
    )
    gen.generate()
    gen.plot(save_path="scaling_curve.png", show=False)
    print("Done. Saved to scaling_curve.png")


if __name__ == "__main__":
    main()
