from src.scaling_curve import MultiScalingCurveGenerator


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
    print("Done. Saved to scaling_curve.png")


if __name__ == "__main__":
    main()
