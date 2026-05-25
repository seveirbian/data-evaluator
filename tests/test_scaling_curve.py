from src.scaling_curve_evaluator.scaling_curve import _compute_steps


def test_compute_steps_bounds():
    steps = _compute_steps(n_total=30, num_points=10)
    assert steps[0] == 1
    assert steps[-1] == 30
    assert all(isinstance(s, int) for s in steps)
    assert len(steps) <= 10


def test_compute_steps_deduplication():
    # 数据量少时去重后应不超过 n_total 个点
    steps = _compute_steps(n_total=3, num_points=20)
    assert steps == [1, 2, 3]


import torch
from unittest.mock import patch
from src.scaling_curve_evaluator.scaling_curve import ScalingCurveGenerator


def _make_obs(n: int) -> list[dict]:
    return [{"cam": torch.zeros(3, 4, 4)} for _ in range(n)]


@patch("src.scaling_curve_evaluator.scaling_curve.PolicyEmbeddingExtractor")
@patch("src.scaling_curve_evaluator.scaling_curve.LeRobotDatasetLoader")
def test_generate_returns_correct_structure(MockLoader, MockExtractor):
    # train: 5 episodes, eval: 2 episodes
    MockLoader.return_value.camera_keys = ["cam"]
    MockLoader.return_value.get_initial_observations.side_effect = [
        _make_obs(5),  # train
        _make_obs(2),  # eval
    ]
    MockExtractor.return_value.extract_per_camera.return_value = {
        "cam": torch.zeros(128)
    }

    gen = ScalingCurveGenerator("p", "t", "e", "m", num_points=3)
    results = gen.generate()

    assert isinstance(results, list)
    assert len(results) >= 1
    n, score = results[0]
    assert isinstance(n, int)
    assert isinstance(score, float)
    assert results[-1][0] == 5  # 最后一点必须是 n_total


import pytest


def _generator_with_results(results):
    """Bypass __init__ and inject results directly."""
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = results
    return gen


def test_plot_raises_if_generate_not_called():
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = None
    with pytest.raises(RuntimeError, match="generate"):
        gen.plot()


def test_plot_saves_file(tmp_path):
    gen = _generator_with_results([(1, 0.4), (5, 0.7), (10, 1.0)])
    save_path = tmp_path / "subdir" / "curve.png"
    gen.plot(save_path=str(save_path))
    assert save_path.exists()


from src.scaling_curve_evaluator.scaling_curve import _infer_labels


def test_infer_labels_no_collision():
    curves = [
        {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "m"},
        {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch2", "hook_module": "m"},
    ]
    labels = _infer_labels(curves)
    assert labels == ["batch1", "batch2"]


def test_infer_labels_collision():
    curves = [
        {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "m"},
        {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch1", "hook_module": "m"},
    ]
    labels = _infer_labels(curves)
    assert labels == ["batch1/act", "batch1/pi0"]


from src.scaling_curve_evaluator.scaling_curve import MultiScalingCurveGenerator


def test_multi_plotter_empty_curves_raises():
    with pytest.raises(ValueError, match="curves"):
        MultiScalingCurveGenerator(eval_data_dir="e", curves=[])


@patch("src.scaling_curve_evaluator.scaling_curve.ScalingCurveGenerator")
def test_generate_all_calls_each_generator(MockGen):
    MockGen.return_value.generate.return_value = [(1, 0.5), (5, 0.9)]
    plotter = MultiScalingCurveGenerator(
        eval_data_dir="e",
        curves=[
            {"policy_dir": "p1", "train_data_dir": "t1", "hook_module": "m"},
            {"policy_dir": "p2", "train_data_dir": "t2", "hook_module": "m"},
        ],
    )
    plotter.generate_all()
    assert MockGen.return_value.generate.call_count == 2


def _multi_plotter_with_results(results_list, labels):
    """Bypass __init__, inject generators with results directly."""
    plotter = object.__new__(MultiScalingCurveGenerator)
    plotter._labels = labels
    plotter._generators = []
    for results in results_list:
        gen = object.__new__(ScalingCurveGenerator)
        gen._results = results
        plotter._generators.append(gen)
    return plotter


def test_multi_plot_raises_if_generate_not_called():
    plotter = object.__new__(MultiScalingCurveGenerator)
    plotter._labels = ["a"]
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = None
    plotter._generators = [gen]
    with pytest.raises(RuntimeError, match="generate"):
        plotter.plot()


def test_multi_plot_saves_file(tmp_path):
    plotter = _multi_plotter_with_results(
        results_list=[
            [(1, 0.4), (5, 0.7), (10, 1.0)],
            [(1, 0.3), (5, 0.6), (10, 0.9)],
        ],
        labels=["batch1", "batch2"],
    )
    save_path = tmp_path / "multi_curve.png"
    plotter.plot(save_path=str(save_path))
    assert save_path.exists()
